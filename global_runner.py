from __future__ import annotations

import multiprocessing as mp
import queue as pyqueue
import time
from typing import Optional

import numpy as np
import torch

from agent_utils import (
    load_weights,
    make_agent,
    model_weights_on_cpu,
    resolve_state_size,
)
from config import AgentConfig, Config, GlobalConfig, TrainingConfig
from game import SnakeGame, WIDTH, HEIGHT
from live_plot import LivePlot


def actor_process(
    actor_id: int,
    state_size: int,
    agent_cfg: AgentConfig,
    global_cfg: GlobalConfig,
    tx_queue: mp.Queue,
    rx_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    torch.set_num_threads(1)
    agent = make_agent(agent_cfg, state_size)
    pending: list = []
    steps = 0

    max_actor_steps = global_cfg.max_actor_steps
    while not stop_event.is_set() and (max_actor_steps is None or steps < max_actor_steps):
        game = SnakeGame(w=WIDTH, h=HEIGHT, render=False, title=f"Actor {actor_id}")
        state = game.get_state()
        episode_steps = 0
        score = 0

        while True:
            try:
                weights = rx_queue.get_nowait()
                load_weights(agent, weights)
            except pyqueue.Empty:
                pass

            action = agent.act(state)
            reward, done, score = game.step(action)
            next_state = game.get_state()

            pending.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1
            episode_steps += 1

            if len(pending) >= global_cfg.actor_batch:
                tx_queue.put(("batch", pending))
                pending = []

            limit_reached = max_actor_steps is not None and steps >= max_actor_steps
            if done or limit_reached:
                try:
                    metrics_queue.put_nowait(
                        (
                            "episode",
                            {
                                "actor": actor_id,
                                "score": score,
                                "steps": episode_steps,
                                "epsilon": agent.current_epsilon(),
                            },
                        )
                    )
                except pyqueue.Full:
                    pass
                agent.on_game_end()
                break

        if stop_event.is_set() or (max_actor_steps is not None and steps >= max_actor_steps):
            break

    if pending:
        tx_queue.put(("batch", pending))

    tx_queue.put(("done", actor_id))


def learner_process(
    state_size: int,
    agent_cfg: AgentConfig,
    global_cfg: GlobalConfig,
    rx_queue: mp.Queue,
    weight_queues: list,
    metrics_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    agent = make_agent(agent_cfg, state_size)
    updates = 0

    while not stop_event.is_set():
        try:
            kind, payload = rx_queue.get(timeout=0.5)
        except pyqueue.Empty:
            continue

        if kind == "batch":
            transitions = payload
            for transition in transitions:
                agent.remember(transition)
            agent.advance_schedule(len(transitions))

            if agent.buffer_size() >= agent.batch_size:
                batch_loss = agent.train_long_memory()
                if batch_loss is not None:
                    updates += 1
                    try:
                        metrics_queue.put_nowait(
                            (
                                "update",
                                {
                                    "updates": updates,
                                    "loss": float(batch_loss),
                                    "buffer": agent.buffer_size(),
                                    "lr": agent.current_lr(),
                                    "per_beta": agent.current_per_beta(),
                                },
                            )
                        )
                    except pyqueue.Full:
                        pass

                    if updates % global_cfg.broadcast_every == 0:
                        weights = model_weights_on_cpu(agent)
                        for queue_out in weight_queues:
                            queue_out.put(weights)

        elif kind == "done":
            actor_id = payload
            print(f"[global] aktor {actor_id} zgłosił zakończenie.")

    weights = model_weights_on_cpu(agent)
    for queue_out in weight_queues:
        queue_out.put(weights)


def run_global(cfg: Config) -> None:
    state_size = resolve_state_size(cfg)
    mp.set_start_method("spawn", force=True)

    stop_event = mp.Event()
    rx_queue: mp.Queue = mp.Queue(maxsize=128)
    weight_queues = [mp.Queue(maxsize=4) for _ in range(cfg.global_cfg.num_actors)]
    metrics_queue: mp.Queue = mp.Queue(maxsize=256)

    learner = mp.Process(
        target=learner_process,
        args=(
            state_size,
            cfg.agent,
            cfg.global_cfg,
            rx_queue,
            weight_queues,
            metrics_queue,
            stop_event,
        ),
        daemon=True,
        name="global_learner",
    )
    learner.start()

    actors = []
    for actor_id in range(cfg.global_cfg.num_actors):
        actor = mp.Process(
            target=actor_process,
            args=(
                actor_id,
                state_size,
                cfg.agent,
                cfg.global_cfg,
                rx_queue,
                weight_queues[actor_id],
                metrics_queue,
                stop_event,
            ),
            daemon=True,
            name=f"actor_{actor_id}",
        )
        actor.start()
        actors.append(actor)

    plotter = LivePlot(enabled=cfg.plot.enabled, smoothing_window=cfg.plot.smoothing_window)
    training_cfg = cfg.training
    train_start = time.time()
    stop_reason: Optional[str] = None
    scores: list[int] = []
    last_loss = 0.0
    last_lr = cfg.agent.lr_start
    last_epsilon = cfg.agent.start_epsilon
    best_score = float("-inf")
    episode_counter = 0
    actor_eps: dict[int, float] = {}
    last_per_beta = cfg.agent.per_beta_start if cfg.agent.use_per else 0.0

    def request_stop(reason: str) -> None:
        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason
            stop_event.set()

    def handle_metric(kind: str, payload: dict) -> None:
        nonlocal episode_counter, last_loss, last_lr, last_epsilon, best_score, last_per_beta
        if kind == "episode":
            episode_counter += 1
            scores.append(payload["score"])
            best_score = max(best_score, payload["score"])
            actor_eps[payload["actor"]] = payload.get("epsilon", last_epsilon)
            mean_eps = float(np.mean(list(actor_eps.values()))) if actor_eps else last_epsilon
            last_epsilon = mean_eps
            avg50 = float(np.mean(scores[-50:])) if scores else 0.0
            avg200 = float(np.mean(scores[-200:])) if scores else 0.0
            if cfg.plot.enabled:
                plotter.update(
                    episode=episode_counter,
                    avg_points=avg50,
                    avg200=avg200,
                    loss=last_loss,
                    epsilon=mean_eps,
                    lr=last_lr,
                    per_beta=last_per_beta,
                )
            print(
                f"[global][ep {episode_counter:05d}] actor={payload['actor']} score={payload['score']} best={best_score} steps={payload['steps']} "
                f"eps={last_epsilon:.3f} loss={last_loss:.4f} lr={last_lr:.6f}"
            )
            target_score = training_cfg.target_score
            if target_score is not None and best_score >= target_score:
                request_stop(f"osiągnięto score {best_score} ≥ cel {target_score}")
            target_avg = cfg.training.target_avg50
            if target_avg is not None and avg50 >= target_avg:
                request_stop(f"osiągnięto avg50={avg50:.2f} ≥ cel {target_avg}")
        elif kind == "update":
            last_loss = payload.get("loss", last_loss)
            last_lr = payload.get("lr", last_lr)
            last_per_beta = payload.get("per_beta", last_per_beta)

    try:
        still_running = True
        while still_running:
            try:
                kind, payload = metrics_queue.get(timeout=0.5)
                handle_metric(kind, payload)
            except pyqueue.Empty:
                pass

            if (
                training_cfg.max_train_seconds is not None
                and time.time() - train_start >= training_cfg.max_train_seconds
            ):
                request_stop(f"limit czasu {training_cfg.max_train_seconds}s")

            still_running = any(actor.is_alive() for actor in actors)
            if not still_running:
                break

        for actor in actors:
            actor.join(timeout=0.5)
    finally:
        stop_event.set()
        for actor in actors:
            if actor.is_alive():
                actor.terminate()
        if learner.is_alive():
            learner.terminate()
        learner.join(timeout=1.0)

    while True:
        try:
            kind, payload = metrics_queue.get_nowait()
            handle_metric(kind, payload)
        except pyqueue.Empty:
            break

    plotter.block()
    if stop_reason:
        print(f"[global][stop] {stop_reason}")
    else:
        print("[global][stop] zakończono wszystkie procesy aktorów.")
