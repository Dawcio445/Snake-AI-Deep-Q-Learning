from __future__ import annotations

import time
from typing import Optional

import numpy as np

from agent_utils import make_agent, resolve_state_size
from config import Config
from game import SnakeGame, WIDTH, HEIGHT
from live_plot import LivePlot


def train(cfg: Config) -> None:
    agent_cfg = cfg.agent
    train_cfg = cfg.training
    plot_cfg = cfg.plot

    state_size = resolve_state_size(cfg)
    agent = make_agent(agent_cfg, state_size)

    scores: list[int] = []
    best_avg50 = float("-inf")
    best_score = float("-inf")

    plotter = LivePlot(enabled=plot_cfg.enabled, smoothing_window=plot_cfg.smoothing_window)
    stop_reason: Optional[str] = None
    train_start = time.time()

    episode = 0
    max_steps_limit = train_cfg.max_steps_per_episode
    while True:
        if train_cfg.max_train_seconds is not None and time.time() - train_start >= train_cfg.max_train_seconds:
            stop_reason = f"limit czasu {train_cfg.max_train_seconds}s"
            break

        if train_cfg.episodes is not None and episode >= train_cfg.episodes:
            stop_reason = f"osiągnięto limit {train_cfg.episodes} epizodów"
            break

        episode += 1

        render_flag = train_cfg.render_every > 0 and (episode % train_cfg.render_every == 0)
        game = SnakeGame(
            w=WIDTH,
            h=HEIGHT,
            render=render_flag,
            title=f"Snake AI | ep {episode}",
            render_fps=train_cfg.render_fps,
        )
        state = game.get_state()

        episode_steps = 0
        episode_loss_sum = 0.0
        episode_loss_updates = 0
        episode_start = time.time()
        while True:
            action = agent.act(state)

            reward, done, score = game.step(action)
            next_state = game.get_state()

            agent.remember((state, action, reward, next_state, done))

            if agent.total_steps % agent.train_every == 0:
                if agent.buffer_size() >= agent.min_replay_size:
                    loss = agent.train_long_memory()
                    if loss is not None:
                        episode_loss_sum += loss
                        episode_loss_updates += 1

            state = next_state
            episode_steps += 1

            if render_flag:
                game.render(f"ep={episode} sc={score}")

            if done or (max_steps_limit is not None and episode_steps >= max_steps_limit):
                agent.on_game_end()
                break

        episode_time = time.time() - episode_start
        fps = episode_steps / episode_time if episode_time > 0 else 0.0
        current_eps = agent.current_epsilon()
        current_lr = agent.current_lr()
        current_beta = agent.current_per_beta()
        scores.append(score)
        loss_mean = episode_loss_sum / max(1, episode_loss_updates)

        avg50 = float(np.mean(scores[-50:])) if scores else 0.0
        avg200 = float(np.mean(scores[-200:])) if scores else 0.0
        avg_all = float(np.mean(scores)) if scores else 0.0
        best_avg50 = max(best_avg50, avg50)
        best_score = max(best_score, score)

        if plot_cfg.enabled and (episode % max(1, plot_cfg.update_every) == 0):
            plotter.update(
                episode=episode,
                avg_points=avg50,
                avg200=avg200,
                loss=loss_mean,
                epsilon=current_eps,
                lr=current_lr,
                per_beta=current_beta,
            )

        print(
            f"[ep {episode:04d}] score={score} avg50={avg50:.2f} avg200={avg200:.2f} "
            f"loss={loss_mean:.5f} eps={current_eps:.4f} lr={current_lr:.6f} "
            f"steps={episode_steps} fps={fps:.1f}"
        )

        if train_cfg.target_score is not None and best_score >= train_cfg.target_score:
            stop_reason = f"score {best_score} ≥ cel {train_cfg.target_score}"
            break

        if train_cfg.target_avg50 is not None and avg50 >= train_cfg.target_avg50:
            stop_reason = f"avg50 {avg50:.2f} ≥ cel {train_cfg.target_avg50}"
            break

        no_prog_limit = train_cfg.no_progress_stop
        if (
            no_prog_limit
            and train_cfg.target_score is None
            and episode >= no_prog_limit
            and avg50 <= train_cfg.no_progress_threshold
        ):
            stop_reason = f"avg50 {avg50:.2f} <= próg {train_cfg.no_progress_threshold} po {episode} ep"
            break

    if stop_reason:
        print(f"[stop] {stop_reason}")

    plotter.block()
