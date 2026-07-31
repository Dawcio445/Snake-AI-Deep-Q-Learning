import random
from collections import deque
from typing import Optional, Tuple

import numpy as np

from model import DQNTrainer

try:
    from per import PrioritizedReplayBuffer
except Exception:
    PrioritizedReplayBuffer = None


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes,
        gamma,
        lr_start,
        lr_end,
        lr_decay_steps,
        tau,
        replay_capacity,
        batch_size,
        start_epsilon,
        end_epsilon,
        epsilon_decay_steps,
        use_per,
        per_alpha,
        per_beta_start,
        per_beta_frames,
        grad_clip,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_start if lr_end is None else lr_end)
        self.lr_decay_steps = max(1, lr_decay_steps)

        self.trainer = DQNTrainer(
            input_size=state_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            lr=self.lr_start,
            gamma=gamma,
            tau=tau,
            grad_clip=grad_clip,
        )

        self.use_per = bool(use_per and PrioritizedReplayBuffer is not None)
        self.per_alpha = float(per_alpha)
        self.per_beta_start = float(per_beta_start)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=replay_capacity,
                alpha=self.per_alpha,
                beta_start=self.per_beta_start,
                beta_frames=per_beta_frames,
            )
        else:
            self.memory = deque(maxlen=replay_capacity)

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.step_count = 0
        self.games_played = 0
        self.total_steps = 0
        self.min_replay_size = 1_000

        self.train_every = 4
        self._step_since_update = 0

    def _linear_schedule(self, start: float, end: float, total_steps: int) -> float:
        alpha = min(max(self.step_count, 0) / max(1, total_steps), 1.0)
        return start + (end - start) * alpha

    def _epsilon(self) -> float:
        return self._linear_schedule(self.start_epsilon, self.end_epsilon, self.epsilon_decay_steps)

    def _lr_schedule(self) -> float:
        return self._linear_schedule(self.lr_start, self.lr_end, self.lr_decay_steps)

    def _apply_lr(self) -> None:
        self.trainer.set_lr(self._lr_schedule())

    def act(self, state: np.ndarray) -> int:
        eps = self._epsilon()
        self.step_count += 1
        self.total_steps += 1
        if random.random() < eps:
            return random.randrange(self.action_size)
        return self.trainer.act(state, epsilon=0.0)

    def current_epsilon(self) -> float:
        return self._epsilon()

    def current_lr(self) -> float:
        return self._lr_schedule()

    def current_per_alpha(self) -> float:
        if not self.use_per:
            return 0.0
        mem_alpha = getattr(self.memory, "alpha", None)
        if mem_alpha is not None:
            return float(mem_alpha)
        return float(self.per_alpha)

    def current_per_beta(self) -> float:
        if not self.use_per:
            return 0.0
        if hasattr(self.memory, "current_beta"):
            return float(self.memory.current_beta())
        return float(self.per_beta_start)

    def advance_schedule(self, steps: int) -> None:
        if steps <= 0:
            return
        self.step_count += steps
        self.total_steps += steps

    def remember(self, transition: Transition, priority: Optional[float] = None) -> None:
        if self.use_per:
            self.memory.add(transition, priority=priority if priority is not None else 1.0)
        else:
            self.memory.append(transition)

    def _sample_batch(self):
        if self.use_per:
            sample = self.memory.sample(self.batch_size)

            if isinstance(sample, tuple) and len(sample) == 7:
                states, actions, rewards, next_states, dones, idxs, weights = sample
            elif isinstance(sample, tuple) and len(sample) == 3:
                data, idxs, weights = sample
                if isinstance(data, (list, tuple)) and len(data) == 5:
                    states, actions, rewards, next_states, dones = data
                elif isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
                    states, actions, rewards, next_states, dones = zip(*data)
                else:
                    raise RuntimeError("Unexpected PER sample payload.")
            else:
                raise RuntimeError("Unexpected PER sample structure.")
        else:
            if len(self.memory) < self.batch_size:
                return None
            transitions = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*transitions)
            idxs, weights = None, None

        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)

        return states, actions, rewards, next_states, dones, idxs, weights

    def train_long_memory(self) -> Optional[float]:
        need = max(self.min_replay_size, self.batch_size)
        if len(self.memory) < need:
            return None

        self._step_since_update += 1
        if self._step_since_update % self.train_every != 0:
            return None

        batch = self._sample_batch()
        if batch is None:
            return None

        states, actions, rewards, next_states, dones, idxs, weights = batch
        self._apply_lr()
        if self.use_per:
            loss, td_err = self.trainer.train_step(
                states,
                actions,
                rewards,
                next_states,
                dones,
                is_weights=np.asarray(weights, dtype=np.float32),
            )
            self.memory.update_priorities(idxs, td_err.tolist())
        else:
            loss, _ = self.trainer.train_step(
                states,
                actions,
                rewards,
                next_states,
                dones,
                is_weights=None,
            )
        return loss
    def train_short_memory(self, state, action, reward, next_state, done):
        state = np.asarray([state], dtype=np.float32)
        next_state = np.asarray([next_state], dtype=np.float32)
        action = np.asarray([action], dtype=np.int64)
        reward = np.asarray([reward], dtype=np.float32)
        done = np.asarray([done], dtype=bool)

        self._apply_lr()

        loss, _ = self.trainer.train_step(
            state,
            action,
            reward,
            next_state,
            done,
            is_weights=None
        )

        return loss

    def on_game_end(self) -> None:
        self.games_played += 1

    def buffer_size(self) -> int:
        return len(self.memory)

    def lr(self) -> float:
        return float(self.trainer.optimizer.param_groups[0]["lr"])
