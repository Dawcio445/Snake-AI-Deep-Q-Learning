import random
from typing import List, Tuple, Any


class SumTree:
    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, "capacity must be a power of 2"
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data: List[Any] = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = idx // 2
        self.tree[parent] += change
        if parent > 1:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx
        right = left + 1
        if left >= 2 * self.capacity:
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[1]

    def add(self, p: float, data: Any):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 200_000):
        cap = 1
        while cap < capacity:
            cap <<= 1
        self.capacity = cap

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, beta_frames)
        self.frame = 1
        self._last_beta = float(beta_start)

        self.tree = SumTree(self.capacity)
        self.eps = 1e-6

    def __len__(self):
        return self.tree.size

    def _beta(self) -> float:
        t = min(self.frame / self.beta_frames, 1.0)
        value = self.beta_start + t * (1.0 - self.beta_start)
        self._last_beta = value
        return value

    def current_beta(self) -> float:
        return float(self._last_beta)

    def add(self, transition: Tuple, priority: float = 1.0):
        p = (abs(priority) + self.eps) ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size: int):
        assert self.tree.size > 0, "Buffer is empty"
        self.frame += 1

        beta = self._beta()
        segment = self.tree.total / batch_size

        samples = []
        indices = []
        weights = []

        min_prob = (min(self.tree.tree[self.capacity:self.capacity + self.tree.size]) / self.tree.total)

        for i in range(batch_size):
            s = random.random() * segment + i * segment
            idx, p, data = self.tree.get(s)
            prob = p / self.tree.total
            w = (prob / min_prob) ** (-beta)
            samples.append(data)
            indices.append(idx)
            weights.append(w)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (list(states), list(actions), list(rewards), list(next_states), list(dones)), indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, pr in zip(indices, priorities):
            p = (abs(pr) + self.eps) ** self.alpha
            self.tree.update(idx, p)
