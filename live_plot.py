from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt


@dataclass
class _Series:
    label: str
    color: str
    ylabel: str
    title: str


class LivePlot:
    def __init__(self, enabled: bool = True, smoothing_window: int = 10):
        self.enabled = enabled
        self.smoothing_window = max(1, smoothing_window)
        self.history: Dict[str, List[float]] = {
            "episode": [],
            "avg50": [],
            "avg200": [],
            "loss": [],
            "epsilon": [],
            "lr": [],
            "per_beta": [],
        }

        self._series: Dict[str, _Series] = {
            "avg50": _Series("avg 50", "tab:blue", "points", "Average score (50)"),
            "avg200": _Series("avg 200", "tab:cyan", "points", "Average score (200)"),
            "loss": _Series("loss", "tab:red", "loss", "Loss"),
            "epsilon": _Series("epsilon", "tab:orange", "epsilon", "Epsilon"),
            "lr": _Series("learning rate", "tab:green", "lr", "Learning rate"),
            "per_beta": _Series("per beta", "tab:purple", "beta", "PER beta"),
        }

        if self.enabled:
            plt.ion()
            self.fig, self.axs = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
        else:
            self.fig = None
            self.axs = None

    def update(
        self,
        episode: int,
        avg_points: float,
        avg200: float,
        loss: float,
        epsilon: float,
        lr: float,
        per_beta: float,
    ) -> None:
        if not self.enabled:
            return

        self.history["episode"].append(float(episode))
        self.history["avg50"].append(float(avg_points))
        self.history["avg200"].append(float(avg200))
        self.history["loss"].append(float(loss))
        self.history["epsilon"].append(float(epsilon))
        self.history["lr"].append(float(lr))
        self.history["per_beta"].append(float(per_beta))

        xs = self.history["episode"]
        panels = [
            (self.axs[0, 0], "avg50"),
            (self.axs[0, 1], "avg200"),
            (self.axs[1, 0], "epsilon"),
            (self.axs[1, 1], "loss"),
            (self.axs[2, 0], "lr"),
            (self.axs[2, 1], "per_beta"),
        ]

        for ax, key in panels:
            ax.clear()
            series = self._series[key]
            ys = self._smooth(self.history[key])
            ax.plot(xs, ys, label=series.label, color=series.color)
            ax.set_title(series.title)
            ax.set_ylabel(series.ylabel)
            ax.legend(loc="best")

        self.axs[2, 0].set_xlabel("Episode")
        self.axs[2, 1].set_xlabel("Episode")
        self.fig.tight_layout()
        plt.pause(0.001)

    def block(self) -> None:
        if not self.enabled:
            return
        plt.ioff()
        plt.show()

    def _smooth(self, values: List[float]) -> List[float]:
        if self.smoothing_window <= 1 or len(values) <= 1:
            return values
        smoothed: List[float] = []
        for idx in range(len(values)):
            start = max(0, idx - self.smoothing_window + 1)
            window = values[start : idx + 1]
            smoothed.append(sum(window) / len(window))
        return smoothed


if __name__ == "__main__":
    import math
    import time
    import random

    plot = LivePlot(enabled=True, smoothing_window=5)
    for ep in range(1, 201):
        plot.update(
            episode=ep,
            avg_points=20 + math.sin(ep / 10) * 5,
            avg200=18 + math.sin(ep / 30) * 3,
            loss=max(0.0, 1.0 - ep / 200) + random.random() * 0.05,
            epsilon=max(0.01, math.cos(ep / 40) * 0.5 + 0.5),
            lr=5e-4,
            per_beta=0.4,
        )
        time.sleep(0.02)
    plot.block()
