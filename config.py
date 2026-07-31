from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

BLOCK_SIZE: int = 20
WIDTH: int = 640
HEIGHT: int = 480

REWARD_EAT: float = 10.0
REWARD_DIE: float = -6.0

STEP_PENALTY: float = -0.08
DIST_COEF_PER_CELL: float = 0.12
CLOSE_BONUS_DIST: float = 2.0
CLOSE_BONUS: float = 0.2
STUCK_MULT: int = 150
MAX_STEPS_WITHOUT_FOOD: int = 1_000


@dataclass
class AgentConfig:
    hidden_sizes = (1536, 512,256)
    gamma = 0.99
    lr_start = 3e-4
    lr_end = 1e-4
    lr_decay_steps = 7_000_000
    tau = 0.01
    replay_capacity = 400_000
    batch_size = 256
    min_replay_size = 150_000
    train_every = 5
    start_epsilon = 1.0
    end_epsilon = 0.01
    epsilon_decay_steps = 500_000
    use_per = True
    per_alpha = 0.5
    per_beta_start = 0.4
    per_beta_frames = 5000
    grad_clip = 5.0


@dataclass
class TrainingConfig:
    episodes: Optional[int] = None
    max_steps_per_episode: Optional[int] = None
    render_every: int = 0
    render_fps: int = 5
    max_train_seconds: Optional[int] = None
    target_score: Optional[int] = None
    target_avg50: Optional[float] = None
    no_progress_stop: Optional[int] = None
    no_progress_threshold: float = 2.0
    gradient_update_ratio: float = 2.0


@dataclass
class PlotConfig:
    enabled: bool = True
    update_every: int = 5
    smoothing_window: int = 10


@dataclass
class GlobalConfig:
    num_actors: int = 6
    actor_batch: int = 128
    broadcast_every: int = 100
    max_actor_steps: Optional[int] = None


@dataclass
class Config:
    mode: str = "global"
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    global_cfg: GlobalConfig = field(default_factory=GlobalConfig)
    state_size: Optional[int] = None


CFG = Config()
