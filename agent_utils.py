from __future__ import annotations

from agent import Agent
from config import AgentConfig, Config
from game import SnakeGame


def resolve_state_size(cfg: Config) -> int:
    if cfg.state_size is not None:
        return cfg.state_size
    probe = SnakeGame(render=False)
    cfg.state_size = len(probe.get_state())
    return cfg.state_size


def make_agent(agent_cfg: AgentConfig, state_size: int) -> Agent:
    agent = Agent(
        state_size=state_size,
        action_size=3,
        hidden_sizes=agent_cfg.hidden_sizes,
        gamma=agent_cfg.gamma,
        lr_start=agent_cfg.lr_start,
        lr_end=agent_cfg.lr_end,
        lr_decay_steps=agent_cfg.lr_decay_steps,
        tau=agent_cfg.tau,
        replay_capacity=agent_cfg.replay_capacity,
        batch_size=agent_cfg.batch_size,
        start_epsilon=agent_cfg.start_epsilon,
        end_epsilon=agent_cfg.end_epsilon,
        epsilon_decay_steps=agent_cfg.epsilon_decay_steps,
        use_per=agent_cfg.use_per,
        per_alpha=agent_cfg.per_alpha,
        per_beta_start=agent_cfg.per_beta_start,
        per_beta_frames=agent_cfg.per_beta_frames,
        grad_clip=agent_cfg.grad_clip,
    )

    agent.min_replay_size = max(1, agent_cfg.min_replay_size)
    agent.train_every = max(1, agent_cfg.train_every)
    return agent


def model_weights_on_cpu(agent: Agent) -> dict:
    state_dict = agent.trainer.model.state_dict()
    return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}


def load_weights(agent: Agent, state_dict: dict) -> None:
    agent.trainer.model.load_state_dict(state_dict)
    agent.trainer.target_model.load_state_dict(state_dict)
