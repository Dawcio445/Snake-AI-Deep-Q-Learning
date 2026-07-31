from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes, output_size: int):
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        elif isinstance(hidden_sizes, list):
            hidden_sizes = tuple(hidden_sizes)

        self.hidden_sizes = hidden_sizes
        self.use_conv = False
        channels = 3
        grid_elems = input_size // channels
        grid_side = math.isqrt(grid_elems)

        if grid_side * grid_side * channels == input_size:
            self.use_conv = True
            self.grid_h = grid_side
            self.grid_w = grid_side

        if self.use_conv:
            conv_mid = 32
            conv_out = 64
            self.conv1 = nn.Conv2d(channels, conv_mid, 3, padding=1)
            self.conv2 = nn.Conv2d(conv_mid, conv_out, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            pooled_h = max(1, self.grid_h // 2)
            pooled_w = max(1, self.grid_w // 2)
            in_dim = conv_out * pooled_h * pooled_w
        else:
            in_dim = input_size

        fc_layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            fc_layers.append(nn.Linear(last_dim, h))
            last_dim = h
        self.fc_layers = nn.ModuleList(fc_layers)

        self.v_out = nn.Linear(last_dim, 1)
        self.a_out = nn.Linear(last_dim, output_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            bsz = x.shape[0]
            if x.dim() == 2:
                x = x.view(bsz, 3, self.grid_h, self.grid_w)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(bsz, -1)

        for layer in self.fc_layers:
            x = F.relu(layer(x))

        v = self.v_out(x)
        a = self.a_out(x)
        return v + (a - a.mean(dim=1, keepdim=True))


class DQNTrainer:
    def __init__(
        self,
        input_size: int,
        output_size: int = 3,
        hidden_sizes=(1024, 512, 256),
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        grad_clip: Optional[float] = 5.0,
    ):
        self.model = DuelingQNet(input_size, hidden_sizes, output_size)
        self.target_model = copy.deepcopy(self.model).eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.model.train()

    @torch.no_grad()
    def _soft_update(self) -> None:
        for target_param, source_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.lerp_(source_param.data, self.tau)

    def train_step(self, state, action, reward, next_state, done, is_weights=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1)

        if is_weights is not None:
            weights = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device).view(-1)
        else:
            weights = None

        q_values = self.model(state)
        q_sa = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_state).argmax(dim=1)
            next_q_target = self.target_model(next_state)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q = next_q.view(-1)
            gamma_val = float(self.gamma) if not isinstance(self.gamma, float) else self.gamma
            target = reward + (1.0 - done) * gamma_val * next_q

        td_error = target - q_sa
        loss = F.smooth_l1_loss(q_sa, target, reduction="none")
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            for target_param, source_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.lerp_(source_param.data, self.tau)

        return float(loss.item()), td_error.abs().detach().cpu().numpy()

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def act(self, state, epsilon: float = 0.0) -> int:
        self.model.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(s)
            greedy = int(q.argmax(dim=1).item())
        self.model.train()
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, q.shape[1], (1,), device=self.device).item()
        return greedy

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "target": self.target_model.state_dict(),
                "opt": self.optimizer.state_dict(),
                "cfg": {"gamma": self.gamma, "tau": self.tau, "grad_clip": self.grad_clip},
            },
            path,
        )

    def load(self, path: str, map_location=None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt.get("target", ckpt["model"]))
        self.optimizer.load_state_dict(ckpt["opt"])
