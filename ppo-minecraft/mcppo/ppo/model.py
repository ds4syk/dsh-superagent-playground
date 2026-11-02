"""Recurrent actor-critic for PPO on Minecraft pixels + inventory."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """Welford-style running mean/std (used to normalise rewards)."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim else 1
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)


def _orthogonal(layer: nn.Module, gain: float) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int, inv_dim: int, equipped_dim: int,
                 frame_channels: int = 12, hidden_size: int = 256,
                 feature_size: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.frame_channels = frame_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # 64x64 -> 15x15 -> 6x6 -> 4x4
        self.cnn_out = 64 * 4 * 4
        self.cnn_fc = nn.Sequential(nn.Linear(self.cnn_out, feature_size), nn.ReLU())
        self.inv_fc = nn.Sequential(
            nn.Linear(inv_dim + equipped_dim + 3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.lstm = nn.LSTM(feature_size + 128, hidden_size)
        self.pi = nn.Linear(hidden_size, n_actions)
        self.v = nn.Linear(hidden_size, 1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                _orthogonal(module, np.sqrt(2))
        _orthogonal(self.pi, 0.01)
        _orthogonal(self.v, 1.0)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.shape[0]
                param.data[n // 4:n // 2].fill_(1.0)  # forget gate
            else:
                nn.init.orthogonal_(param, 1.0)

    def initial_hidden(self, batch: int, device: torch.device):
        h = torch.zeros(1, batch, self.hidden_size, device=device)
        c = torch.zeros(1, batch, self.hidden_size, device=device)
        return h, c

    def forward(
        self,
        image: torch.Tensor,
        inventory: torch.Tensor,
        scalars: torch.Tensor,
        equipped: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reset_mask: Optional[torch.Tensor] = None,
    ):
        """Shapes: image (T,B,C,H,W); inventory (T,B,I); scalars (T,B,3);
        equipped (T,B,E).  Returns logits (T,B,A), value (T,B), hidden."""
        t, b = image.shape[:2]
        x = image.reshape(t * b, *image.shape[2:])
        x = self.cnn_fc(self.cnn(x))
        y = torch.cat([inventory.reshape(t * b, -1),
                       scalars.reshape(t * b, -1),
                       equipped.reshape(t * b, -1)], dim=-1)
        y = self.inv_fc(y)
        z = torch.cat([x, y], dim=-1).reshape(t, b, -1)
        if reset_mask is None:
            out, hidden = self.lstm(z, hidden)
        else:
            # slow path used only for PPO updates: reset (h, c) after a done
            h, c = hidden
            outs = []
            for i in range(t):
                o, (h, c) = self.lstm(z[i:i + 1], (h, c))
                outs.append(o)
                m = reset_mask[i].view(1, b, 1).to(o.dtype)
                h = h * (1.0 - m)
                c = c * (1.0 - m)
            out = torch.cat(outs, dim=0)
            hidden = (h, c)
        return self.pi(out), self.v(out).squeeze(-1), hidden
