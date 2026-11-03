"""CNN fusion front end with trainable ACT or conditional DDPM action heads."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .policies import ACT, DiffusionPolicy


class VisualPolicy(nn.Module):
    def __init__(self, kind: str, use_proprio: bool, history: int = 2, horizon: int = 4):
        super().__init__()
        if kind not in ("act", "diffusion") or history < 1:
            raise ValueError("Unsupported visual policy configuration")
        self.kind, self.use_proprio, self.history, self.horizon = kind, use_proprio, history, horizon
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proprio_encoder = nn.Sequential(nn.Linear(history * 10, 64), nn.ReLU()) if use_proprio else None
        fused = history * 2 * 64 + (64 if use_proprio else 0)
        self.fusion = nn.Sequential(nn.Linear(fused, 128), nn.ReLU(), nn.Linear(128, 64))
        self.core = (ACT(64, 4, horizon, hidden=64, latent=16, layers=2)
                     if kind == "act" else DiffusionPolicy(64, 4, horizon, hidden=128, timesteps=20))

    def encode(self, images: Tensor, proprio: Tensor | None = None) -> Tensor:
        if images.ndim != 6 or images.shape[1:] != (self.history, 2, 3, 64, 64):
            raise ValueError("Expected [batch,history,2,3,64,64] RGB tensor")
        batch = len(images)
        pixels = images.reshape(batch * self.history * 2, 3, 64, 64).float() / 255.0
        visual = self.image_encoder(pixels).reshape(batch, -1)
        if self.use_proprio:
            if proprio is None or proprio.shape != (batch, self.history, 10):
                raise ValueError("Robot proprioception shape mismatch")
            assert self.proprio_encoder is not None
            visual = torch.cat([visual, self.proprio_encoder(proprio.flatten(1))], -1)
        return self.fusion(visual)

    def loss(self, images: Tensor, proprio: Tensor | None, actions: Tensor) -> Tensor:
        state = self.encode(images, proprio)
        return self.core.loss(state, actions)

    @torch.no_grad()
    def act(self, images: Tensor, proprio: Tensor | None = None) -> Tensor:
        self.eval()
        return self.core.act(self.encode(images, proprio))
