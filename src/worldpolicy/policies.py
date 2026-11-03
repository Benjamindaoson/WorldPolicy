"""Trainable ACT-style CVAE transformer and conditional DDPM action policies."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ACT(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 2, horizon: int = 4,
                 hidden: int = 64, latent: int = 16, layers: int = 2, heads: int = 4,
                 prior_training_fraction: float = .5):
        super().__init__()
        self.state_dim, self.action_dim, self.horizon = state_dim, action_dim, horizon
        if not 0 <= prior_training_fraction <= 1:
            raise ValueError("prior_training_fraction must be in [0, 1]")
        self.prior_training_fraction = prior_training_fraction
        self.state_encoder = nn.Linear(state_dim, hidden)
        self.action_encoder = nn.Linear(action_dim, hidden)
        self.action_position = nn.Parameter(torch.randn(horizon, hidden) * .02)
        self.latent_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True), layers)
        self.posterior = nn.Linear(hidden, latent * 2)
        self.latent_project = nn.Linear(latent, hidden)
        self.query = nn.Parameter(torch.randn(horizon, hidden) * .02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden, heads, hidden * 2, batch_first=True), layers)
        self.head = nn.Linear(hidden, action_dim)

    def forward(self, state: Tensor, actions: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError("ACT state dimension mismatch")
        memory = self.state_encoder(state).unsqueeze(1)
        kl = state.new_zeros(())
        if actions is not None:
            if actions.shape != (len(state), self.horizon, self.action_dim):
                raise ValueError("ACT action chunk dimension mismatch")
            encoded = self.latent_encoder(self.action_encoder(actions) + self.action_position)
            mu, logvar = self.posterior(encoded.mean(1)).chunk(2, -1)
            z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar)
            if self.training and self.prior_training_fraction:
                prior_mask = torch.rand((len(state), 1), device=state.device) < self.prior_training_fraction
                z = torch.where(prior_mask, torch.zeros_like(z), z)
            kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
        else:
            z = state.new_zeros((len(state), self.latent_project.in_features))
        memory = memory + self.latent_project(z).unsqueeze(1)
        query = self.query.unsqueeze(0).expand(len(state), -1, -1)
        prediction = self.head(self.decoder(query, memory)).tanh()
        return prediction, kl

    def loss(self, state: Tensor, actions: Tensor, kl_weight: float = .001) -> Tensor:
        prediction, kl = self(state, actions)
        return F.mse_loss(prediction, actions) + kl_weight * kl

    @torch.no_grad()
    def act(self, state: Tensor) -> Tensor:
        self.eval()
        return self(state)[0][:, 0]


class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 2, horizon: int = 4,
                 hidden: int = 128, timesteps: int = 20):
        super().__init__()
        self.state_dim, self.action_dim, self.horizon = state_dim, action_dim, horizon
        self.timesteps = timesteps
        # Cosine schedule reaches near-isotropic noise even for short smoke runs.
        grid = torch.linspace(0, 1, timesteps + 1)
        cumulative = torch.cos((grid + .008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-4, .999)
        self.register_buffer("alpha", 1 - beta)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))
        self.net = nn.Sequential(nn.Linear(state_dim + horizon * action_dim + 32, hidden),
                                 nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, horizon * action_dim))

    def _time_embed(self, t: Tensor) -> Tensor:
        frequencies = torch.exp(torch.arange(16, device=t.device) * (-math.log(10000) / 15))
        phase = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat([phase.sin(), phase.cos()], dim=1)

    def predict_noise(self, state: Tensor, noisy: Tensor, t: Tensor) -> Tensor:
        if state.shape[-1] != self.state_dim or noisy.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("Diffusion input shape mismatch")
        features = torch.cat([state, noisy.flatten(1), self._time_embed(t)], dim=1)
        return self.net(features).view_as(noisy)

    def loss(self, state: Tensor, actions: Tensor) -> Tensor:
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("Diffusion action shape mismatch")
        t = torch.randint(self.timesteps, (len(state),), device=state.device)
        noise = torch.randn_like(actions)
        a = self.alpha_bar[t].view(-1, 1, 1)
        noisy = a.sqrt() * actions + (1 - a).sqrt() * noise
        return F.mse_loss(self.predict_noise(state, noisy, t), noise)

    @torch.no_grad()
    def sample(self, state: Tensor) -> Tensor:
        self.eval()
        x = torch.randn(len(state), self.horizon, self.action_dim, device=state.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((len(state),), step, device=state.device, dtype=torch.long)
            alpha, abar = self.alpha[step], self.alpha_bar[step]
            x = (x - (1 - alpha) / (1 - abar).sqrt() * self.predict_noise(state, x, t)) / alpha.sqrt()
            if step:
                previous_abar = self.alpha_bar[step - 1]
                posterior_variance = (1 - previous_abar) / (1 - abar) * (1 - alpha)
                x += posterior_variance.sqrt() * torch.randn_like(x)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def act(self, state: Tensor) -> Tensor:
        return self.sample(state)[:, 0]
