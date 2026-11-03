"""Continuous-action implicit Q-learning with expectile value and advantage BC."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def mlp(inputs: int, outputs: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, outputs))


class IQL:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 expectile: float = .7, discount: float = .99, tau: float = .005):
        self.actor = mlp(state_dim, action_dim)
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        self.value = mlp(state_dim, 1)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.v_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.expectile, self.discount, self.tau = expectile, discount, tau

    def update(self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]) -> dict:
        state, action, reward, next_state, done, timeout = batch
        if torch.any((done > 0) & (timeout > 0)):
            raise ValueError("A transition cannot terminate and time out simultaneously")
        sa = torch.cat([state, action], dim=-1)
        with torch.no_grad():
            q = torch.minimum(self.target_q1(sa), self.target_q2(sa))
        v = self.value(state)
        difference = q - v
        weight = torch.where(difference > 0, self.expectile, 1 - self.expectile)
        v_loss = (weight * difference.square()).mean()
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        with torch.no_grad():
            target = reward + self.discount * (1 - done) * self.value(next_state)
        q_loss = F.mse_loss(self.q1(sa), target) + F.mse_loss(self.q2(sa), target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        with torch.no_grad():
            advantage = torch.minimum(self.target_q1(sa), self.target_q2(sa)) - self.value(state)
            weights = torch.exp(advantage * 3).clamp(max=100)
        actor_loss = (weights * (self.actor(state).tanh() - action).square().mean(-1, keepdim=True)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        with torch.no_grad():
            for online, target_net in [(self.q1, self.target_q1), (self.q2, self.target_q2)]:
                for p, tp in zip(online.parameters(), target_net.parameters(), strict=True):
                    tp.lerp_(p, self.tau)
        return {"value_loss": float(v_loss.detach()), "q_loss": float(q_loss.detach()),
                "actor_loss": float(actor_loss.detach())}

    @torch.no_grad()
    def act(self, state: Tensor) -> Tensor:
        return self.actor(state).tanh()
