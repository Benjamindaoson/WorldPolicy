"""Small learned dynamics ensemble for point-reach planning experiments."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .planning import WorldModelMetadata
from .simulation import SPEC, PointReach


class Dynamics(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, 4))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return state + self.net(torch.cat([state, action], -1))


class LearnedPointDynamics:
    def __init__(self, models: list[Dynamics], checkpoint_id: str, horizon: int = 8):
        if len(models) < 2:
            raise ValueError("At least two models needed for ensemble uncertainty")
        self.models = [m.eval() for m in models]
        self.metadata = WorldModelMetadata("learned_point_dynamics", "three_mlp_ensemble",
                                           checkpoint_id, "1", SPEC, 4, (4,), horizon,
                                           "state_trajectory", "ensemble_state_variance",
                                           "learned_dynamics")

    @torch.no_grad()
    def predict(self, state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if actions.shape[1] > self.metadata.horizon or actions.shape[-1] != SPEC.dimension:
            raise ValueError("Prediction horizon or action dimension mismatch")
        all_rollouts = []
        for model in self.models:
            current = torch.tensor(np.broadcast_to(state, (len(actions), 4)).copy(), dtype=torch.float32)
            states = []
            for t in range(actions.shape[1]):
                current = model(current, torch.tensor(actions[:, t], dtype=torch.float32))
                states.append(current.numpy())
            all_rollouts.append(np.stack(states, 1))
        ensemble = np.stack(all_rollouts)
        return ensemble.mean(0), ensemble.var(0).mean(-1)


def collect_transitions(episodes: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []
    for i in range(episodes):
        env = PointReach(seed + i)
        state = env.reset()
        for _ in range(35):
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            next_state, _, done, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            state = next_state
            if done or truncated:
                break
    return tuple(np.asarray(x, dtype=np.float32) for x in (states, actions, next_states))


def train_dynamics(output: Path, seed: int = 0, episodes: int = 64,
                   steps: int = 500) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    states, actions, next_states = collect_transitions(episodes, seed)
    x, a, y = (torch.tensor(z) for z in (states, actions, next_states))
    models = []
    losses = []
    for member in range(3):
        torch.manual_seed(seed + member)
        model = Dynamics()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(steps):
            index = torch.randint(len(x), (64,))
            loss = (model(x[index], a[index]) - y[index]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        models.append(model)
        losses.append(float(loss.detach()))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "dynamics.pt"
    torch.save({"models": [m.state_dict() for m in models], "seed": seed,
                "steps": steps, "episodes": episodes, "protocol": SPEC.__dict__}, path)
    digest = sha256(path.read_bytes()).hexdigest()
    # Independent transitions generated with unseen environment seeds.
    held_x, held_a, held_y = collect_transitions(8, seed + 10000)
    adapter = LearnedPointDynamics(models, digest)
    predicted, uncertainty = adapter.predict(held_x, held_a[:, None])
    mse = float(np.square(predicted[:, 0] - held_y).mean())
    record = {"status": "EXECUTED", "evidence": "learned_dynamics",
              "train_episodes": episodes, "held_out_episodes": 8, "steps_per_member": steps,
              "ensemble_train_losses": losses, "held_out_one_step_mse": mse,
              "mean_predicted_uncertainty": float(uncertainty.mean()), "checkpoint_sha256": digest}
    (output / "train.json").write_text(json.dumps(record, indent=2))
    return record


def load_dynamics(path: Path, digest: str) -> LearnedPointDynamics:
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Dynamics checkpoint integrity failure")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["protocol"] != SPEC.__dict__:
        raise ValueError("Dynamics action protocol mismatch")
    models = [Dynamics() for _ in payload["models"]]
    for model, state in zip(models, payload["models"], strict=True):
        model.load_state_dict(state)
    return LearnedPointDynamics(models, digest)
