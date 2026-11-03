"""Versioned world-model contract and real CEM optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .data import ActionSpec
from .simulation import SPEC


@dataclass(frozen=True)
class WorldModelMetadata:
    model_id: str
    architecture: str
    checkpoint_id: str
    version: str
    action_spec: ActionSpec
    state_dim: int
    observation_shape: tuple[int, ...]
    horizon: int
    output_schema: str
    uncertainty_type: str
    evidence: str


class WorldModelAdapter(Protocol):
    metadata: WorldModelMetadata

    def predict(self, state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted states [population,horizon,4] and uncertainty [population,horizon]."""


def check_contract(model: WorldModelAdapter, spec: ActionSpec, horizon: int) -> None:
    if model.metadata.action_spec != spec or model.metadata.horizon < horizon:
        raise ValueError("World model action protocol or horizon mismatch")
    if model.metadata.state_dim < 1 or model.metadata.observation_shape != (model.metadata.state_dim,):
        raise ValueError("World model state protocol mismatch")


class PhysicsOracle:
    """Known Euler dynamics, explicitly NOT a learned world model."""

    def __init__(self, horizon: int = 8):
        self.metadata = WorldModelMetadata("point_reach_physics", "known_euler", "none",
                                           "1", SPEC, 4, (4,), horizon,
                                           "state_trajectory", "zero", "physics_oracle")

    def predict(self, state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = np.broadcast_to(state[:2], (len(actions), 2)).copy()
        goal = state[:2] + state[2:4]
        states = []
        for t in range(actions.shape[1]):
            position = np.clip(position + .1 * np.clip(actions[:, t], -1, 1), -1, 1)
            states.append(np.concatenate([position, goal - position], axis=1))
        return np.stack(states, axis=1), np.zeros(actions.shape[:2])


class CEMPlanner:
    def __init__(self, model: WorldModelAdapter, spec: ActionSpec, horizon: int = 8,
                 population: int = 128, elites: int = 16, iterations: int = 4,
                 seed: int = 0, uncertainty_weight: float = 1.0):
        check_contract(model, spec, horizon)
        if not 0 < elites <= population or iterations < 1:
            raise ValueError("Invalid CEM configuration")
        self.model, self.spec, self.horizon = model, spec, horizon
        self.population, self.elites, self.iterations = population, elites, iterations
        self.rng = np.random.default_rng(seed)
        self.uncertainty_weight = uncertainty_weight

    def plan(self, state: np.ndarray) -> tuple[np.ndarray, dict]:
        if state.shape != self.model.metadata.observation_shape or not np.isfinite(state).all():
            raise ValueError("Planner state protocol mismatch")
        dims = self.spec.dimension
        mean = np.zeros((self.horizon, dims))
        std = np.ones_like(mean)
        low, high = np.array(self.spec.lower), np.array(self.spec.upper)
        best_cost = float("inf")
        best = None
        for _ in range(self.iterations):
            candidates = np.clip(self.rng.normal(mean, std, (self.population, self.horizon, dims)), low, high)
            future, uncertainty = self.model.predict(state, candidates)
            if future.shape != (self.population, self.horizon, len(state)) or uncertainty.shape != (self.population, self.horizon):
                raise ValueError("World model prediction shape mismatch")
            cost = np.linalg.norm(future[:, -1, 2:4], axis=1)
            cost += self.uncertainty_weight * uncertainty.mean(axis=1)
            cost += .01 * np.square(np.diff(candidates, axis=1)).mean(axis=(1, 2))
            elite = np.argsort(cost)[:self.elites]
            if cost[elite[0]] < best_cost:
                best_cost = float(cost[elite[0]])
                best = candidates[elite[0]].copy()
            mean = candidates[elite].mean(axis=0)
            std = np.maximum(candidates[elite].std(axis=0), .05)
        assert best is not None
        return best[0], {"predicted_cost": best_cost, "planned_sequence": best.tolist(),
                         "model_evidence": self.model.metadata.evidence}
