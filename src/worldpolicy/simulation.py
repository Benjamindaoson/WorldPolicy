"""Deterministic continuous-control point reaching physics baseline (not MuJoCo)."""

from __future__ import annotations

import numpy as np

from .data import ActionSpec, Episode

SPEC = ActionSpec("point_mass_v1", 2, "m/s", "world_xy", 10.0, (-1., -1.), (1., 1.))


class PointReach:
    version = "point_reach_euler_v1"

    def __init__(self, seed: int = 0, shift: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.shift = shift
        self.pos = np.zeros(2)
        self.goal = np.array([0.65, 0.65])
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.pos = self.rng.uniform(-.8, -.4, size=2) + self.shift
        self.steps = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.concatenate([self.pos, self.goal - self.pos]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid action")
        self.pos = np.clip(self.pos + .1 * np.clip(action, -1, 1), -1, 1)
        self.steps += 1
        distance = float(np.linalg.norm(self.goal - self.pos))
        success = distance < .09
        truncated = self.steps >= 35
        return self.observe(), -distance, success, truncated, {"success": success}

    def snapshot(self) -> tuple[np.ndarray, int]:
        return self.pos.copy(), self.steps

    def restore(self, snapshot: tuple[np.ndarray, int]) -> None:
        self.pos, self.steps = snapshot[0].copy(), snapshot[1]


def expert_action(state: np.ndarray) -> np.ndarray:
    return np.clip(3 * state[2:4], -1, 1).astype(np.float32)


def collect_expert(count: int, seed: int = 0) -> list[Episode]:
    episodes = []
    for i in range(count):
        env = PointReach(seed + i)
        state = env.reset()
        states, actions = [], []
        for _ in range(35):
            action = expert_action(state)
            states.append(state)
            actions.append(action)
            state, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        episodes.append(Episode(
            f"expert_{seed}_{i}", "point_reach", SPEC.embodiment, "reach goal",
            np.array(states), np.array(actions), np.arange(len(actions)) / SPEC.frequency_hz,
            done, "physics_oracle_expert", "point_reach_v1", "simulation", SPEC,
        ))
    return episodes


def rollout(policy, seed: int, shift: float = 0.0) -> dict:
    env = PointReach(seed, shift)
    state = env.reset()
    actions = []
    total = 0.0
    for _ in range(35):
        action = np.asarray(policy(state), dtype=float).reshape(-1)[:2]
        state, reward, done, truncated, info = env.step(action)
        actions.append(action.tolist())
        total += reward
        if done or truncated:
            break
    return {"seed": seed, "success": info["success"], "return": total,
            "length": len(actions), "actions": actions, "environment": env.version,
            "evidence": "physics_baseline_simulation"}
