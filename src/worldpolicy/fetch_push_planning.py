"""Exploratory physics-oracle CEM on official FetchPush-v4; never a learned model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .fetch_push import SPEC, FetchPush, PushExpert
from .planning import CEMPlanner, WorldModelMetadata


class FetchPhysicsOracle:
    def __init__(self, env: FetchPush, horizon: int = 3):
        self.env = env
        self.metadata = WorldModelMetadata("fetch_push_mujoco_oracle", "mujoco_forward",
                                           "none", env.version, SPEC, 28, (28,), horizon,
                                           "state_trajectory", "zero", "physics_oracle")

    def predict(self, state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if actions.ndim != 3 or actions.shape[1:] != (self.metadata.horizon, 4):
            raise ValueError("Fetch oracle action protocol mismatch")
        snapshot = self.env.snapshot()
        current = self.env.restore(snapshot)
        if not np.allclose(current, state, atol=1e-5):
            raise ValueError("Oracle state does not match planning state")
        futures = []
        try:
            for sequence in actions:
                self.env.restore(snapshot)
                states = []
                for action in sequence:
                    observation, _, _, _, _ = self.env.step(action)
                    states.append(observation)
                futures.append(np.stack(states))
        finally:
            self.env.restore(snapshot)
        future = np.stack(futures)
        return future, np.zeros(future.shape[:2])


def push_terminal_cost(states: np.ndarray) -> np.ndarray:
    object_pos = states[:, 3:6]
    goal = states[:, 25:28]
    grip = states[:, :3]
    direction = goal[:, :2] - object_pos[:, :2]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    behind = object_pos.copy()
    behind[:, :2] -= .07 * direction
    behind[:, 2] += .025
    return np.linalg.norm(object_pos - goal, axis=1) + .25 * np.linalg.norm(grip - behind, axis=1)


def run(seeds: list[int], output: Path, population: int = 16,
        iterations: int = 2, horizon: int = 3) -> dict:
    rows = []
    for seed in seeds:
        methods = {}
        for method in ("expert", "random_shooting_oracle", "cem_oracle"):
            env = FetchPush(seed)
            state = env.reset()
            expert = PushExpert()
            model = FetchPhysicsOracle(env, horizon)
            planner = None
            if method != "expert":
                pop = population * iterations if method == "random_shooting_oracle" else population
                iters = 1 if method == "random_shooting_oracle" else iterations
                planner = CEMPlanner(model, SPEC, horizon=horizon, population=pop,
                                     elites=max(2, pop // 4), iterations=iters, seed=seed,
                                     terminal_cost=push_terminal_cost)
            actions, rewards = [], []
            started = time.perf_counter()
            for _ in range(env.max_steps):
                action = expert(state) if planner is None else planner.plan(state)[0]
                state, reward, terminated, truncated, info = env.step(action)
                actions.append(action.tolist())
                rewards.append(reward)
                if info["success"] or terminated or truncated:
                    break
            methods[method] = {"success": info["success"], "distance": info["goal_distance"],
                               "actions": actions, "rewards": rewards,
                               "planning_seconds": time.perf_counter() - started}
            env.close()
        rows.append({"seed": seed, "methods": methods})
    report = {"status": "CPU_VERIFIED", "task": "FetchPushDense-v4",
              "evidence": "physics_oracle_exploratory", "population": population,
              "iterations": iterations, "horizon": horizon,
              "candidate_sequences_per_action": population * iterations,
              "seeds": seeds, "episodes": rows,
              "successes": {name: sum(row["methods"][name]["success"] for row in rows)
                            for name in ("expert", "random_shooting_oracle", "cem_oracle")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key != "episodes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v12/fetch_push_oracle.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[30000, 30001, 30002])
    args = parser.parse_args()
    print(json.dumps(run(args.seeds, args.output), indent=2))


if __name__ == "__main__":
    main()
