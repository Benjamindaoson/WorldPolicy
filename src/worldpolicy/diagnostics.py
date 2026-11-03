"""Paired point-reach diagnostics for planner and learned-dynamics failure modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .planning import CEMPlanner, PhysicsOracle
from .simulation import SPEC, PointReach, expert_action, rollout
from .world_model import load_dynamics


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def run(output: Path, checkpoint: Path, checkpoint_sha256: str,
        seeds: list[int] | None = None) -> dict:
    seeds = seeds or list(range(2000, 2008))
    learned = load_dynamics(checkpoint, checkpoint_sha256)
    oracle = PhysicsOracle()
    variants = {
        "random_action": None,
        "expert": expert_action,
        "random_shooting_oracle": (oracle, 8, 1, 0.0),
        "cem_oracle": (oracle, 8, 4, 0.0),
        "random_shooting_learned": (learned, 8, 1, 0.0),
        "cem_learned": (learned, 8, 4, 0.0),
        "cem_learned_uncertainty": (learned, 8, 4, 10.0),
        "cem_learned_short_horizon": (learned, 4, 4, 0.0),
    }
    cases = {}
    for variant, setup in variants.items():
        rows = []
        for seed in seeds:
            if setup is None:
                rng = np.random.default_rng(seed + 4000)
                policy = lambda _s, generator=rng: generator.uniform(-1, 1, 2)
            elif callable(setup):
                policy = setup
            else:
                model, horizon, iterations, penalty = setup
                planner = CEMPlanner(model, SPEC, horizon=horizon, population=64,
                                     elites=8, iterations=iterations, seed=seed,
                                     uncertainty_weight=penalty)
                policy = lambda s, controller=planner: controller.plan(s)[0]
            rows.append(rollout(policy, seed))
        cases[variant] = {"successes": sum(r["success"] for r in rows),
                          "mean_return": float(np.mean([r["return"] for r in rows])),
                          "episodes": rows}

    rng = np.random.default_rng(77)
    horizon_results = {}
    for horizon in (1, 2, 4, 8):
        errors, correlations, cost_bias = [], [], []
        for seed in seeds:
            env = PointReach(seed)
            state = env.reset()
            candidates = rng.uniform(-1, 1, (128, horizon, 2))
            actual, _ = oracle.predict(state, candidates)
            predicted, _ = learned.predict(state, candidates)
            errors.append(float(np.square(predicted[:, -1] - actual[:, -1]).mean()))
            actual_cost = np.linalg.norm(actual[:, -1, 2:4], axis=1)
            predicted_cost = np.linalg.norm(predicted[:, -1, 2:4], axis=1)
            correlations.append(_rank_correlation(actual_cost, predicted_cost))
            cost_bias.append(float(np.mean(predicted_cost - actual_cost)))
        horizon_results[str(horizon)] = {"state_mse": float(np.mean(errors)),
                                          "candidate_rank_correlation": float(np.mean(correlations)),
                                          "predicted_minus_actual_goal_cost": float(np.mean(cost_bias))}
    selected_plan_gaps = {}
    for name, model in (("oracle", oracle), ("learned", learned)):
        rows = []
        for seed in seeds:
            env = PointReach(seed)
            state = env.reset()
            planner = CEMPlanner(model, SPEC, horizon=8, population=64, elites=8,
                                 iterations=4, seed=seed, uncertainty_weight=0)
            _, info = planner.plan(state)
            sequence = np.asarray(info["planned_sequence"])[None]
            prediction, _ = model.predict(state, sequence)
            actual, _ = oracle.predict(state, sequence)
            predicted_cost = float(np.linalg.norm(prediction[0, -1, 2:4]))
            actual_cost = float(np.linalg.norm(actual[0, -1, 2:4]))
            rows.append({"seed": seed, "predicted_goal_cost": predicted_cost,
                         "actual_goal_cost": actual_cost, "cost_error": predicted_cost - actual_cost})
        selected_plan_gaps[name] = rows
    result = {"status": "CPU_VERIFIED", "task": "point_reach_euler_v1",
              "checkpoint_sha256": checkpoint_sha256, "seeds": seeds,
              "cases": cases, "prediction_by_horizon": horizon_results,
              "selected_plan_cost_gaps": selected_plan_gaps,
              "action_bounds": [SPEC.lower, SPEC.upper],
              "planning_horizon_steps": 8, "control_frequency_hz": SPEC.frequency_hz}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return {**result, "cases": {name: {k: v for k, v in row.items() if k != "episodes"}
                               for name, row in cases.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dynamics/dynamics.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v11/point_reach_diagnostics.json"))
    args = parser.parse_args()
    source = json.loads(args.checkpoint.with_name("train.json").read_text())
    print(json.dumps(run(args.output, args.checkpoint, source["checkpoint_sha256"]), indent=2))


if __name__ == "__main__":
    main()
