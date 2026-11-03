"""Held-out robot dynamics planning diagnostics with exact physics candidate replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fetch_push import SPEC, FetchPush
from .fetch_push_dynamics import load_dynamics
from .fetch_push_planning import FetchPhysicsOracle, push_terminal_cost
from .planning import CEMPlanner


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def run(checkpoint: Path, digest: str, seeds: list[int], output: Path) -> dict:
    learned = load_dynamics(checkpoint, digest)
    rng = np.random.default_rng(17)
    rows = []
    for seed in seeds:
        env = FetchPush(seed)
        state = env.reset()
        oracle = FetchPhysicsOracle(env)
        candidates = rng.uniform(-1, 1, (64, 3, 4)).astype(np.float32)
        actual, _ = oracle.predict(state, candidates)
        prediction, uncertainty = learned.predict(state, candidates)
        rank = rank_correlation(push_terminal_cost(actual[:, -1]),
                                push_terminal_cost(prediction[:, -1]))
        planner = CEMPlanner(learned, SPEC, horizon=3, population=16, elites=4,
                             iterations=2, seed=seed, uncertainty_weight=0,
                             terminal_cost=push_terminal_cost)
        _, info = planner.plan(state)
        sequence = np.asarray(info["planned_sequence"], dtype=np.float32)[None]
        actual_selected, _ = oracle.predict(state, sequence)
        learned_selected, _ = learned.predict(state, sequence)
        rows.append({"seed": seed, "candidate_rank_correlation": rank,
                     "state_mse_horizon_1": float(np.square(actual[:, 0] - prediction[:, 0]).mean()),
                     "state_mse_horizon_3": float(np.square(actual[:, -1] - prediction[:, -1]).mean()),
                     "object_xy_mse_horizon_3": float(np.square(actual[:, -1, 3:5] - prediction[:, -1, 3:5]).mean()),
                     "mean_uncertainty": float(uncertainty.mean()),
                     "selected_predicted_cost": float(push_terminal_cost(learned_selected[:, -1])[0]),
                     "selected_actual_cost": float(push_terminal_cost(actual_selected[:, -1])[0])})
        env.close()
    report = {"status": "CPU_VERIFIED", "task": "FetchPushDense-v4",
              "checkpoint_sha256": digest, "seeds": seeds,
              "mean_state_mse_horizon_1": float(np.mean([r["state_mse_horizon_1"] for r in rows])),
              "mean_state_mse_horizon_3": float(np.mean([r["state_mse_horizon_3"] for r in rows])),
              "mean_object_xy_mse_horizon_3": float(np.mean([r["object_xy_mse_horizon_3"] for r in rows])),
              "mean_abs_selected_cost_error": float(np.mean([
                  abs(r["selected_predicted_cost"] - r["selected_actual_cost"]) for r in rows])),
              "episodes": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return {key: value for key, value in report.items() if key != "episodes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/v12/fetch_push_dynamics/checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v12/fetch_push_dynamics/diagnostics.json"))
    args = parser.parse_args()
    digest = json.loads(args.checkpoint.with_name("train.json").read_text())["checkpoint_sha256"]
    print(json.dumps(run(args.checkpoint, digest, list(range(30000, 30010)), args.output), indent=2))


if __name__ == "__main__":
    main()
