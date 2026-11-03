"""Paired ID/OOD object-friction evaluation on frozen FetchPush policy checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .fetch_push import rollout
from .fetch_push_experiment import make_model
from .statistics import paired_counts, wilson_interval
from .training import load_checkpoint


def run(root: Path, output: Path, seeds: list[int], friction_scale: float = 2.0) -> dict:
    torch.set_num_threads(1)
    results = {"status": "CPU_VERIFIED", "task": "FetchPushDense-v4",
               "ood_variable": "object_contact_friction", "ood_scale": friction_scale,
               "evaluation_seeds": seeds, "training_runs": {}}
    for run_dir in sorted(root.glob("seed_*")):
        if not run_dir.is_dir():
            continue
        normalizer = np.load(run_dir / "state_normalization.npz")
        per_seed = {}
        for kind in ("act", "diffusion"):
            policy_dir = run_dir / kind
            training = json.loads((policy_dir / "train.json").read_text())
            model = make_model(kind, 4)
            load_checkpoint(model, policy_dir / "checkpoint.pt", training["checkpoint_sha256"])
            conditions = {}
            for label, scale in (("id", 1.0), ("ood", friction_scale)):
                rows = []
                for seed in seeds:
                    torch.manual_seed(seed + 700000)

                    def policy(state: np.ndarray) -> np.ndarray:
                        features = (state - normalizer["mean"]) / normalizer["std"]
                        return model.act(torch.tensor(features[None], dtype=torch.float32))[0].numpy()

                    rows.append(rollout(seed, policy, friction_scale=scale))
                successes = sum(row["success"] for row in rows)
                conditions[label] = {"successes": successes,
                                     "wilson_95": wilson_interval(successes, len(rows)),
                                     "episodes": rows}
            conditions["paired_id_ood"] = paired_counts(
                [row["success"] for row in conditions["id"]["episodes"]],
                [row["success"] for row in conditions["ood"]["episodes"]])
            conditions["checkpoint_sha256"] = training["checkpoint_sha256"]
            per_seed[kind] = conditions
        results["training_runs"][run_dir.name] = per_seed
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    return {**results, "training_runs": {
        name: {kind: {condition: {key: value for key, value in row.items() if key != "episodes"}
                       if isinstance(row, dict) else row
                       for condition, row in details.items()}
               for kind, details in policies.items()}
        for name, policies in results["training_runs"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/v12/fetch_push_b"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v12/fetch_push_ood.json"))
    parser.add_argument("--friction-scale", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, list(range(70000, 70050)),
                         args.friction_scale), indent=2))


if __name__ == "__main__":
    main()
