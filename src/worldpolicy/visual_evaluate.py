"""Independent read-only visual checkpoint evaluation and fixed-camera shift test."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .statistics import wilson_interval
from .visual_experiment import load_visual, visual_rollout
from .visual_policy import VisualPolicy


def run(run_dir: Path, output: Path, seed_start: int, episodes: int,
        camera_shift: tuple[float, float, float],
        object_shift: tuple[float, float] = (.03, 0.)) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    # Run summary defines the executed camera ordering, while each checkpoint defines its architecture.
    setup = "task_visible" if summary["camera_names"][0] == "external_camera_0" else "legacy"
    train_seeds = {row["seed"] for row in json.loads(
        (run_dir / "train_dataset" / "manifest.json").read_text())["episodes"]}
    val_seeds = {row["seed"] for row in json.loads(
        (run_dir / "validation_dataset" / "manifest.json").read_text())["episodes"]}
    seeds = list(range(seed_start, seed_start + episodes))
    if set(seeds) & (train_seeds | val_seeds | set(summary["evaluation_seeds"])):
        raise ValueError("Independent evaluation seed leakage")
    results = {}
    for name in ("vision_only_act", "visual_proprio_act", "visual_proprio_diffusion"):
        detail = summary["policies"][name]
        model = VisualPolicy("diffusion" if name.endswith("diffusion") else "act",
                             name != "vision_only_act")
        checkpoint = run_dir / name / "checkpoint.pt"
        load_visual(model, checkpoint, detail["training"]["checkpoint_sha256"])
        pairs = []
        for seed in seeds:
            torch.manual_seed(seed + 700000)
            nominal = visual_rollout(model, seed, camera_setup=setup)
            torch.manual_seed(seed + 700000)
            shifted = visual_rollout(model, seed, camera_shift, camera_setup=setup)
            torch.manual_seed(seed + 700000)
            moved = visual_rollout(model, seed, camera_setup=setup, object_shift=object_shift)
            pairs.append({"seed": seed, "nominal": nominal,
                          "camera_shift": shifted, "object_shift": moved})
        nominal_count = sum(row["nominal"]["success"] for row in pairs)
        shifted_count = sum(row["camera_shift"]["success"] for row in pairs)
        object_count = sum(row["object_shift"]["success"] for row in pairs)
        results[name] = {"checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                         "nominal_successes": nominal_count,
                         "shifted_successes": shifted_count,
                         "object_shift_successes": object_count,
                         "nominal_wilson_95": wilson_interval(nominal_count, episodes),
                         "shifted_wilson_95": wilson_interval(shifted_count, episodes),
                         "object_shift_wilson_95": wilson_interval(object_count, episodes),
                         "median_nominal_inference_ms": float(np.median([
                             row["nominal"]["inference_ms_p50"] for row in pairs])),
                         "paired_traces": pairs}
    report = {"status": "CPU_VERIFIED_SIMULATION", "source_protocol": summary["protocol_id"],
              "source_commit": summary["source_commit"],
              "evaluation_seeds": seeds, "camera_shift_m": camera_shift,
              "object_shift_m": object_shift,
              "policies": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Evaluation output exists; use a new run ID")
    output.write_text(json.dumps(report, indent=2))
    return {**report, "policies": {name: {k: v for k, v in row.items() if k != "paired_traces"}
                                    for name, row in results.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=104000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--camera-shift", type=float, nargs=3, default=(.05, 0., 0.))
    parser.add_argument("--object-shift", type=float, nargs=2, default=(.03, 0.))
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir, args.output, args.seed_start,
                         args.episodes, tuple(args.camera_shift), tuple(args.object_shift)), indent=2))


if __name__ == "__main__":
    main()
