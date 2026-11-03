"""Fresh-seed ACT/DDPM comparison with approximately matched CPU training time."""

from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .data import chunks, load_dataset
from .fetch_push_experiment import evaluate, make_model
from .statistics import exact_mcnemar_pvalue, paired_counts, wilson_interval
from .training import load_checkpoint, seed_all, train


def run(config_path: Path, output: Path) -> dict:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seed = config["training_seed"]
    seed_all(seed)
    pairs, manifest = load_dataset(Path(config["dataset"]))
    x, y = chunks(pairs, 4)
    normalization = np.load(config["normalization"])
    mean, std = normalization["mean"], normalization["std"]
    x = (x - mean) / std
    act = make_model("act", 4)
    act_record = json.loads(Path(config["act_checkpoint"]).with_name("train.json").read_text())
    load_checkpoint(act, Path(config["act_checkpoint"]), act_record["checkpoint_sha256"])
    diffusion = make_model("diffusion", 4)
    training = train(diffusion, x, y, steps=config["diffusion_updates"],
                     batch_size=config["batch_size"], lr=config["learning_rate"],
                     seed=seed, output=output / "diffusion")
    load_checkpoint(diffusion, output / "diffusion" / "checkpoint.pt", training["checkpoint_sha256"])
    eval_seeds = list(range(config["confirmatory_seed_start"],
                            config["confirmatory_seed_start"] + config["confirmatory_episodes"]))
    act_rows, act_latency = evaluate(act, eval_seeds, mean, std)
    diffusion_rows, diffusion_latency = evaluate(diffusion, eval_seeds, mean, std)
    counts = paired_counts([row["success"] for row in act_rows],
                           [row["success"] for row in diffusion_rows])
    outcome = {"status": "CPU_VERIFIED", "protocol_id": config["protocol_id"],
               "protocol_sha256": sha256(config_bytes).hexdigest(),
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "dataset_manifest_sha256": sha256(Path(config["dataset"], "manifest.json").read_bytes()).hexdigest(),
               "dataset_episodes": len(manifest["episodes"]), "training_seed": seed,
               "eval_seeds": eval_seeds,
               "act": {"training_seconds": act_record["seconds"],
                       "checkpoint_sha256": act_record["checkpoint_sha256"],
                       "successes": sum(row["success"] for row in act_rows),
                       "wilson_95": wilson_interval(sum(row["success"] for row in act_rows), len(act_rows)),
                       "latency_ms_p50": float(np.median(act_latency)), "episodes": act_rows},
               "diffusion": {"training_seconds": training["seconds"],
                             "checkpoint_sha256": training["checkpoint_sha256"],
                             "successes": sum(row["success"] for row in diffusion_rows),
                             "wilson_95": wilson_interval(sum(row["success"] for row in diffusion_rows), len(diffusion_rows)),
                             "latency_ms_p50": float(np.median(diffusion_latency)),
                             "episodes": diffusion_rows},
               "paired_counts": counts, "exact_mcnemar_p": exact_mcnemar_pvalue(counts)}
    (output / "evaluation.json").write_text(json.dumps(outcome, indent=2))
    return {**outcome, "act": {k: v for k, v in outcome["act"].items() if k != "episodes"},
            "diffusion": {k: v for k, v in outcome["diffusion"].items() if k != "episodes"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/v12/fetch_push_compute_control.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v12/fetch_push_compute_control"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
