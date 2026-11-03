"""Frozen multi-seed ACT/DDPM training and paired official FetchPush evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .data import chunks, load_dataset, save_dataset
from .fetch_push import PushExpert, collect_expert, rollout
from .policies import ACT, DiffusionPolicy
from .statistics import paired_counts, wilson_interval
from .training import load_checkpoint, seed_all, train


def make_model(kind: str, horizon: int) -> ACT | DiffusionPolicy:
    if kind == "act":
        return ACT(28, 4, horizon, hidden=64, latent=16, layers=2)
    if kind == "diffusion":
        return DiffusionPolicy(28, 4, horizon, hidden=128, timesteps=20)
    raise ValueError(f"Unknown policy: {kind}")


def evaluate(model: ACT | DiffusionPolicy, seeds: list[int]) -> tuple[list[dict], list[float]]:
    rows = []
    latencies = []
    model.eval()
    for seed in seeds:
        torch.manual_seed(seed + 700000)

        def policy(state: np.ndarray) -> np.ndarray:
            start = time.perf_counter()
            action = model.act(torch.tensor(state[None], dtype=torch.float32))[0].numpy()
            latencies.append((time.perf_counter() - start) * 1000)
            return action

        rows.append(rollout(seed, policy))
    return rows, latencies


def run(config_path: Path, output: Path, *, training_seeds: list[int] | None = None) -> dict:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    seeds = training_seeds or config["training_seeds"]
    if any(seed not in config["training_seeds"] for seed in seeds):
        raise ValueError("Training seed outside frozen protocol")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    confirm = list(range(config["confirmatory_evaluation_seed_start"],
                         config["confirmatory_evaluation_seed_start"] +
                         config["confirmatory_evaluation_episodes"]))
    exploratory = config["exploratory_evaluation_seeds"]
    summary = {"status": "CPU_VERIFIED", "protocol_id": config["protocol_id"],
               "protocol_sha256": sha256(config_bytes).hexdigest(), "source_commit": commit,
               "task": config["task"], "confirmatory_seeds": confirm,
               "training_runs": {}}
    for seed in seeds:
        index = config["training_seeds"].index(seed)
        root = output / f"seed_{seed}"
        train_start = config["train_episode_start_seeds"][index]
        val_start = config["validation_episode_start_seeds"][index]
        train_episodes = collect_expert(config["train_episodes_per_seed"], train_start)
        val_episodes = collect_expert(config["validation_episodes_per_seed"], val_start)
        save_dataset(train_episodes, root / "train_dataset")
        save_dataset(val_episodes, root / "validation_dataset")
        train_pairs, manifest = load_dataset(root / "train_dataset")
        val_pairs, _ = load_dataset(root / "validation_dataset")
        if set(range(train_start, train_start + len(train_episodes))) & set(
            range(val_start, val_start + len(val_episodes))
        ) or set(confirm) & set(range(train_start, train_start + len(train_episodes))):
            raise ValueError("Dataset seed leakage")
        x, y = chunks(train_pairs, config["action_chunk"])
        vx, vy = chunks(val_pairs, config["action_chunk"])
        dataset_hash = sha256((root / "train_dataset" / "manifest.json").read_bytes()).hexdigest()
        run_rows = {"dataset_manifest_sha256": dataset_hash,
                    "training_successes": sum(ep.success for ep in train_episodes),
                    "validation_successes": sum(ep.success for ep in val_episodes),
                    "policies": {}}
        for kind in ("act", "diffusion"):
            seed_all(seed)
            directory = root / kind
            training = train(make_model(kind, config["action_chunk"]), x, y,
                             steps=config["updates"], batch_size=config["batch_size"],
                             lr=config["learning_rate"], seed=seed, output=directory)
            model = make_model(kind, config["action_chunk"])
            load_checkpoint(model, directory / "checkpoint.pt", training["checkpoint_sha256"])
            torch.manual_seed(seed + 999)
            with torch.no_grad():
                predicted = model(torch.tensor(vx))[0] if kind == "act" else model.sample(torch.tensor(vx))
                validation_mse = float(torch.square(predicted - torch.tensor(vy)).mean())
            exploration_rows, _ = evaluate(model, exploratory)
            confirm_rows, latency = evaluate(model, confirm)
            success = sum(row["success"] for row in confirm_rows)
            detail = {"training": training, "training_seed": seed,
                      "dataset_manifest_sha256": dataset_hash,
                      "validation_action_mse": validation_mse,
                      "exploratory_successes": sum(row["success"] for row in exploration_rows),
                      "exploratory_episodes": len(exploratory),
                      "confirmatory_successes": success,
                      "confirmatory_episodes": len(confirm),
                      "wilson_95": wilson_interval(success, len(confirm)),
                      "inference_latency_ms_p50": float(np.percentile(latency, 50)),
                      "inference_latency_ms_p95": float(np.percentile(latency, 95)),
                      "exploratory_traces": exploration_rows, "confirmatory_traces": confirm_rows}
            (directory / "evaluation.json").write_text(json.dumps(detail, indent=2))
            run_rows["policies"][kind] = {k: v for k, v in detail.items()
                                         if k not in ("exploratory_traces", "confirmatory_traces")}
        act_rows = json.loads((root / "act" / "evaluation.json").read_text())["confirmatory_traces"]
        diff_rows = json.loads((root / "diffusion" / "evaluation.json").read_text())["confirmatory_traces"]
        run_rows["paired_act_diffusion"] = paired_counts(
            [row["success"] for row in act_rows], [row["success"] for row in diff_rows])
        summary["training_runs"][str(seed)] = run_rows
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v12/fetch_push_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v12/fetch_push"))
    parser.add_argument("--seed", type=int, help="Run only one frozen training seed")
    args = parser.parse_args()
    result = run(args.config, args.output,
                 training_seeds=[args.seed] if args.seed is not None else None)
    print(json.dumps({"status": result["status"], "task": result["task"],
                      "source_commit": result["source_commit"],
                      "training_runs": result["training_runs"]}, indent=2))


if __name__ == "__main__":
    main()
