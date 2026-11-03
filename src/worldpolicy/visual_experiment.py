"""Train visual ACT/DDPM policies on synchronized FetchPush RGB demonstrations."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import deque
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .fetch_push import FetchPush
from .fetch_push_experiment import evaluate as evaluate_state_policy
from .fetch_push_experiment import make_model as make_state_model
from .statistics import wilson_interval
from .training import load_checkpoint, seed_all
from .visual_data import (
    collect_visual_episode,
    load_visual_dataset,
    proprio_from_state,
    save_visual_dataset,
    visual_chunks,
)
from .visual_policy import VisualPolicy


def torch_inputs(images: np.ndarray, proprio: np.ndarray,
                 device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(images.transpose(0, 1, 2, 5, 3, 4), device=device),
            torch.tensor(proprio, dtype=torch.float32, device=device))


def train_visual(model: VisualPolicy, images: np.ndarray, proprio: np.ndarray,
                 actions: np.ndarray, output: Path, updates: int, batch_size: int,
                 lr: float, seed: int, device: str = "cpu", resume: bool = False) -> dict:
    seed_all(seed)
    if device == "cpu":
        torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model.to(device)
    x, p = torch_inputs(images, proprio, device)
    y = torch.tensor(actions, dtype=torch.float32, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.pt"
    initial_update = 0
    if resume:
        if not progress.exists():
            raise FileNotFoundError("No visual training progress checkpoint to resume")
        previous = torch.load(progress, map_location=device, weights_only=False)
        if previous["updates_target"] != updates or previous["seed"] != seed:
            raise ValueError("Resume protocol mismatch")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["cpu_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state(previous["cuda_rng"])
        losses = previous["losses"]
        initial_update = previous["update"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for update in range(initial_update, updates):
        index = torch.randint(len(x), (batch_size,), device=device)
        loss = model.loss(x[index], p[index] if model.use_proprio else None, y[index])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite visual training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 100 == 0 and update + 1 < updates:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
                        "losses": losses, "update": update + 1,
                        "updates_target": updates, "seed": seed}, progress)
    checkpoint = output / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "kind": model.kind, "use_proprio": model.use_proprio,
                "history": model.history, "horizon": model.horizon,
                "updates": updates, "seed": seed, "torch_rng_state": torch.get_rng_state()}, checkpoint)
    record = {"status": "GPU_VERIFIED" if device == "cuda" else "CPU_VERIFIED",
              "device": device, "updates": updates, "seed": seed,
              "training_seconds": time.perf_counter() - start, "first_loss": losses[0],
              "last_loss": losses[-1], "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
              "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "resumed_from_update": initial_update}
    (output / "train.json").write_text(json.dumps(record, indent=2))
    return record


def load_visual(model: VisualPolicy, checkpoint: Path, digest: str) -> None:
    if sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError("Visual checkpoint checksum mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload["kind"], payload["use_proprio"], payload["history"], payload["horizon"]) != (
        model.kind, model.use_proprio, model.history, model.horizon
    ):
        raise ValueError("Visual checkpoint architecture mismatch")
    model.load_state_dict(payload["model"])
    model.eval()


def visual_rollout(model: VisualPolicy, seed: int,
                   camera_shift: tuple[float, float, float] = (0, 0, 0),
                   camera_setup: str = "legacy") -> dict:
    env = FetchPush(seed, camera_shift=camera_shift, camera_setup=camera_setup)
    state = env.reset()
    images: deque[np.ndarray] = deque(maxlen=model.history)
    proprio: deque[np.ndarray] = deque(maxlen=model.history)
    actions, rewards, latencies = [], [], []
    for _ in range(env.max_steps):
        frame = env.camera_frames()
        robot = proprio_from_state(state) if model.use_proprio else np.zeros(10, dtype=np.float32)
        if not images:
            for _ in range(model.history - 1):
                images.append(frame)
                proprio.append(robot)
        images.append(frame)
        proprio.append(robot)
        pixels = torch.tensor(np.asarray(images).transpose(0, 1, 4, 2, 3)[None])
        robot_tensor = torch.tensor(np.asarray(proprio)[None]) if model.use_proprio else None
        started = time.perf_counter()
        action = model.act(pixels, robot_tensor)[0].numpy()
        latencies.append((time.perf_counter() - started) * 1000)
        state, reward, terminated, truncated, info = env.step(action)
        actions.append(action.tolist())
        rewards.append(reward)
        if info["success"] or terminated or truncated:
            break
    result = {"seed": seed, "success": info["success"], "goal_distance": info["goal_distance"],
              "steps": len(actions), "actions": actions, "rewards": rewards,
              "inference_ms_p50": float(np.median(latencies)),
              "camera_shift_m": camera_shift, "evidence": "SIMULATION",
              "student_observation": "RGB only" if not model.use_proprio else "RGB plus robot proprio"}
    env.close()
    return result


def run(config_path: Path, output: Path, device: str = "cpu", resume: bool = False) -> dict:
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    groups = [set(range(config[key], config[key] + config[count])) for key, count in (
        ("training_episode_seed_start", "training_episodes"),
        ("validation_episode_seed_start", "validation_episodes"),
        ("confirmatory_episode_seed_start", "confirmatory_episodes"))]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Visual experiment seed leakage")
    output.mkdir(parents=True, exist_ok=True)
    train_dir, val_dir = output / "train_dataset", output / "validation_dataset"
    train_seeds = sorted(groups[0])
    val_seeds = sorted(groups[1])
    camera_setup = config.get("camera_setup", "legacy")
    if not resume:
        if train_dir.exists() or val_dir.exists():
            raise FileExistsError("Visual run directory already contains datasets; choose a new run ID")
        save_visual_dataset([collect_visual_episode(seed, camera_setup=camera_setup)
                             for seed in train_seeds], train_dir, camera_setup)
        save_visual_dataset([collect_visual_episode(seed, camera_setup=camera_setup)
                             for seed in val_seeds], val_dir, camera_setup)
    training, train_manifest = load_visual_dataset(train_dir)
    validation, _ = load_visual_dataset(val_dir)
    images, proprio, actions = visual_chunks(training, config["observation_history"], config["action_chunk"])
    vi, vp, va = visual_chunks(validation, config["observation_history"], config["action_chunk"])
    valid_pixels, valid_robot = torch_inputs(vi, vp)
    valid_actions = torch.tensor(va)
    seeds = sorted(groups[2])
    summary = {"status": "GPU_VERIFIED" if device == "cuda" else "CPU_VERIFIED",
               "device": device, "task": "FetchPushDense-v4",
               "protocol_id": config["protocol_id"], "protocol_sha256": sha256(config_bytes).hexdigest(),
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "dataset_manifest_sha256": sha256((train_dir / "manifest.json").read_bytes()).hexdigest(),
               "training_episodes": len(training), "training_successes": sum(ep.success for ep in training),
               "training_chunks": len(actions), "validation_episodes": len(validation),
               "evaluation_seeds": seeds, "camera_names": train_manifest["cameras"], "policies": {}}
    for name, kind, use_proprio in (("vision_only_act", "act", False),
                                    ("visual_proprio_act", "act", True),
                                    ("visual_proprio_diffusion", "diffusion", True)):
        seed_all(config["training_seed"])
        model = VisualPolicy(kind, use_proprio, config["observation_history"], config["action_chunk"])
        directory = output / name
        if resume and (directory / "evaluation.json").exists():
            detail = json.loads((directory / "evaluation.json").read_text())
            summary["policies"][name] = {k: v for k, v in detail.items() if k != "traces"}
            continue
        if resume and (directory / "train.json").exists():
            train_record = json.loads((directory / "train.json").read_text())
        else:
            train_record = train_visual(model, images, proprio, actions, directory,
                                        config["updates"], config["batch_size"],
                                        config["learning_rate"], config["training_seed"],
                                        device=device, resume=resume and (directory / "progress.pt").exists())
        restored = VisualPolicy(kind, use_proprio, config["observation_history"], config["action_chunk"])
        load_visual(restored, directory / "checkpoint.pt", train_record["checkpoint_sha256"])
        torch.manual_seed(config["training_seed"] + 999)
        with torch.no_grad():
            if kind == "act":
                prediction = restored.core(restored.encode(valid_pixels,
                    valid_robot if use_proprio else None))[0]
            else:
                prediction = restored.core.sample(restored.encode(valid_pixels, valid_robot))
            validation_mse = float(torch.square(prediction - valid_actions).mean())
        rows = []
        for seed in seeds:
            torch.manual_seed(seed + 700000)
            rows.append(visual_rollout(restored, seed, camera_setup=camera_setup))
        success = sum(row["success"] for row in rows)
        detail = {"training": train_record, "validation_action_mse": validation_mse,
                  "successes": success, "episodes": len(rows), "wilson_95": wilson_interval(success, len(rows)),
                  "traces": rows}
        (directory / "evaluation.json").write_text(json.dumps(detail, indent=2))
        summary["policies"][name] = {k: v for k, v in detail.items() if k != "traces"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    # Privileged state-only baselines are marked separately and use an older, larger dataset.
    state_root = Path("artifacts/v12/fetch_push_b/seed_21")
    normalization = np.load(state_root / "state_normalization.npz")
    for name in ("act", "diffusion"):
        model = make_state_model(name, 4)
        record = json.loads((state_root / name / "train.json").read_text())
        load_checkpoint(model, state_root / name / "checkpoint.pt", record["checkpoint_sha256"])
        rows, _ = evaluate_state_policy(model, seeds, normalization["mean"], normalization["std"])
        summary["policies"][f"state_only_{name}"] = {
            "successes": sum(row["success"] for row in rows), "episodes": len(rows),
            "training_episodes": 64, "checkpoint_sha256": record["checkpoint_sha256"],
            "comparison_warning": "different training demonstrations and update budget",
            "traces": rows}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return {**summary, "policies": {key: {k: v for k, v in detail.items() if k != "traces"}
                                     for key, detail in summary["policies"].items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v13/visual_fetch_push.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v13/visual_fetch_push"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.device, args.resume), indent=2))


if __name__ == "__main__":
    main()
