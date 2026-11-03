"""Short real visual-policy training probe with measured device memory and step time."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from hashlib import sha256
from pathlib import Path

import torch

from worldpolicy.visual_data import load_visual_dataset, visual_chunks
from worldpolicy.visual_experiment import torch_inputs
from worldpolicy.visual_policy import VisualPolicy


def probe(train_dir: Path, output: Path, batch_size: int, updates: int,
          target_updates: int, device: str) -> dict:
    if updates < 2 or batch_size < 1:
        raise ValueError("Probe needs at least two updates and a positive batch")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested, but this PyTorch build has no CUDA access")
    torch.manual_seed(113)
    if device == "cpu":
        torch.set_num_threads(1)
    episodes, manifest = load_visual_dataset(train_dir)
    images, proprio, actions = visual_chunks(episodes)
    pixels, robot = torch_inputs(images, proprio, device)
    targets = torch.tensor(actions, dtype=torch.float32, device=device)
    records = {}
    for name, kind, use_proprio in (("vision_only_act", "act", False),
                                    ("visual_proprio_act", "act", True),
                                    ("visual_proprio_diffusion", "diffusion", True)):
        model = VisualPolicy(kind, use_proprio).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        parameters = sum(p.numel() for p in model.parameters())
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(updates):
            index = torch.randint(len(pixels), (batch_size,), device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            loss = model.loss(pixels[index], robot[index] if use_proprio else None,
                              targets[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        records[name] = {"parameters": parameters, "probe_updates": updates,
                         "last_loss": float(loss.detach()),
                         "median_step_seconds_excluding_first": statistics.median(times[1:]),
                         "estimated_target_training_seconds": statistics.median(times[1:]) * target_updates,
                         "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated()
                         if device == "cuda" else None,
                         "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()
                         if device == "cuda" else None}
        del model, optimizer
    result = {"status": "GPU_SIZING_REQUIRED" if device == "cpu" else "GPU_PROBE_MEASURED",
              "platform": platform.system(), "device": device, "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "gpu_name": torch.cuda.get_device_name()
              if device == "cuda" else None,
              "gpu_total_vram_bytes": torch.cuda.get_device_properties(0).total_memory
              if device == "cuda" else None,
              "batch_size": batch_size, "observation_history": 2, "action_horizon": 4,
              "image_shape": [2, 64, 64, 3], "target_updates_per_policy": target_updates,
              "dataset_episodes": len(episodes), "dataset_chunks": len(actions),
              "dataset_manifest_sha256": sha256((train_dir / "manifest.json").read_bytes()).hexdigest(),
              "camera_setup": manifest.get("camera_setup", "legacy"),
              "mixed_precision": "unverified; trainer uses float32",
              "multi_gpu_required": False,
              "policies": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--target-updates", type=int, default=1500)
    args = parser.parse_args()
    print(json.dumps(probe(args.train_dir, args.output, args.batch_size,
                           args.updates, args.target_updates, args.device), indent=2))


if __name__ == "__main__":
    main()
