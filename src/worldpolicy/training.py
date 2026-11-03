"""Shared supervised trainer with checkpoint integrity and run metadata."""

from __future__ import annotations

import json
import random
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model: torch.nn.Module, states: np.ndarray, actions: np.ndarray, *,
          steps: int, batch_size: int, lr: float, seed: int, output: Path,
          kl_weight: float = .001) -> dict:
    seed_all(seed)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    y = torch.as_tensor(actions, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    started = time.perf_counter()
    for _ in range(steps):
        ix = torch.randint(len(x), (batch_size,), device=device)
        kwargs = {"kl_weight": kl_weight} if hasattr(model, "posterior") else {}
        loss: Tensor = model.loss(x[ix], y[ix], **kwargs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    checkpoint = output / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "steps": steps, "seed": seed, "architecture": type(model).__name__}, checkpoint)
    record = {"status": "EXECUTED", "architecture": type(model).__name__, "device": device,
              "seed": seed, "steps": steps, "initial_loss": losses[0], "final_loss": losses[-1],
              "seconds": time.perf_counter() - started, "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest()}
    (output / "train.json").write_text(json.dumps(record, indent=2))
    return record


def load_checkpoint(model: torch.nn.Module, path: Path, expected_sha256: str | None = None) -> dict:
    if expected_sha256 and sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["architecture"] != type(model).__name__:
        raise ValueError("Checkpoint architecture mismatch")
    model.load_state_dict(payload["model"])
    model.eval()
    return payload
