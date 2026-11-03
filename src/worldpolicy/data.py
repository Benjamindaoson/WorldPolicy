"""Versioned episode format and strict action protocol validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    embodiment: str
    dimension: int
    unit: str
    frame: str
    frequency_hz: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def validate(self) -> None:
        if self.dimension < 1 or self.frequency_hz <= 0:
            raise ValueError("Invalid action dimension or frequency")
        if len(self.lower) != self.dimension or len(self.upper) != self.dimension:
            raise ValueError("Action bounds dimension mismatch")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("Invalid action bounds")


@dataclass
class Episode:
    episode_id: str
    task_id: str
    embodiment: str
    instruction: str
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    success: bool
    source: str
    dataset_version: str
    evidence_type: str
    action_spec: ActionSpec
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    source_hash: str = ""

    def validate(self) -> None:
        self.action_spec.validate()
        n = len(self.actions)
        if n < 1 or self.states.shape[0] != n or self.timestamps.shape != (n,):
            raise ValueError("Episode length mismatch")
        if self.actions.shape[1] != self.action_spec.dimension:
            raise ValueError("Action dimension mismatch")
        if self.rgb is not None and len(self.rgb) != n:
            raise ValueError("RGB synchronization mismatch")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("Nonfinite state or action")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Nonmonotonic timestamps")
        if np.any(self.actions < np.array(self.action_spec.lower) - 1e-6) or np.any(
            self.actions > np.array(self.action_spec.upper) + 1e-6
        ):
            raise ValueError("Action outside protocol bounds")


def save_dataset(episodes: list[Episode], directory: Path) -> dict:
    """Save compact NPZ episodes and a content-hash manifest, rejecting duplicate IDs."""
    directory.mkdir(parents=True, exist_ok=True)
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("Duplicate episode ID")
    manifest = {"schema": 1, "episodes": []}
    for episode in episodes:
        episode.validate()
        path = directory / f"{episode.episode_id}.npz"
        np.savez_compressed(path, states=episode.states, actions=episode.actions,
                            timestamps=episode.timestamps)
        digest = sha256(path.read_bytes()).hexdigest()
        manifest["episodes"].append({
            "id": episode.episode_id, "task": episode.task_id,
            "success": episode.success, "source": episode.source,
            "version": episode.dataset_version, "evidence": episode.evidence_type,
            "action_spec": asdict(episode.action_spec), "sha256": digest,
        })
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_dataset(directory: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    result = []
    for row in manifest["episodes"]:
        path = directory / f"{row['id']}.npz"
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Dataset integrity failure: {path}")
        with np.load(path) as data:
            states, actions = data["states"], data["actions"]
        if len(states) != len(actions) or actions.shape[1] != row["action_spec"]["dimension"]:
            raise ValueError("Dataset shape/protocol mismatch")
        result.append((states, actions))
    return result, manifest


def split_episodes(ids: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Stable episode-level split, never sharing an episode between partitions."""
    ordered = list(np.random.default_rng(seed).permutation(sorted(ids)))
    n = len(ordered)
    return {"train": ordered[: max(1, int(.7*n))],
            "validation": ordered[max(1, int(.7*n)):int(.85*n)],
            "test": ordered[int(.85*n):]}


def chunks(episodes: list[tuple[np.ndarray, np.ndarray]], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for states, actions in episodes:
        for t in range(len(actions) - horizon + 1):
            xs.append(states[t])
            ys.append(actions[t:t+horizon])
    if not xs:
        raise ValueError("No complete action chunks")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
