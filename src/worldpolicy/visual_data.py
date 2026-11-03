"""Synchronized dual-camera FetchPush demonstrations with a strict student input boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .fetch_push import SPEC, FetchPush, PushExpert

CAMERAS = ("head_camera_rgb", "gripper_camera_rgb")
TASK_VISIBLE_CAMERAS = ("external_camera_0", "wrist_follow_camera")
PROPRIO_INDICES = (0, 1, 2, 9, 10, 20, 21, 22, 23, 24)


def proprio_from_state(state: np.ndarray) -> np.ndarray:
    if state.shape != (28,):
        raise ValueError("Expected 28D FetchPush simulator observation")
    return state[list(PROPRIO_INDICES)].astype(np.float32)


@dataclass
class VisualEpisode:
    episode_id: str
    seed: int
    images: np.ndarray  # [T, 2, H, W, 3], uint8
    proprio: np.ndarray  # [T, 10]
    actions: np.ndarray  # [T, 4]
    timestamps: np.ndarray  # [T]
    success: bool

    def validate(self) -> None:
        n = len(self.actions)
        if (n < 1 or self.images.shape != (n, 2, 64, 64, 3)
                or self.images.dtype != np.uint8 or self.proprio.shape != (n, 10)
                or self.actions.shape != (n, 4) or self.timestamps.shape != (n,)):
            raise ValueError("Visual episode shape or synchronization mismatch")
        if not np.isfinite(self.proprio).all() or not np.isfinite(self.actions).all():
            raise ValueError("Nonfinite visual episode state/action")
        if np.any(np.abs(self.actions) > 1 + 1e-6):
            raise ValueError("Action exceeds FetchPush bounds")


def collect_visual_episode(seed: int, camera_shift: tuple[float, float, float] = (0, 0, 0),
                           camera_setup: str = "legacy") -> VisualEpisode:
    env = FetchPush(seed, camera_shift=camera_shift, camera_setup=camera_setup)
    state = env.reset()
    teacher = PushExpert()
    images, proprio, actions = [], [], []
    for _ in range(env.max_steps):
        images.append(env.camera_frames())
        proprio.append(proprio_from_state(state))
        action = teacher(state)
        actions.append(action)
        state, _, terminated, truncated, info = env.step(action)
        if info["success"] or terminated or truncated:
            break
    episode = VisualEpisode(f"visual_fetch_{seed}", seed, np.asarray(images, dtype=np.uint8),
                            np.asarray(proprio, dtype=np.float32), np.asarray(actions, dtype=np.float32),
                            np.arange(len(actions), dtype=np.float32) / SPEC.frequency_hz,
                            info["success"])
    env.close()
    episode.validate()
    return episode


def save_visual_dataset(episodes: list[VisualEpisode], directory: Path,
                        camera_setup: str = "legacy") -> dict:
    if not episodes:
        raise ValueError("Visual dataset cannot be empty")
    if len({ep.seed for ep in episodes}) != len(episodes):
        raise ValueError("Duplicate visual episode seed")
    directory.mkdir(parents=True, exist_ok=True)
    calibration_env = FetchPush(episodes[0].seed, camera_setup=camera_setup)
    calibration = {}
    cameras = CAMERAS if camera_setup == "legacy" else TASK_VISIBLE_CAMERAS
    for name in cameras:
        if name == "wrist_follow_camera":
            calibration[name] = {"mount": "proprioception-following free camera",
                                 "lookat": "current gripper xyz", "distance_m": .6,
                                 "azimuth_degrees": 180, "elevation_degrees": -45}
            continue
        camera = calibration_env.env.unwrapped.model.camera(name)
        calibration[name] = {"local_position_m": camera.pos.tolist(),
                             "local_quaternion_wxyz": camera.quat.tolist(),
                             "field_of_view_degrees": float(camera.fovy[0])}
    calibration_env.close()
    manifest = {"schema": 1, "task": "FetchPushDense-v4", "cameras": cameras,
                "camera_setup": camera_setup,
                "goal_specification": "red rendered target marker" if camera_setup == "task_visible"
                else "virtual numeric target, not visually rendered",
                "resolution": [64, 64], "proprio_indices_in_official_observation": PROPRIO_INDICES,
                "camera_calibration_frame": "MuJoCo model local; wrist world pose varies with robot",
                "camera_calibration": calibration,
                "action_frequency_hz": SPEC.frequency_hz,
                "student_input": "RGB plus optional robot proprioception; no object/goal coordinates",
                "episodes": []}
    for episode in episodes:
        episode.validate()
        path = directory / f"{episode.episode_id}.npz"
        np.savez_compressed(path, images=episode.images, proprio=episode.proprio,
                            actions=episode.actions, timestamps=episode.timestamps)
        manifest["episodes"].append({"id": episode.episode_id, "seed": episode.seed,
                                      "steps": len(episode.actions), "success": episode.success,
                                      "sha256": sha256(path.read_bytes()).hexdigest()})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_visual_dataset(directory: Path) -> tuple[list[VisualEpisode], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    episodes = []
    for row in manifest["episodes"]:
        path = directory / f"{row['id']}.npz"
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Visual dataset integrity failure")
        with np.load(path) as data:
            episode = VisualEpisode(row["id"], row["seed"], data["images"],
                                    data["proprio"], data["actions"], data["timestamps"],
                                    row["success"])
        episode.validate()
        episodes.append(episode)
    return episodes, manifest


def visual_chunks(episodes: list[VisualEpisode], history: int = 2,
                  horizon: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images, proprio, actions = [], [], []
    for episode in episodes:
        for t in range(len(episode.actions) - horizon + 1):
            indices = [max(0, t - history + 1 + offset) for offset in range(history)]
            images.append(episode.images[indices])
            proprio.append(episode.proprio[indices])
            actions.append(episode.actions[t:t + horizon])
    if not images:
        raise ValueError("No complete visual action chunks")
    return np.asarray(images, dtype=np.uint8), np.asarray(proprio, dtype=np.float32), np.asarray(actions, dtype=np.float32)
