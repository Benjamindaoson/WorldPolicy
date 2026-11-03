"""Replay a saved visual student trace and a privileged teacher with synchronized RGB video."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from worldpolicy.fetch_push import FetchPush, PushExpert


def record(seed: int, actions: list[list[float]] | None, output: Path) -> dict:
    env = FetchPush(seed, camera_setup="task_visible")
    state = env.reset()
    teacher = PushExpert()
    frames = []
    applied = []
    for step in range(env.max_steps):
        views = env.camera_frames()
        paired = np.concatenate([views[0], views[1]], axis=1)
        frames.append(np.asarray(Image.fromarray(paired).resize((512, 256))))
        if actions is not None and step >= len(actions):
            raise ValueError("Saved student trace ended before simulator termination")
        action = teacher(state) if actions is None else np.asarray(actions[step], dtype=np.float32)
        applied.append(action.tolist())
        state, _, terminated, truncated, info = env.step(action)
        if terminated or truncated or info["success"]:
            break
    env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Video exists; use a new filename")
    imageio.mimwrite(output, frames, fps=25, codec="libx264", quality=7)
    result = {"seed": seed, "success": info["success"], "goal_distance": info["goal_distance"],
              "steps": len(applied), "video_sha256": sha256(output.read_bytes()).hexdigest(),
              "camera_setup": "task_visible", "evidence": "SIMULATION",
              "source": "privileged_teacher" if actions is None else "saved_visual_student_actions"}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", default="visual_proprio_act")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    rows = json.loads((args.run_dir / args.policy / "evaluation.json").read_text())["traces"]
    original = next(row for row in rows if row["seed"] == args.seed)
    output = args.run_dir / "videos"
    student = record(args.seed, original["actions"], output / f"{args.policy}_{args.seed}.mp4")
    if student["success"] != original["success"] or student["steps"] != original["steps"]:
        raise RuntimeError("Visual student replay diverged from evaluation")
    teacher = record(args.seed, None, output / f"teacher_{args.seed}.mp4")
    print(json.dumps({"student": student, "teacher": teacher}, indent=2))


if __name__ == "__main__":
    main()
