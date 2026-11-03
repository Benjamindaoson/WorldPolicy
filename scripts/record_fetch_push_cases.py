"""Replay one success and one failure for each saved FetchPush policy."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from worldpolicy.fetch_push import rollout
from worldpolicy.fetch_push_experiment import make_model
from worldpolicy.training import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("artifacts/v12/fetch_push_b/seed_21"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    normalization = np.load(args.run / "state_normalization.npz") if (
        args.run / "state_normalization.npz").exists() else None
    for kind in ("act", "diffusion"):
        directory = args.run / kind
        evaluation = json.loads((directory / "evaluation.json").read_text())
        model = make_model(kind, 4)
        digest = evaluation["training"]["checkpoint_sha256"]
        load_checkpoint(model, directory / "checkpoint.pt", digest)
        for label, expected in (("success", True), ("failure", False)):
            candidates = [row for row in evaluation["confirmatory_traces"]
                          if row["success"] == expected]
            if not candidates:
                continue
            original = candidates[0]
            seed = original["seed"]
            torch.manual_seed(seed + 700000)

            def policy(state: np.ndarray, trained_model=model) -> np.ndarray:
                if normalization is not None:
                    state = (state - normalization["mean"]) / normalization["std"]
                return trained_model.act(torch.tensor(state[None], dtype=torch.float32))[0].numpy()

            replay = rollout(seed, policy, video=True)
            if replay["success"] != original["success"] or not np.allclose(
                replay["actions"], original["actions"], atol=1e-5
            ):
                raise RuntimeError(f"Video replay diverged for {kind} seed {seed}")
            frames = replay.pop("frames")
            small = [np.asarray(Image.fromarray(frame).resize((240, 240))) for frame in frames]
            video_path = directory / f"{label}_{seed}.mp4"
            imageio.mimwrite(video_path, small, fps=25, codec="libx264", quality=7)
            metadata = {"kind": kind, "label": label, "seed": seed, "frames": len(frames),
                        "checkpoint_sha256": digest, "video": str(video_path),
                        "trajectory": replay}
            video_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
            print(json.dumps({k: v for k, v in metadata.items() if k != "trajectory"}))


if __name__ == "__main__":
    main()
