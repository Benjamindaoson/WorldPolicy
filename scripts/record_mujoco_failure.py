"""Record the first failed ACT evaluation episode with the saved checkpoint."""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from worldpolicy.mujoco_reach import rollout
from worldpolicy.policies import ACT
from worldpolicy.training import load_checkpoint


def main() -> None:
    base = Path("artifacts/v11/mujoco_reach/act")
    evaluation = json.loads((base / "evaluation.json").read_text())
    failures = [row for row in evaluation["episodes"] if not row["success"]]
    if not failures:
        raise RuntimeError("No failed ACT episode to record")
    seed = failures[0]["seed"]
    model = ACT(8, 2, 4, hidden=64, latent=16, layers=2)
    digest = evaluation["training"]["checkpoint_sha256"]
    load_checkpoint(model, base / "checkpoint.pt", digest)
    result = rollout(lambda state: model.act(torch.tensor(state[None], dtype=torch.float32))[0].numpy(),
                     seed, record_rgb=True)
    if result["success"] != failures[0]["success"]:
        raise RuntimeError("Failure replay did not reproduce")
    frames = np.asarray(result.pop("frames"), dtype=np.uint8)
    video = base / f"failed_seed_{seed}.mp4"
    imageio.mimwrite(video, frames, fps=20, codec="libx264", quality=7)
    record = {"seed": seed, "success": result["success"], "frames": len(frames),
              "checkpoint_sha256": digest, "video": str(video), "trajectory": result}
    (base / f"failed_seed_{seed}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({key: value for key, value in record.items() if key != "trajectory"}, indent=2))


if __name__ == "__main__":
    main()
