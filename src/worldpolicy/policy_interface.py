"""Versioned visual policy contract for read-only consumers such as WorldSystems."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from .fetch_push import SPEC, FetchPush
from .visual_experiment import load_visual
from .visual_policy import VisualPolicy


def export_visual_contract(run_dir: Path, policy_name: str) -> Path:
    if policy_name not in ("vision_only_act", "visual_proprio_act", "visual_proprio_diffusion"):
        raise ValueError("Unknown visual policy ID")
    summary = json.loads((run_dir / "summary.json").read_text())
    detail = summary["policies"][policy_name]
    train = detail["training"]
    checkpoint = run_dir / policy_name / "checkpoint.pt"
    if sha256(checkpoint.read_bytes()).hexdigest() != train["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after evaluation")
    contract = {
        "schema_version": 1,
        "policy_id": policy_name,
        "architecture": "CNN+ACT-CVAE" if policy_name.endswith("act") else "CNN+MLP-DDPM",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": train["checkpoint_sha256"],
        "observation_schema": {"images": [2, 2, 3, 64, 64], "dtype": "uint8",
                               "camera_order": summary["camera_names"],
                               "proprio": None if policy_name == "vision_only_act" else [2, 10],
                               "privileged_object_goal_state": False},
        "action_schema": {"shape": [4], "bounds": [-1.0, 1.0], "control": SPEC.unit,
                          "frame": SPEC.frame},
        "action_frequency_hz": SPEC.frequency_hz,
        "action_chunk_length": 4,
        "execution": "first action of predicted chunk, replan every simulator step",
        "precision": "float32",
        "inference_interface": "worldpolicy.policy_interface.load_visual_contract(path).act(images, proprio)",
        "evaluation_protocol": summary["protocol_id"],
        "environment_version": FetchPush.version,
        "source_commit": summary["source_commit"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "training_seed": train["seed"],
        "evaluation_seeds": summary["evaluation_seeds"],
        "evaluation_successes": detail["successes"],
        "evaluation_episodes": detail["episodes"],
        "evidence": "CPU_VERIFIED_SIMULATION",
    }
    path = run_dir / policy_name / "policy_contract.json"
    path.write_text(json.dumps(contract, indent=2))
    return path


class VisualContractAdapter:
    def __init__(self, contract: dict):
        self.contract = contract
        policy_name = contract["policy_id"]
        model = VisualPolicy("diffusion" if policy_name.endswith("diffusion") else "act",
                             policy_name != "vision_only_act")
        load_visual(model, Path(contract["checkpoint_path"]), contract["checkpoint_sha256"])
        self.model = model

    def act(self, images: np.ndarray, proprio: np.ndarray | None = None) -> np.ndarray:
        """Images are [history=2,cameras=2,channels=3,64,64]; no simulator state is accepted."""
        if images.shape != (2, 2, 3, 64, 64) or images.dtype != np.uint8:
            raise ValueError("Visual observation contract mismatch")
        robot = None
        if self.model.use_proprio:
            if proprio is None or proprio.shape != (2, 10):
                raise ValueError("Proprioception contract mismatch")
            robot = torch.tensor(proprio[None], dtype=torch.float32)
        action = self.model.act(torch.tensor(images[None]), robot)[0].numpy()
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Policy emitted invalid FetchPush action")
        return action


def load_visual_contract(path: Path) -> VisualContractAdapter:
    contract = json.loads(path.read_text())
    if contract["schema_version"] != 1 or contract["action_schema"]["shape"] != [4]:
        raise ValueError("Unsupported policy contract")
    return VisualContractAdapter(contract)
