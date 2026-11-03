"""Matched ACT / DDPM MuJoCo training and closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import chunks, load_dataset, save_dataset
from .mujoco_reach import MuJoCoReach, collect_expert, rollout
from .policies import ACT, DiffusionPolicy
from .statistics import paired_counts, wilson_interval
from .training import load_checkpoint, seed_all, train


def run(output: Path, train_episodes: int = 64, updates: int = 500,
        evaluation_episodes: int = 12, seed: int = 11) -> dict:
    torch.set_num_threads(1)
    seed_all(seed)
    output.mkdir(parents=True, exist_ok=True)
    train_seed = seed + 1000
    eval_seeds = [seed + 2000 + i for i in range(evaluation_episodes)]
    dataset = output / "dataset"
    manifest = save_dataset(collect_expert(train_episodes, train_seed), dataset)
    pairs, _ = load_dataset(dataset)
    train_pairs = pairs[: int(.8 * len(pairs))]
    validation_pairs = pairs[int(.8 * len(pairs)):]
    x, y = chunks(train_pairs, 4)
    vx, vy = chunks(validation_pairs, 4)
    outcome = {"status": "CPU_VERIFIED", "task": "mujoco_two_link_reach_v1",
               "train_episode_seed_range": [train_seed, train_seed + train_episodes - 1],
               "eval_seeds": eval_seeds, "dataset_episode_count": len(manifest["episodes"]),
               "policies": {}}
    for kind in ("act", "diffusion"):
        seed_all(seed)
        if kind == "act":
            model = ACT(8, 2, 4, hidden=64, latent=16, layers=2)
            restored = ACT(8, 2, 4, hidden=64, latent=16, layers=2)
        else:
            model = DiffusionPolicy(8, 2, 4, hidden=128, timesteps=20)
            restored = DiffusionPolicy(8, 2, 4, hidden=128, timesteps=20)
        policy_dir = output / kind
        training = train(model, x, y, steps=updates, batch_size=64, lr=5e-4,
                         seed=seed, output=policy_dir)
        load_checkpoint(restored, policy_dir / "checkpoint.pt", training["checkpoint_sha256"])
        with torch.no_grad():
            validation = restored(torch.tensor(vx))[0] if kind == "act" else restored.sample(torch.tensor(vx))
            action_mse = float(torch.square(validation - torch.tensor(vy)).mean())
        episodes = [rollout(lambda s, policy=restored: policy.act(torch.tensor(s[None], dtype=torch.float32))[0].numpy(),
                            eval_seed) for eval_seed in eval_seeds]
        success = sum(e["success"] for e in episodes)
        detail = {"training": training, "validation_action_mse": action_mse,
                  "evaluation_successes": success, "evaluation_episodes": evaluation_episodes,
                  "success_rate": success / evaluation_episodes,
                  "wilson_95": wilson_interval(success, evaluation_episodes),
                  "episodes": episodes}
        (policy_dir / "evaluation.json").write_text(json.dumps(detail, indent=2))
        outcome["policies"][kind] = {k: v for k, v in detail.items() if k != "episodes"}
    expert = []
    for eval_seed in eval_seeds:
        env = MuJoCoReach(eval_seed)
        env.reset()
        for _ in range(env.max_steps):
            _, _, done, truncated, info = env.step(env.expert_action())
            if done or truncated:
                break
        expert.append({"seed": eval_seed, "success": info["success"],
                       "final_distance": info["distance"]})
    outcome["expert"] = {"successes": sum(e["success"] for e in expert),
                         "episodes": expert}
    act_rows = json.loads((output / "act" / "evaluation.json").read_text())["episodes"]
    diffusion_rows = json.loads((output / "diffusion" / "evaluation.json").read_text())["episodes"]
    outcome["paired_act_diffusion"] = paired_counts(
        [e["success"] for e in act_rows], [e["success"] for e in diffusion_rows])
    (output / "summary.json").write_text(json.dumps(outcome, indent=2))
    return {**outcome, "expert": {"successes": outcome["expert"]["successes"],
                                "episode_count": evaluation_episodes}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v11/mujoco_reach"))
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.train_episodes, args.updates,
                         args.evaluation_episodes, args.seed), indent=2))


if __name__ == "__main__":
    main()
