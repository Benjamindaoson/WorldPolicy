"""Reproducible local execution entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import chunks, load_dataset, save_dataset
from .planning import CEMPlanner, PhysicsOracle
from .policies import ACT, DiffusionPolicy
from .rl import IQL
from .safety import RobotGate
from .simulation import SPEC, collect_expert, expert_action, rollout
from .training import load_checkpoint, seed_all, train
from .world_model import load_dynamics, train_dynamics


def _policy(kind: str, config: dict):
    common = {"state_dim": 4, "action_dim": SPEC.dimension, "horizon": config["horizon"]}
    if kind == "act":
        return ACT(**common, hidden=config.get("hidden", 64), latent=config.get("latent", 16),
                   layers=config.get("layers", 2), heads=config.get("heads", 4))
    return DiffusionPolicy(**common, hidden=config.get("hidden", 128),
                           timesteps=config.get("timesteps", 20))


def run_training(kind: str, config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    seed_all(config["seed"])
    torch.set_num_threads(1)
    root = Path(config.get("output", "artifacts")) / kind
    dataset_dir = root / "dataset"
    episodes = collect_expert(config["episodes"], config["seed"])
    manifest = save_dataset(episodes, dataset_dir)
    pairs, _ = load_dataset(dataset_dir)
    train_pairs = pairs[:max(1, int(.8 * len(pairs)))]
    test_pairs = pairs[max(1, int(.8 * len(pairs))):]
    states, actions = chunks(train_pairs, config["horizon"])
    model = _policy(kind, config)
    record = train(model, states, actions, steps=config["steps"],
                   batch_size=config["batch_size"], lr=config["lr"],
                   seed=config["seed"], output=root)
    restored = _policy(kind, config)
    load_checkpoint(restored, root / "checkpoint.pt", record["checkpoint_sha256"])
    test_states, test_actions = chunks(test_pairs, config["horizon"])
    with torch.no_grad():
        predictions = restored(torch.tensor(test_states))[0] if kind == "act" else restored.sample(torch.tensor(test_states))
        prediction_mse = float(torch.mean((predictions - torch.tensor(test_actions)) ** 2))
    results = [rollout(lambda s: restored.act(torch.tensor(s[None], dtype=torch.float32))[0].numpy(),
                       config["seed"] + 1000 + i) for i in range(config["evaluation_episodes"])]
    evaluation = {"status": "EXECUTED", "environment": "point_reach_euler_v1",
                  "evidence": "physics_baseline_simulation", "checkpoint_sha256": record["checkpoint_sha256"],
                  "dataset_manifest_episodes": len(manifest["episodes"]),
                  "held_out_action_mse": prediction_mse,
                  "success_rate": sum(x["success"] for x in results) / len(results),
                  "episodes": results}
    (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    return {"training": record, "evaluation": {k: v for k, v in evaluation.items() if k != "episodes"}}


def run_planning(seed: int, episodes: int, output: Path) -> dict:
    oracle = PhysicsOracle()
    planner = CEMPlanner(oracle, SPEC, seed=seed)
    results = [rollout(lambda s: planner.plan(s)[0], seed + i) for i in range(episodes)]
    result = {"status": "EXECUTED", "model_evidence": "physics_oracle",
              "success_rate": sum(x["success"] for x in results) / episodes, "episodes": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k != "episodes"}


def run_iql(seed: int, steps: int, output: Path) -> dict:
    seed_all(seed)
    torch.set_num_threads(1)
    episodes = collect_expert(24, seed)
    transitions = []
    for ep in episodes:
        for i in range(len(ep.actions) - 1):
            distance = np.linalg.norm(ep.states[i + 1, 2:4])
            transitions.append((ep.states[i], ep.actions[i], -distance,
                                ep.states[i + 1], float(i == len(ep.actions) - 2)))
    arrays = [torch.tensor(np.asarray(column), dtype=torch.float32)
              for column in zip(*transitions, strict=True)]
    arrays[2] = arrays[2].unsqueeze(-1)
    arrays[4] = arrays[4].unsqueeze(-1)
    agent = IQL(4, 2)
    metrics = {}
    for _ in range(steps):
        index = torch.randint(len(arrays[0]), (64,))
        metrics = agent.update(tuple(x[index] for x in arrays))
    results = [rollout(lambda s: agent.act(torch.tensor(s[None])).numpy()[0], seed + 1000 + i)
               for i in range(8)]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": agent.actor.state_dict(), "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(), "value": agent.value.state_dict(),
                "steps": steps, "seed": seed}, output.with_suffix(".pt"))
    result = {"status": "EXECUTED", "dataset": "physics_oracle_point_reach",
              "steps": steps, "losses": metrics,
              "success_rate": sum(x["success"] for x in results) / len(results), "episodes": results}
    output.write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k != "episodes"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="worldpolicy")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train-act", "train-diffusion"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
    p = sub.add_parser("plan")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=4)
    p = sub.add_parser("train-iql")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p = sub.add_parser("train-dynamics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--steps", type=int, default=500)
    sub.add_parser("robot-gate")
    args = parser.parse_args()
    if args.command in ("train-act", "train-diffusion"):
        result = run_training(args.command.removeprefix("train-"), args.config)
    elif args.command == "plan":
        result = run_planning(args.seed, args.episodes, Path("results/planning.json"))
    elif args.command == "train-iql":
        result = run_iql(args.seed, args.steps, Path("results/iql.json"))
    elif args.command == "train-dynamics":
        record = train_dynamics(Path("artifacts/dynamics"), args.seed, args.episodes, args.steps)
        model = load_dynamics(Path("artifacts/dynamics/dynamics.pt"), record["checkpoint_sha256"])
        planner = CEMPlanner(model, SPEC, seed=args.seed)
        oracle = CEMPlanner(PhysicsOracle(), SPEC, seed=args.seed)
        seeds = [args.seed + 2000 + i for i in range(8)]
        episodes = {
            "learned_cem": [rollout(lambda s: planner.plan(s)[0], seed) for seed in seeds],
            "oracle_cem": [rollout(lambda s: oracle.plan(s)[0], seed) for seed in seeds],
            "expert": [rollout(expert_action, seed) for seed in seeds],
        }
        result = {**record, "closed_loop_success_rates": {
            name: sum(e["success"] for e in rows) / len(rows) for name, rows in episodes.items()},
            "evaluation_evidence": "paired_held_out_physics_baseline_simulation",
            "evaluation_seeds": seeds, "episodes": episodes}
        Path("artifacts/dynamics/evaluation.json").write_text(json.dumps(result, indent=2))
        result.pop("episodes")
    else:
        try:
            RobotGate().assert_motion_allowed()
        except PermissionError as exc:
            result = {"motion_enabled": False, "reason": str(exc)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
