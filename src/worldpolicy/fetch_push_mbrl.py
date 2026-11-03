"""Compact FetchPush model-based actor-critic with real replay and filtered imagination."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fetch_push import FetchPush, PushExpert
from .fetch_push_dynamics import load_dynamics


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(28, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, 4), nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], -1))


class RewardTermination(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, 2))

    def forward(self, next_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.net(torch.cat([next_state, action], -1))
        return value[:, :1], value[:, 1:]


def bootstrap_target(reward: torch.Tensor, terminated: torch.Tensor,
                     next_value: torch.Tensor, gamma: float = .98) -> torch.Tensor:
    """Time-limit truncation still bootstraps; only true success termination masks value."""
    return reward + gamma * (1 - terminated) * next_value


def collect_real_replay(episodes: int, seed_start: int, output: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list] = {k: [] for k in ("state", "action", "next_state", "reward",
                                                "terminated", "truncated", "expert")}
    rng = np.random.default_rng(seed_start)
    for episode in range(episodes):
        env = FetchPush(seed_start + episode)
        state = env.reset()
        teacher = PushExpert()
        use_expert = episode % 2 == 0
        for _ in range(env.max_steps):
            action = teacher(state) if use_expert else rng.uniform(-1, 1, 4).astype(np.float32)
            next_state, reward, terminated, truncated, info = env.step(action)
            for key, value in (("state", state), ("action", action),
                               ("next_state", next_state), ("reward", reward),
                               ("terminated", terminated or info["success"]),
                               ("truncated", truncated), ("expert", use_expert)):
                rows[key].append(value)
            state = next_state
            if terminated or truncated or info["success"]:
                break
        env.close()
    result = {key: np.asarray(value, dtype=np.float32) for key, value in rows.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    return result


def imagine(ensemble, actor: Actor, states: torch.Tensor, cutoff: float,
            horizon: int = 3) -> tuple[dict[str, torch.Tensor], int]:
    """Roll out candidate actions through learned dynamics only; discard uncertain branches."""
    current = states.clone()
    rows: dict[str, list[torch.Tensor]] = {k: [] for k in ("state", "action", "next_state", "uncertainty")}
    rejected = 0
    with torch.no_grad():
        for _ in range(horizon):
            action = actor(current)
            predictions = torch.stack([model(current, action) for model in ensemble])
            uncertainty = predictions.var(0, unbiased=False).mean(-1)
            kept = uncertainty <= cutoff
            rejected += int((~kept).sum())
            if not kept.any():
                break
            for key, value in (("state", current), ("action", action),
                               ("next_state", predictions.mean(0)), ("uncertainty", uncertainty)):
                rows[key].append(value[kept])
            current = predictions.mean(0)[kept]
    if not rows["state"]:
        return {}, rejected
    return {key: torch.cat(value) for key, value in rows.items()}, rejected


def evaluate_actor(actor: Actor, mean: np.ndarray, std: np.ndarray, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        env = FetchPush(seed)
        state = env.reset()
        for step in range(env.max_steps):
            with torch.no_grad():
                action = actor(torch.tensor(((state - mean) / std)[None], dtype=torch.float32))[0].numpy()
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated or info["success"]:
                break
        rows.append({"seed": seed, "success": info["success"],
                     "goal_distance": info["goal_distance"], "steps": step + 1})
        env.close()
    return rows


def run(output: Path, dynamics_checkpoint: Path, episodes: int = 24,
        warmup_updates: int = 400, policy_updates: int = 500) -> dict:
    """Train on fresh real replay, then policy updates with one-step/short imagined samples."""
    torch.set_num_threads(1)
    torch.manual_seed(413)
    np.random.seed(413)
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    replay_path = output / "real_replay.npz"
    replay = collect_real_replay(episodes, 94000, replay_path)
    dynamics_record = json.loads(dynamics_checkpoint.with_name("train.json").read_text())
    dynamics = load_dynamics(dynamics_checkpoint, dynamics_record["checkpoint_sha256"])
    for model in dynamics.models:
        model.requires_grad_(False)
    mean, std = dynamics.state_mean, dynamics.state_std
    state = torch.tensor((replay["state"] - mean) / std)
    next_state = torch.tensor((replay["next_state"] - mean) / std)
    action = torch.tensor(replay["action"])
    reward = torch.tensor(replay["reward"][:, None])
    terminated = torch.tensor(replay["terminated"][:, None])
    expert = torch.tensor(replay["expert"] > 0)
    reward_term = RewardTermination()
    rt_opt = torch.optim.AdamW(reward_term.parameters(), lr=5e-4)
    for _ in range(400):
        index = torch.randint(len(state), (128,))
        r, logit = reward_term(next_state[index], action[index])
        loss = (r - reward[index]).square().mean() + nn.functional.binary_cross_entropy_with_logits(
            logit, terminated[index], pos_weight=torch.tensor([5.0]))
        rt_opt.zero_grad(set_to_none=True)
        loss.backward()
        rt_opt.step()
    actor, critic = Actor(), Critic()
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=3e-4)
    expert_indices = torch.where(expert)[0]
    for _ in range(warmup_updates):
        index = expert_indices[torch.randint(len(expert_indices), (128,))]
        loss = (actor(state[index]) - action[index]).square().mean()
        actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_opt.step()
    bc_actor = copy.deepcopy(actor).eval()
    target_actor, target_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    with torch.no_grad():
        preds = torch.stack([model(state, action) for model in dynamics.models])
        real_uncertainty = preds.var(0, unbiased=False).mean(-1)
        cutoff = float(torch.quantile(real_uncertainty, .75))
    imagined_count = rejected_count = 0
    actor_losses, critic_losses = [], []
    for update in range(policy_updates):
        real_index = torch.randint(len(state), (64,))
        starts = state[torch.randint(len(state), (64,))]
        fake, rejected = imagine(dynamics.models, actor, starts, cutoff, horizon=3)
        rejected_count += rejected
        states = state[real_index]
        actions = action[real_index]
        next_states = next_state[real_index]
        rewards = reward[real_index]
        dones = terminated[real_index]
        if fake:
            n = min(64, len(fake["state"]))
            idx = torch.randperm(len(fake["state"]))[:n]
            fake_state, fake_action, fake_next = (fake[key][idx] for key in ("state", "action", "next_state"))
            with torch.no_grad():
                fake_reward, fake_logit = reward_term(fake_next, fake_action)
            states = torch.cat([states, fake_state])
            actions = torch.cat([actions, fake_action])
            next_states = torch.cat([next_states, fake_next])
            rewards = torch.cat([rewards, fake_reward])
            dones = torch.cat([dones, (fake_logit.sigmoid() > .5).float()])
            imagined_count += n
        with torch.no_grad():
            target = bootstrap_target(rewards, dones, target_critic(
                next_states, target_actor(next_states)))
        critic_loss = (critic(states, actions) - target).square().mean()
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        actor_index = expert_indices[torch.randint(len(expert_indices), (64,))]
        actor_loss = -critic(state[actor_index], actor(state[actor_index])).mean()
        actor_loss += .1 * (actor(state[actor_index]) - action[actor_index]).square().mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()
        with torch.no_grad():
            for target_param, param in zip(target_actor.parameters(), actor.parameters(), strict=True):
                target_param.lerp_(param, .01)
            for target_param, param in zip(target_critic.parameters(), critic.parameters(), strict=True):
                target_param.lerp_(param, .01)
        if update % 50 == 0:
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
    checkpoint = output / "checkpoint.pt"
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "reward_termination": reward_term.state_dict(),
                "target_actor": target_actor.state_dict(), "target_critic": target_critic.state_dict(),
                "bc_actor": bc_actor.state_dict(), "actor_optimizer": actor_opt.state_dict(),
                "critic_optimizer": critic_opt.state_dict(), "mean": mean, "std": std,
                "policy_updates": policy_updates, "warmup_updates": warmup_updates}, checkpoint)
    seeds = list(range(95000, 95010))
    bc_rows = evaluate_actor(bc_actor, mean, std, seeds)
    mbrl_rows = evaluate_actor(actor.eval(), mean, std, seeds)
    report = {"status": "CPU_VERIFIED", "task": "FetchPushDense-v4",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "dynamics_checkpoint_sha256": dynamics_record["checkpoint_sha256"],
              "real_replay_sha256": sha256(replay_path.read_bytes()).hexdigest(),
              "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
              "real_episodes": episodes, "real_transitions": len(state),
              "reward_termination_updates": 400, "bc_warmup_updates": warmup_updates,
              "policy_updates": policy_updates, "imagined_samples_used": imagined_count,
              "imagined_branches_rejected": rejected_count, "uncertainty_cutoff": cutoff,
              "actor_losses": actor_losses, "critic_losses": critic_losses,
              "evaluation_seeds": seeds, "bc_warmup_successes": sum(row["success"] for row in bc_rows),
              "mbrl_successes": sum(row["success"] for row in mbrl_rows),
              "bc_warmup_traces": bc_rows, "mbrl_traces": mbrl_rows,
              "training_and_evaluation_seconds": time.perf_counter() - start,
              "note": "Exploratory single training seed; not a validated MBRL improvement"}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v13/fetch_push_mbrl"))
    parser.add_argument("--dynamics-checkpoint", type=Path,
                        default=Path("artifacts/v12/fetch_push_dynamics/checkpoint.pt"))
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--warmup-updates", type=int, default=400)
    parser.add_argument("--policy-updates", type=int, default=500)
    args = parser.parse_args()
    result = run(args.output, args.dynamics_checkpoint, args.episodes,
                 args.warmup_updates, args.policy_updates)
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("traces")}, indent=2))


if __name__ == "__main__":
    main()
