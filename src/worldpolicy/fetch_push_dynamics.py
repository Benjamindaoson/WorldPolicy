"""Robot-conditioned ensemble dynamics trained on actual FetchPush MuJoCo transitions."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fetch_push import SPEC, FetchPush, PushExpert
from .planning import WorldModelMetadata


class RobotDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, 28))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([state, action], -1))
        prediction = state + delta
        # Goal is exogenous and cannot drift within a candidate rollout.
        return torch.cat([prediction[:, :25], state[:, 25:28]], -1)


def collect_transitions(count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = {"states": [], "actions": [], "next_states": []}
    for episode in range(count):
        env = FetchPush(seed + episode)
        state = env.reset()
        expert = PushExpert()
        for _ in range(env.max_steps):
            action = expert(state) if episode % 2 == 0 else rng.uniform(-1, 1, 4).astype(np.float32)
            next_state, _, terminated, truncated, info = env.step(action)
            rows["states"].append(state)
            rows["actions"].append(action)
            rows["next_states"].append(next_state)
            state = next_state
            if terminated or truncated or info["success"]:
                break
        env.close()
    return {key: np.asarray(value, dtype=np.float32) for key, value in rows.items()}


class LearnedFetchDynamics:
    def __init__(self, models: list[RobotDynamics], state_mean: np.ndarray,
                 state_std: np.ndarray, checkpoint_id: str, horizon: int = 3):
        self.models = [m.eval() for m in models]
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.metadata = WorldModelMetadata("fetch_push_robot_dynamics", "three_mlp_ensemble",
                                           checkpoint_id, "1", SPEC, 28, (28,), horizon,
                                           "state_trajectory", "ensemble_state_variance",
                                           "learned_dynamics")

    @torch.no_grad()
    def predict(self, state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if state.shape != (28,) or actions.ndim != 3 or actions.shape[1:] != (self.metadata.horizon, 4):
            raise ValueError("Fetch learned dynamics state/action protocol mismatch")
        start = (state - self.state_mean) / self.state_std
        ensemble = []
        for model in self.models:
            current = torch.tensor(np.broadcast_to(start, (len(actions), 28)).copy(), dtype=torch.float32)
            trajectory = []
            for t in range(actions.shape[1]):
                current = model(current, torch.tensor(actions[:, t], dtype=torch.float32))
                trajectory.append(current.numpy() * self.state_std + self.state_mean)
            ensemble.append(np.stack(trajectory, 1))
        array = np.stack(ensemble)
        return array.mean(0), array.var(0).mean(-1)


def train_dynamics(output: Path, episodes: int = 64, seed: int = 80000,
                   updates: int = 1500) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    training = collect_transitions(episodes, seed)
    validation = collect_transitions(8, seed + 10000)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "train_transitions.npz", **training)
    np.savez_compressed(output / "validation_transitions.npz", **validation)
    mean = training["states"].mean(0)
    std = np.maximum(training["states"].std(0), .01)
    states = torch.tensor((training["states"] - mean) / std)
    next_states = torch.tensor((training["next_states"] - mean) / std)
    actions = torch.tensor(training["actions"])
    models = []
    losses = []
    for member in range(3):
        torch.manual_seed(seed + member)
        model = RobotDynamics()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        for _ in range(updates):
            index = torch.randint(len(states), (128,))
            loss = (model(states[index], actions[index]) - next_states[index]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        models.append(model)
        losses.append(float(loss.detach()))
    checkpoint = output / "checkpoint.pt"
    torch.save({"models": [m.state_dict() for m in models], "mean": mean, "std": std,
                "episodes": episodes, "updates": updates, "seed": seed, "action_spec": SPEC.__dict__}, checkpoint)
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    adapter = LearnedFetchDynamics(models, mean, std, digest)
    predictions = []
    uncertainties = []
    for state, action in zip(validation["states"], validation["actions"], strict=True):
        future, uncertainty = adapter.predict(state, action[None, None].repeat(3, axis=1))
        predictions.append(future[0, 0])
        uncertainties.append(uncertainty[0, 0])
    mse = float(np.square(np.asarray(predictions) - validation["next_states"]).mean())
    result = {"status": "CPU_VERIFIED", "evidence": "learned_robot_dynamics",
              "environment": "FetchPushDense-v4", "training_episodes": episodes,
              "training_transitions": len(states), "validation_episodes": 8,
              "validation_transitions": len(validation["states"]),
              "updates_per_ensemble_member": updates, "train_losses": losses,
              "held_out_one_step_mse": mse,
              "mean_predicted_uncertainty": float(np.mean(uncertainties)),
              "checkpoint_sha256": digest,
              "train_transitions_sha256": sha256((output / "train_transitions.npz").read_bytes()).hexdigest()}
    (output / "train.json").write_text(json.dumps(result, indent=2))
    return result


def load_dynamics(path: Path, digest: str) -> LearnedFetchDynamics:
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Robot dynamics checkpoint integrity failure")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["action_spec"] != SPEC.__dict__:
        raise ValueError("Robot dynamics action protocol mismatch")
    models = [RobotDynamics() for _ in payload["models"]]
    for model, weights in zip(models, payload["models"], strict=True):
        model.load_state_dict(weights)
    return LearnedFetchDynamics(models, payload["mean"], payload["std"], digest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v12/fetch_push_dynamics"))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--updates", type=int, default=1500)
    args = parser.parse_args()
    print(json.dumps(train_dynamics(args.output, args.episodes, updates=args.updates), indent=2))


if __name__ == "__main__":
    main()
