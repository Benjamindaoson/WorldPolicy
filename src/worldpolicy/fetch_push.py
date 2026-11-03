"""Official Gymnasium-Robotics FetchPush-v4 adapter and analytic demonstration policy."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np

from .data import ActionSpec, Episode

gym.register_envs(gymnasium_robotics)

SPEC = ActionSpec("fetch_push_v4", 4, "normalized_cartesian_delta", "world_xyz+gripper",
                  25.0, (-1., -1., -1., -1.), (1., 1., 1., 1.))


@dataclass(frozen=True)
class FetchSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    warmstart: np.ndarray
    time: float
    goal: np.ndarray
    elapsed_steps: int


def flatten(observation: dict) -> np.ndarray:
    """25D robot/object state plus 3D desired goal; no future information."""
    return np.concatenate([observation["observation"], observation["desired_goal"]]).astype(np.float32)


class FetchPush:
    version = f"FetchPushDense-v4_gymnasium_robotics_{gymnasium_robotics.__version__}_mujoco_{mujoco.__version__}"

    def __init__(self, seed: int, render: bool = False, max_steps: int = 80):
        self.seed = seed
        self.max_steps = max_steps
        self.env = gym.make("FetchPushDense-v4", max_episode_steps=max_steps,
                            render_mode="rgb_array" if render else None)
        self.raw: dict | None = None

    def reset(self) -> np.ndarray:
        self.raw, _ = self.env.reset(seed=self.seed)
        return flatten(self.raw)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("FetchPush action must be finite 4D")
        self.raw, reward, terminated, truncated, info = self.env.step(np.clip(action, -1, 1))
        return flatten(self.raw), float(reward), bool(terminated), bool(truncated), {
            "success": bool(info["is_success"]),
            "goal_distance": float(np.linalg.norm(self.raw["achieved_goal"] - self.raw["desired_goal"])),
        }

    def rgb(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Rendering was not enabled")
        return frame

    def snapshot(self) -> FetchSnapshot:
        raw = self.env.unwrapped
        return FetchSnapshot(raw.data.qpos.copy(), raw.data.qvel.copy(), raw.data.act.copy(),
                             raw.data.ctrl.copy(), raw.data.mocap_pos.copy(),
                             raw.data.mocap_quat.copy(), raw.data.qacc_warmstart.copy(),
                             float(raw.data.time), raw.goal.copy(), self.env._elapsed_steps)

    def restore(self, snapshot: FetchSnapshot) -> np.ndarray:
        raw = self.env.unwrapped
        raw.data.qpos[:] = snapshot.qpos
        raw.data.qvel[:] = snapshot.qvel
        raw.data.act[:] = snapshot.act
        raw.data.ctrl[:] = snapshot.ctrl
        raw.data.mocap_pos[:] = snapshot.mocap_pos
        raw.data.mocap_quat[:] = snapshot.mocap_quat
        raw.data.qacc_warmstart[:] = snapshot.warmstart
        raw.data.time = snapshot.time
        raw.goal = snapshot.goal.copy()
        self.env._elapsed_steps = snapshot.elapsed_steps
        mujoco.mj_forward(raw.model, raw.data)
        self.raw = raw._get_obs()
        return flatten(self.raw)

    def close(self) -> None:
        self.env.close()


class PushExpert:
    """Finite-state geometric push controller; an imperfect, auditable baseline."""

    def __init__(self):
        self.stage = 0

    def __call__(self, state: np.ndarray) -> np.ndarray:
        grip = state[:3]
        obj = state[3:6]
        goal = state[25:28]
        displacement = goal[:2] - obj[:2]
        distance = float(np.linalg.norm(displacement))
        direction = displacement / max(distance, 1e-6)
        behind = obj.copy()
        behind[:2] -= .07 * direction
        behind[2] = obj[2] + .025
        high = behind.copy()
        high[2] = obj[2] + .12
        if self.stage == 0:
            target = high
            if np.linalg.norm(grip - high) < .025:
                self.stage = 1
        elif self.stage == 1:
            target = behind
            if np.linalg.norm(grip - behind) < .02:
                self.stage = 2
        else:
            target = obj.copy()
            target[:2] += direction * min(.12, distance + .05)
            target[2] = behind[2]
        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(10 * (target - grip), -1, 1)
        return action


def rollout(seed: int, policy, *, video: bool = False, max_steps: int = 80) -> dict:
    env = FetchPush(seed, render=video, max_steps=max_steps)
    state = env.reset()
    initial = {"gripper": state[:3].tolist(), "object": state[3:6].tolist(),
               "goal": state[25:28].tolist()}
    states, actions, rewards, frames = [], [], [], []
    success = False
    info = {"goal_distance": float("nan")}
    for _ in range(max_steps):
        if video:
            frames.append(env.rgb())
        action = np.asarray(policy(state), dtype=np.float32)
        states.append(state.tolist())
        actions.append(action.tolist())
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        success = info["success"]
        if success or terminated or truncated:
            break
    result = {"seed": seed, "initial": initial, "states": states, "actions": actions,
              "rewards": rewards, "success": success, "goal_distance": info["goal_distance"],
              "termination": "success" if success else "time_limit",
              "simulator": env.version, "task": "FetchPushDense-v4", "evidence": "SIMULATION"}
    if video:
        result["frames"] = np.asarray(frames, dtype=np.uint8)
    env.close()
    return result


def collect_expert(count: int, seed: int) -> list[Episode]:
    episodes = []
    for index in range(count):
        row = rollout(seed + index, PushExpert())
        n = len(row["actions"])
        episodes.append(Episode(f"fetch_push_{seed}_{index}", "FetchPushDense-v4",
                                SPEC.embodiment, "push the block to the target",
                                np.asarray(row["states"], dtype=np.float32),
                                np.asarray(row["actions"], dtype=np.float32),
                                np.arange(n, dtype=np.float32) / SPEC.frequency_hz,
                                row["success"], "geometric_push_expert",
                                row["simulator"], "simulation", SPEC))
    return episodes
