"""Actual MuJoCo two-joint arm reach task; simulation evidence only."""

from __future__ import annotations

from importlib.resources import files

import mujoco
import numpy as np

from .data import ActionSpec, Episode

SPEC = ActionSpec("mujoco_two_link_v1", 2, "rad/s", "joint_velocity",
                  20.0, (-1.0, -1.0), (1.0, 1.0))


class MuJoCoReach:
    version = f"mujoco_{mujoco.__version__}_two_link_reach_v1"
    max_steps = 80

    def __init__(self, seed: int = 0):
        asset = files("worldpolicy").joinpath("assets/two_link_reach.xml")
        self.model = mujoco.MjModel.from_xml_path(str(asset))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.tip_id = self.model.site("tip").id
        self.target_id = self.model.site("target").id
        self.steps = 0
        self.target = np.zeros(3)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        initial = self.rng.uniform([-.9, -.8], [.1, .8])
        goal = self.rng.uniform([-.8, -.7], [.3, .7])
        self.data.qpos[:] = goal
        mujoco.mj_forward(self.model, self.data)
        self.target = self.data.site_xpos[self.tip_id].copy()
        self.model.site_pos[self.target_id] = self.target
        self.data.qpos[:] = initial
        self.data.qvel[:] = 0
        self.data.ctrl[:] = initial
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        tip = self.data.site_xpos[self.tip_id]
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy(),
                               tip[[0, 2]], (self.target - tip)[[0, 2]]]).astype(np.float32)

    def rgb(self) -> np.ndarray:
        with mujoco.Renderer(self.model, height=96, width=96) as renderer:
            renderer.update_scene(self.data, camera="overview")
            return renderer.render().copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid joint velocity action")
        target_q = np.clip(self.data.qpos + .05 * np.clip(action, -1, 1), -2.4, 2.4)
        self.data.ctrl[:] = target_q
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        distance = float(np.linalg.norm((self.target - self.data.site_xpos[self.tip_id])[[0, 2]]))
        success = distance < .035
        truncated = self.steps >= self.max_steps
        return self.observe(), -distance, success, truncated, {"success": success,
                                                               "distance": distance}

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        return (self.data.qpos.copy(), self.data.qvel.copy(), self.target.copy(), self.steps)

    def restore(self, snapshot: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> None:
        self.data.qpos[:] = snapshot[0]
        self.data.qvel[:] = snapshot[1]
        self.target = snapshot[2].copy()
        self.model.site_pos[self.target_id] = self.target
        self.steps = snapshot[3]
        mujoco.mj_forward(self.model, self.data)

    def expert_action(self) -> np.ndarray:
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.tip_id)
        error = (self.target - self.data.site_xpos[self.tip_id])[[0, 2]]
        velocity = np.linalg.pinv(jac[[0, 2], :], rcond=.05) @ (5 * error)
        return np.clip(velocity, -1, 1).astype(np.float32)


def collect_expert(count: int, seed: int = 0) -> list[Episode]:
    result = []
    for index in range(count):
        env = MuJoCoReach(seed + index)
        state = env.reset()
        states, actions, timestamps = [], [], []
        for t in range(env.max_steps):
            action = env.expert_action()
            states.append(state)
            actions.append(action)
            timestamps.append(t / SPEC.frequency_hz)
            state, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        result.append(Episode(f"mujoco_expert_{seed}_{index}", "two_link_reach",
                              SPEC.embodiment, "move end effector to green target",
                              np.asarray(states), np.asarray(actions), np.asarray(timestamps),
                              done, "mujoco_jacobian_expert", env.version, "simulation", SPEC))
    return result


def rollout(policy, seed: int, record_rgb: bool = False) -> dict:
    env = MuJoCoReach(seed)
    state = env.reset()
    initial = env.snapshot()[0].tolist()
    states, actions, rewards, frames = [], [], [], []
    for _ in range(env.max_steps):
        action = np.asarray(policy(state), dtype=np.float32).reshape(-1)
        if record_rgb:
            frames.append(env.rgb())
        states.append(state.tolist())
        actions.append(action.tolist())
        state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        if done or truncated:
            break
    result = {"seed": seed, "initial_joint_state": initial, "target": env.target.tolist(),
              "states": states, "actions": actions, "rewards": rewards,
              "success": info["success"], "termination": "success" if done else "time_limit",
              "distance": info["distance"], "simulator": env.version, "evidence": "SIMULATION"}
    if record_rgb:
        result["frames"] = np.asarray(frames)
    return result
