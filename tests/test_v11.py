import numpy as np
import pytest
import torch

from worldpolicy.data import chunks
from worldpolicy.mujoco_reach import MuJoCoReach, collect_expert
from worldpolicy.planning import CEMPlanner, PhysicsOracle
from worldpolicy.policies import ACT, DiffusionPolicy
from worldpolicy.rl import IQL
from worldpolicy.simulation import SPEC, collect_offline_transitions
from worldpolicy.training import train


def test_mujoco_reset_restore_and_expert():
    env = MuJoCoReach(42)
    first = env.reset()
    snapshot = env.snapshot()
    action = env.expert_action()
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)
    env.step(action)
    env.restore(snapshot)
    np.testing.assert_allclose(env.observe(), first, atol=1e-6)
    assert env.rgb().shape == (96, 96, 3)
    episodes = collect_expert(2, 42)
    assert all(e.success for e in episodes)
    assert all(e.action_spec.unit == "rad/s" for e in episodes)


def test_mujoco_episodes_are_disjoint_and_chunked():
    train = collect_expert(3, 100)
    test = collect_expert(2, 200)
    assert not ({e.episode_id for e in train} & {e.episode_id for e in test})
    x, y = chunks([(e.states, e.actions) for e in train], 4)
    assert x.shape[1] == 8 and y.shape[1:] == (4, 2)


def test_diffusion_schedule_and_shapes():
    model = DiffusionPolicy(timesteps=20)
    assert model.alpha_bar[-1] < 0.01
    state = torch.zeros(3, 4)
    actions = torch.zeros(3, 4, 2)
    loss = model.loss(state, actions)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.sample(state).shape == actions.shape


def test_act_order_and_gradient():
    model = ACT(hidden=32, layers=1)
    loss = model.loss(torch.randn(2, 4), torch.randn(2, 4, 2))
    loss.backward()
    assert model.action_position.grad is not None


def test_iql_timeout_and_terminal_semantics():
    replay = collect_offline_transitions(3, 7)
    assert len(replay["state"]) == len(replay["next_state"])
    assert np.all(replay["terminated"] * replay["timeout"] == 0)
    agent = IQL(4, 2)
    batch = (torch.randn(2, 4), torch.zeros(2, 2), torch.zeros(2, 1),
             torch.randn(2, 4), torch.ones(2, 1), torch.ones(2, 1))
    with pytest.raises(ValueError, match="cannot"):
        agent.update(batch)


def test_cem_seed_repeatability_and_bounds():
    state = np.array([-.5, -.5, 1.15, 1.15])
    a = CEMPlanner(PhysicsOracle(), SPEC, seed=4)
    b = CEMPlanner(PhysicsOracle(), SPEC, seed=4)
    action_a, info_a = a.plan(state)
    action_b, info_b = b.plan(state)
    np.testing.assert_allclose(action_a, action_b)
    assert info_a == info_b
    assert np.all(action_a >= -1) and np.all(action_a <= 1)


def test_checkpoint_resume(tmp_path):
    rng = np.random.default_rng(4)
    states = rng.normal(size=(16, 4)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(16, 4, 2)).astype(np.float32)
    train(ACT(hidden=32, layers=1), states, actions, steps=2, batch_size=4,
          lr=1e-3, seed=4, output=tmp_path)
    record = train(ACT(hidden=32, layers=1), states, actions, steps=4, batch_size=4,
                   lr=1e-3, seed=4, output=tmp_path, resume_from=tmp_path / "checkpoint.pt")
    assert record["resumed_from_step"] == 2
