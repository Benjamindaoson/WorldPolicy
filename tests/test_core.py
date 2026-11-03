import numpy as np
import pytest
import torch

from worldpolicy.data import ActionSpec, chunks, load_dataset, save_dataset, split_episodes
from worldpolicy.planning import CEMPlanner, PhysicsOracle
from worldpolicy.policies import ACT, DiffusionPolicy
from worldpolicy.rl import IQL
from worldpolicy.safety import RobotGate
from worldpolicy.simulation import SPEC, collect_expert, rollout
from worldpolicy.training import load_checkpoint, train


def test_episode_and_integrity(tmp_path):
    episodes = collect_expert(5)
    save_dataset(episodes, tmp_path)
    pairs, manifest = load_dataset(tmp_path)
    assert len(pairs) == len(manifest["episodes"]) == 5
    assert len(chunks(pairs, 4)[0]) > 0
    splits = split_episodes([e.episode_id for e in episodes])
    assert len(set(splits["train"]) & set(splits["test"])) == 0
    file = next(tmp_path.glob("*.npz"))
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("model", [ACT(hidden=32, layers=1), DiffusionPolicy(hidden=32, timesteps=4)])
def test_policy_gradient_and_reload(model, tmp_path):
    x = np.random.default_rng(0).normal(size=(16, 4)).astype("float32")
    y = np.random.default_rng(1).uniform(-1, 1, (16, 4, 2)).astype("float32")
    record = train(model, x, y, steps=2, batch_size=4, lr=1e-3, seed=0, output=tmp_path)
    assert np.isfinite(record["final_loss"])
    restored = ACT(hidden=32, layers=1) if isinstance(model, ACT) else DiffusionPolicy(hidden=32, timesteps=4)
    load_checkpoint(restored, tmp_path / "checkpoint.pt", record["checkpoint_sha256"])
    assert restored.act(torch.tensor(x[:1])).shape == (1, 2)


def test_act_deterministic_inference_and_dimension():
    model = ACT(hidden=32, layers=1)
    x = torch.ones(2, 4)
    assert torch.equal(model.act(x), model.act(x))
    with pytest.raises(ValueError, match="dimension"):
        model(torch.ones(2, 5))


def test_planner_and_contract():
    planner = CEMPlanner(PhysicsOracle(), SPEC, population=32, elites=4, iterations=3)
    state = np.array([-.5, -.5, 1.15, 1.15])
    action, info = planner.plan(state)
    assert action.shape == (2,) and info["model_evidence"] == "physics_oracle"
    assert np.isfinite(info["predicted_cost"])
    with pytest.raises(ValueError, match="protocol"):
        CEMPlanner(PhysicsOracle(), ActionSpec("other", 2, "rad", "joint", 10, (-1,-1), (1,1)))


def test_iql_update_and_safety():
    agent = IQL(4, 2)
    batch = (torch.randn(8, 4), torch.rand(8, 2), torch.rand(8, 1),
             torch.randn(8, 4), torch.zeros(8, 1))
    metrics = agent.update(batch)
    assert all(np.isfinite(v) for v in metrics.values())
    with pytest.raises(PermissionError):
        RobotGate().assert_motion_allowed()


def test_expert_rollout():
    from worldpolicy.simulation import expert_action
    assert rollout(expert_action, 0)["success"]
