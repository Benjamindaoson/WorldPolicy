import numpy as np
import pytest

from worldpolicy.fetch_push import SPEC
from worldpolicy.fetch_push_dynamics import load_dynamics, train_dynamics
from worldpolicy.fetch_push_planning import push_terminal_cost
from worldpolicy.planning import CEMPlanner


def test_robot_dynamics_training_contract(tmp_path):
    record = train_dynamics(tmp_path, episodes=4, seed=12345, updates=2)
    model = load_dynamics(tmp_path / "checkpoint.pt", record["checkpoint_sha256"])
    state = np.zeros(28, dtype=np.float32)
    future, uncertainty = model.predict(state, np.zeros((2, 3, 4), dtype=np.float32))
    assert future.shape == (2, 3, 28) and uncertainty.shape == (2, 3)
    assert model.metadata.evidence == "learned_dynamics"
    assert CEMPlanner(model, SPEC, horizon=3, population=4, elites=2,
                      terminal_cost=push_terminal_cost).plan(state)[0].shape == (4,)
    with pytest.raises(ValueError, match="protocol"):
        model.predict(state, np.zeros((2, 3, 2)))
