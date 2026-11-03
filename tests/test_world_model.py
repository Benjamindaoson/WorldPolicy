import numpy as np
import pytest

from worldpolicy.planning import CEMPlanner
from worldpolicy.simulation import SPEC
from worldpolicy.world_model import load_dynamics, train_dynamics


def test_learned_dynamics_contract_and_checkpoint(tmp_path):
    record = train_dynamics(tmp_path, seed=3, episodes=4, steps=2)
    model = load_dynamics(tmp_path / "dynamics.pt", record["checkpoint_sha256"])
    future, uncertainty = model.predict(np.zeros(4), np.zeros((3, 2, 2)))
    assert future.shape == (3, 2, 4)
    assert uncertainty.shape == (3, 2)
    assert model.metadata.evidence == "learned_dynamics"
    assert CEMPlanner(model, SPEC, horizon=2, population=8, elites=2).plan(np.zeros(4))[0].shape == (2,)
    with pytest.raises(ValueError, match="horizon"):
        model.predict(np.zeros(4), np.zeros((3, 9, 2)))
