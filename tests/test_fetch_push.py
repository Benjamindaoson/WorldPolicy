import numpy as np
import pytest

from worldpolicy.fetch_push import SPEC, FetchPush, PushExpert, collect_expert, rollout


def test_fetch_push_reset_restore_and_action_contract():
    env = FetchPush(123)
    initial = env.reset()
    snapshot = env.snapshot()
    env.step(np.zeros(4))
    np.testing.assert_allclose(env.restore(snapshot), initial, atol=1e-5)
    np.testing.assert_allclose(env.env.unwrapped.data.mocap_pos, snapshot.mocap_pos)
    with pytest.raises(ValueError, match="finite 4D"):
        env.step(np.zeros(2))
    assert SPEC.dimension == 4 and SPEC.frequency_hz == 25
    env.close()


def test_fetch_push_expert_and_dataset():
    result = rollout(3, PushExpert())
    assert result["success"] and result["goal_distance"] < .05
    episode = collect_expert(1, 3)[0]
    episode.validate()
    assert episode.success and episode.states.shape[1] == 28
