import numpy as np

from worldpolicy.fetch_push import SPEC, FetchPush
from worldpolicy.fetch_push_planning import FetchPhysicsOracle, push_terminal_cost
from worldpolicy.planning import CEMPlanner


def test_oracle_prediction_restores_full_state():
    env = FetchPush(55)
    state = env.reset()
    before = env.snapshot()
    model = FetchPhysicsOracle(env, horizon=2)
    future, uncertainty = model.predict(state, np.zeros((3, 2, 4)))
    after = env.snapshot()
    assert future.shape == (3, 2, 28)
    assert uncertainty.shape == (3, 2)
    np.testing.assert_allclose(before.qpos, after.qpos)
    np.testing.assert_allclose(before.mocap_pos, after.mocap_pos)
    np.testing.assert_allclose(before.goal, after.goal)
    assert before.elapsed_steps == after.elapsed_steps
    np.testing.assert_allclose(push_terminal_cost(future[:, -1]), push_terminal_cost(future[:, -1]))
    action, _ = CEMPlanner(model, SPEC, horizon=2, population=4, elites=2,
                           iterations=1, terminal_cost=push_terminal_cost).plan(state)
    assert action.shape == (4,)
    env.close()
