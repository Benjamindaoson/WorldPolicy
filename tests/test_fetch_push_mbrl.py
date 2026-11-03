import torch

from worldpolicy.fetch_push_dynamics import RobotDynamics
from worldpolicy.fetch_push_mbrl import Actor, Critic, RewardTermination, bootstrap_target, imagine


def test_mbrl_actor_critic_reward_gradients_and_imagination_filter():
    torch.manual_seed(4)
    actor, critic, reward_term = Actor(), Critic(), RewardTermination()
    state = torch.randn(8, 28)
    action = actor(state)
    value = critic(state, action)
    reward, termination_logit = reward_term(state, action)
    loss = value.square().mean() + reward.square().mean() + termination_logit.square().mean()
    loss.backward()
    assert actor.net[0].weight.grad is not None
    assert critic.net[0].weight.grad is not None
    assert reward_term.net[0].weight.grad is not None
    ensemble = [RobotDynamics(), RobotDynamics()]
    kept, rejected = imagine(ensemble, actor, state, cutoff=float("inf"), horizon=2)
    assert len(kept["state"]) == 16 and rejected == 0
    assert kept["action"].shape == (16, 4)


def test_time_limit_is_not_termination_target():
    reward = torch.tensor([[-.2], [-.2]])
    terminated = torch.tensor([[1.], [0.]])
    truncated = torch.tensor([[0.], [1.]])
    next_q = torch.ones_like(reward)
    target = bootstrap_target(reward, terminated, next_q)
    assert target[0, 0] == reward[0, 0]
    assert target[1, 0] > reward[1, 0] and truncated[1, 0] == 1
