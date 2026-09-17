"""Numeric verification of the C-DQN loss formula on a hand-computed batch.

``hyperactive.models.masked_cdqn.MaskedCDQN.train`` implements (all actions
valid, so masking plays no role here - see ``test_masking.py`` for that):

    a*     = argmax_a Q(s', a)                      (no gradient, Double DQN)
    y_DQN  = r + gamma * (1 - done) * Q'(s', a*)
    y_MSBE = r + gamma * (1 - done) * max_a Q(s', a)
    L      = mean(max(l_Huber(Q(s, a), y_DQN), l_Huber(Q(s, a), y_MSBE)))

Every value below is computed independently in plain Python/PyTorch from a
fixed, fully-controlled Q-network (online and target networks patched to
known constant outputs) and checked against what ``train()`` actually
produces - not against another copy of the same formula.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hyperactive.env import PlanEnv
from hyperactive.models import MaskedCDQN
from hyperactive.planning import ClusterRandomRiskStrategy

START = datetime(2025, 1, 1)
END = START + timedelta(days=3650)
N_ACTIONS = 8
GAMMA = 0.99


@pytest.fixture
def vec_env(small_pool, npv, movement, make_team_pool, linear_profile):
    raw_env = PlanEnv(
        wells=list(small_pool), team_pool=make_team_pool(2, 2), movement=movement,
        cost_function=npv, n_actions=N_ACTIONS, start=START, end=END,
        production_profile=linear_profile,
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
    )
    return VecNormalize(DummyVecEnv([lambda: raw_env]), training=True, norm_obs=True,
                        norm_reward=True)


def make_agent(vec_env, use_cdqn: bool) -> MaskedCDQN:
    agent = MaskedCDQN("MlpPolicy", vec_env, policy_kwargs={"net_arch": [16]},
                       buffer_size=200, learning_starts=8, batch_size=8,
                       gamma=GAMMA, use_cdqn=use_cdqn, seed=0, device="cpu")
    agent._setup_learn(total_timesteps=1)
    return agent


def fill_toy_batch(agent, rewards, dones, current_q_row, next_q_online_row, next_q_target_row):
    """One transition per element of ``rewards``; every action is valid.

    All samples share the same (patched) Q-network output rows so the expected
    loss can be computed by hand once and compared against every sample.
    """
    batch = len(rewards)
    obs_dim = agent.observation_space.shape
    obs = np.zeros((1, *obs_dim), dtype=np.float32)
    mask = np.ones(N_ACTIONS, dtype=bool)

    for i in range(batch):
        agent.replay_buffer.add(
            obs=obs, next_obs=obs, action=np.array([[0]]),
            reward=np.array([rewards[i]], dtype=np.float32),
            done=np.array([dones[i]], dtype=np.float32),
            infos=[{"action_mask": mask, "next_action_mask": mask}],
        )

    # self.q_net (the online network) is called twice per train() step, in a
    # fixed order: once for current_q (over `observations`), once for
    # next_q_online (over `next_observations`). Dispatch on call order so each
    # gets its own, independently chosen row.
    calls = {"n": 0}

    def q_net_dispatch(o, action_mask=None):
        calls["n"] += 1
        row = current_q_row if calls["n"] % 2 == 1 else next_q_online_row
        return row.unsqueeze(0).expand(o.shape[0], -1).clone().requires_grad_(True)

    agent.q_net.forward = q_net_dispatch
    agent.q_net_target.forward = (
        lambda o, action_mask=None: next_q_target_row.unsqueeze(0).expand(o.shape[0], -1).clone()
    )
    return batch


class TestCDQNLossFormula:
    def test_plain_dqn_loss_matches_hand_computed_huber(self, vec_env):
        """use_cdqn=False: L must equal mean Huber(current_q, y_DQN) exactly,
        with no contribution from y_MSBE at all."""
        agent = make_agent(vec_env, use_cdqn=False)
        current_q_row = th.tensor([2.0] * N_ACTIONS)
        next_q_online_row = th.tensor([1.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # argmax -> index 1
        next_q_target_row = th.tensor([9.0, 7.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Q'(s', 1) = 7.0

        reward, done = 1.0, 0.0
        batch = fill_toy_batch(agent, [reward], [done], current_q_row, next_q_online_row,
                               next_q_target_row)
        agent.train(gradient_steps=1, batch_size=batch)

        expected_target = reward + GAMMA * (1 - done) * 7.0  # Q'(s', argmax_a Q(s',a)=1)
        expected_loss = F.smooth_l1_loss(th.tensor(2.0), th.tensor(expected_target)).item()
        assert agent.last_update_info["target_mean"] == pytest.approx(expected_target, abs=1e-4)
        assert agent.last_update_info["loss"] == pytest.approx(expected_loss, abs=1e-4)
        assert np.isnan(agent.last_update_info["loss_msbe"])

    def test_cdqn_loss_is_the_elementwise_max_of_both_branches(self, vec_env):
        """``max_a Q(s', a)`` (for y_MSBE) and ``argmax_a Q(s', a)`` (for
        Double-DQN's action selection) are computed from the *same* tensor
        (next_q_online), so they always pick the same action index; y_DQN and
        y_MSBE differ only in which network's value is read at that index -
        the target network's for y_DQN, the online network's own for y_MSBE.

        Here the greedy action is index 1: next_q_online puts its max there
        (9.0), next_q_target puts a small value there (0.5) - so y_MSBE lands
        far from current_q and y_DQN lands close to it, and L must equal the
        (larger) MSBE-branch Huber loss, matching L = mean(max(l_DQN, l_MSBE))
        literally, not just "close to it".
        """
        agent = make_agent(vec_env, use_cdqn=True)
        current_q_row = th.tensor([2.0] * N_ACTIONS)
        next_q_online_row = th.tensor([1.0, 9.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # argmax = index 1
        next_q_target_row = th.tensor([9.0, 0.5, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Q'(s', 1) = 0.5

        reward, done = 1.0, 0.0
        batch = fill_toy_batch(agent, [reward], [done], current_q_row, next_q_online_row,
                               next_q_target_row)
        agent.train(gradient_steps=1, batch_size=batch)

        y_dqn = reward + GAMMA * (1 - done) * 0.5   # Q'(s', argmax) - target network
        y_msbe = reward + GAMMA * (1 - done) * 9.0  # Q(s', argmax) - online network itself
        loss_dqn = F.smooth_l1_loss(th.tensor(2.0), th.tensor(y_dqn)).item()
        loss_msbe = F.smooth_l1_loss(th.tensor(2.0), th.tensor(y_msbe)).item()
        assert loss_msbe > loss_dqn, "fixture must make the MSBE branch the larger one"

        assert agent.last_update_info["loss_dqn"] == pytest.approx(loss_dqn, abs=1e-4)
        assert agent.last_update_info["loss_msbe"] == pytest.approx(loss_msbe, abs=1e-4)
        assert agent.last_update_info["loss"] == pytest.approx(max(loss_dqn, loss_msbe), abs=1e-4)
        assert agent.last_update_info["msbe_active_fraction"] == pytest.approx(1.0)

    def test_msbe_active_fraction_is_the_share_where_msbe_dominates(self, vec_env):
        """A batch of two: one sample where DQN's target is larger (MSBE
        inactive), one where MSBE's is (MSBE active) -> fraction must be 0.5."""
        agent = make_agent(vec_env, use_cdqn=True)
        current_q_row = th.tensor([2.0] * N_ACTIONS)
        next_q_online_row = th.tensor([1.0, 8.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # max=8, argmax=1
        next_q_target_row = th.tensor([9.0, 0.5, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Q'(s',1)=0.5

        # y_DQN = r + gamma*0.5 (small); y_MSBE = r + gamma*8 (large) -> MSBE active here.
        batch = fill_toy_batch(agent, [1.0], [0.0], current_q_row, next_q_online_row,
                               next_q_target_row)
        agent.train(gradient_steps=1, batch_size=batch)
        assert agent.last_update_info["msbe_active_fraction"] == pytest.approx(1.0)

    def test_terminal_transition_bootstraps_from_zero_not_from_the_target_network(self, vec_env):
        """done=1: both y_DQN and y_MSBE must equal the reward alone, with no
        gamma * Q(...) term - the value-after-termination is by definition 0."""
        agent = make_agent(vec_env, use_cdqn=False)
        current_q_row = th.tensor([2.0] * N_ACTIONS)
        next_q_online_row = th.tensor([100.0] * N_ACTIONS)  # would dominate if not zeroed out
        next_q_target_row = th.tensor([100.0] * N_ACTIONS)

        reward = 3.0
        batch = fill_toy_batch(agent, [reward], [1.0], current_q_row, next_q_online_row,
                               next_q_target_row)
        agent.train(gradient_steps=1, batch_size=batch)
        assert agent.last_update_info["target_mean"] == pytest.approx(reward, abs=1e-4)

    def test_use_cdqn_false_and_true_agree_when_target_matches_online_at_the_greedy_action(
        self, vec_env
    ):
        """y_DQN and y_MSBE both bootstrap from the value at the same greedy
        action (argmax of next_q_online); they coincide exactly when the
        target network agrees with the online network's own value there - not
        when "argmax" and "max" differ, since they never do (same tensor).
        With that value equal (6.0 at index 1 in both rows), use_cdqn must not
        change the loss at all."""
        current_q_row = th.tensor([2.0] * N_ACTIONS)
        next_q_online_row = th.tensor([1.0, 6.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # argmax = index 1
        next_q_target_row = th.tensor([0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # same value there

        losses = {}
        for use_cdqn in (False, True):
            agent = make_agent(vec_env, use_cdqn=use_cdqn)
            batch = fill_toy_batch(agent, [1.0], [0.0], current_q_row, next_q_online_row,
                                   next_q_target_row)
            agent.train(gradient_steps=1, batch_size=batch)
            losses[use_cdqn] = agent.last_update_info["loss"]
        assert losses[False] == pytest.approx(losses[True], abs=1e-4)

    def test_loss_and_grad_norm_are_finite(self, vec_env):
        """Sanity floor under every test above: nothing here should ever
        produce NaN/Inf - train() itself raises RuntimeError if it does."""
        agent = make_agent(vec_env, use_cdqn=True)
        current_q_row = th.tensor([2.0] * N_ACTIONS)
        next_q_online_row = th.tensor([1.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        next_q_target_row = th.tensor([9.0, 7.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        batch = fill_toy_batch(agent, [1.0], [0.0], current_q_row, next_q_online_row,
                               next_q_target_row)
        agent.train(gradient_steps=1, batch_size=batch)
        assert np.isfinite(agent.last_update_info["loss"])
        assert np.isfinite(agent.last_update_info["grad_norm"])
