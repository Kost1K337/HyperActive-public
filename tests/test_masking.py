"""Action masking in ``MaskedCDQN``: an invalid action is never chosen, either
by ``predict`` (act) or inside ``train`` (bootstrap / Double-DQN target).

Uses a tiny hand-built model, not the trained ``arrive39`` weights: the point
is to exercise the masking machinery itself under adversarial masks, including
ones the real environment would never actually produce (e.g. only the last
action valid), which is exactly where a masking bug would otherwise hide.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hyperactive.env import PlanEnv
from hyperactive.models import MaskedCDQN
from hyperactive.planning import ClusterRandomRiskStrategy

START = datetime(2025, 1, 1)
END = START + timedelta(days=3650)
N_ACTIONS = 8


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


@pytest.fixture
def model(vec_env):
    agent = MaskedCDQN("MlpPolicy", vec_env, policy_kwargs={"net_arch": [16]},
                       buffer_size=200, learning_starts=8, batch_size=8,
                       seed=0, device="cpu")
    agent._setup_learn(total_timesteps=1)  # populates _last_obs / _last_action_masks
    return agent


class TestPredictRespectsTheMask:
    @pytest.mark.parametrize("valid_index", range(N_ACTIONS))
    def test_deterministic_predict_only_ever_picks_a_valid_action(self, model, vec_env,
                                                                   valid_index):
        """One valid action at a time, in every position: the argmax must never
        land outside it, regardless of what the (randomly initialised) Q-network
        happens to prefer."""
        mask = np.zeros((1, N_ACTIONS), dtype=bool)
        mask[0, valid_index] = True
        obs = vec_env.reset()
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        assert int(action[0]) == valid_index

    def test_stochastic_predict_also_never_picks_an_invalid_action(self, model, vec_env):
        mask = np.zeros((1, N_ACTIONS), dtype=bool)
        mask[0, [1, 4, 6]] = True
        obs = vec_env.reset()
        seen = set()
        rng_state = np.random.get_state()
        try:
            np.random.seed(0)
            for _ in range(200):
                action, _ = model.predict(obs, deterministic=False, action_masks=mask)
                seen.add(int(action[0]))
        finally:
            np.random.set_state(rng_state)
        assert seen <= {1, 4, 6}

    def test_predict_raises_rather_than_pick_an_action_with_no_valid_mask(self, model, vec_env):
        """The empty-mask guard lives in ``_random_valid_action``, reached from
        ``predict`` only via its exploration branch; force it with
        exploration_rate=1.0 so the check is exercised deterministically
        rather than with some probability."""
        mask = np.zeros((1, N_ACTIONS), dtype=bool)
        obs = vec_env.reset()
        model.exploration_rate = 1.0
        with pytest.raises(RuntimeError):
            model.predict(obs, deterministic=False, action_masks=mask)

    def test_random_valid_action_raises_directly_with_no_valid_mask(self, model):
        with pytest.raises(RuntimeError):
            model._random_valid_action(np.zeros(N_ACTIONS, dtype=bool))


class TestTrainRejectsInvalidStoredActions:
    def test_train_raises_if_the_replay_buffer_holds_an_action_invalid_under_its_own_mask(
        self, model
    ):
        """A corrupted buffer (action valid at collection time but not under the
        mask stored alongside it) must fail loudly, not train on it silently."""
        obs_dim = model.observation_space.shape
        obs = np.zeros((1, *obs_dim), dtype=np.float32)
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[0] = True  # only action 0 is valid ...
        model.replay_buffer.add(
            obs=obs, next_obs=obs,
            action=np.array([[3]]),  # ... but action 3 is what gets stored
            reward=np.array([1.0], dtype=np.float32),
            done=np.array([0.0], dtype=np.float32),
            infos=[{"action_mask": mask, "next_action_mask": mask}],
        )
        with pytest.raises(RuntimeError, match="invalid"):
            model.train(gradient_steps=1, batch_size=1)


class TestBootstrapExcludesInvalidActions:
    @pytest.mark.parametrize("use_cdqn", [False, True], ids=["double_dqn", "c_dqn"])
    def test_target_never_bootstraps_through_a_masked_next_action(self, vec_env, use_cdqn):
        """Replace the Q-network's output with a fixed row where the *masked*
        action 5 has a huge value (1000) and the two *allowed* actions {0, 2}
        have small ones (10 and 3). With reward=0 and gamma=0.99, a target that
        correctly ignores action 5 must equal ``0.99 * 10 = 9.9``; a target
        that leaked the masked action in would be close to ``0.99 * 1000``.
        This holds for both the Double-DQN target and the C-DQN ``l_MSBE``
        bootstrap (``use_cdqn`` parametrised), since both take the max/argmax
        over ``next_action_masks``.
        """
        model = MaskedCDQN("MlpPolicy", vec_env, policy_kwargs={"net_arch": [16]},
                           buffer_size=200, learning_starts=8, batch_size=8,
                           use_cdqn=use_cdqn, seed=0, device="cpu")
        model._setup_learn(total_timesteps=1)

        fixed_row = th.tensor([10.0, -5.0, 3.0, -5.0, -5.0, 1000.0, -5.0, -5.0])

        def fake_forward(obs, action_mask=None):
            return fixed_row.unsqueeze(0).expand(obs.shape[0], -1).clone().requires_grad_(True)

        model.q_net.forward = fake_forward
        model.q_net_target.forward = fake_forward

        obs_dim = model.observation_space.shape
        batch = 4
        obs = np.random.default_rng(0).normal(size=(batch, *obs_dim)).astype(np.float32)
        next_obs = np.random.default_rng(1).normal(size=(batch, *obs_dim)).astype(np.float32)
        mask_all = np.ones((batch, N_ACTIONS), dtype=bool)
        next_mask = np.zeros((batch, N_ACTIONS), dtype=bool)
        next_mask[:, [0, 2]] = True  # action 5 (the huge fake value) is masked out

        for i in range(batch):
            model.replay_buffer.add(
                obs=obs[i:i + 1], next_obs=next_obs[i:i + 1],
                action=np.array([[0]]),  # valid under mask_all
                reward=np.array([0.0], dtype=np.float32),
                done=np.array([0.0], dtype=np.float32),
                infos=[{"action_mask": mask_all[i], "next_action_mask": next_mask[i]}],
            )

        model.train(gradient_steps=1, batch_size=batch)

        # target_mean always reflects the Double-DQN target (computed
        # unconditionally, regardless of use_cdqn), so this alone checks that
        # branch's masking for both parametrisations.
        expected_target = 0.99 * 10.0  # gamma * Q(s', a=0), the best *allowed* action
        assert model.last_update_info["target_mean"] == pytest.approx(expected_target, abs=1e-4)

        if use_cdqn:
            # The C-DQN (l_MSBE) branch is separate: it bootstraps through
            # next_q_online.max() over the masked row, not through target_mean.
            # current_q = fixed_row[0] = 10.0, and a correctly masked MSBE
            # target is also 0.99 * 10 = 9.9, giving a tiny Huber loss. If
            # action 5's fake value (1000) leaked past the mask, the target
            # would jump to ~990 and this loss would explode by three orders
            # of magnitude - that gap is what this bound is set to catch.
            assert model.last_update_info["loss_msbe"] < 1.0
