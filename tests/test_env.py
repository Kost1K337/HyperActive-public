"""``PlanEnv`` invariants: observation shape, action mask, termination, reward.

Uses a scripted "always take the first valid action" policy throughout - the
mask contract (never violated, never empty until termination) has to hold
regardless of which valid action is chosen.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from hyperactive.env import EMBEDDING_WIDTH, PlanEnv
from hyperactive.env.plan_env import INVALID_ACTION_REWARD
from hyperactive.planning import ClusterRandomRiskStrategy

START = datetime(2025, 1, 1)
END = START + timedelta(days=3650)
N_ACTIONS = 8


@pytest.fixture
def env(small_pool, npv, movement, make_team_pool, linear_profile):
    return PlanEnv(
        wells=list(small_pool), team_pool=make_team_pool(2, 2), movement=movement,
        cost_function=npv, n_actions=N_ACTIONS, start=START, end=END,
        production_profile=linear_profile,
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
    )


def run_to_completion(env, max_steps=20):
    """Scripted rollout: always the first valid action. Returns the step log."""
    obs, info = env.reset()
    log = [{"obs": obs, "mask": info["action_mask"]}]
    for _ in range(max_steps):
        mask = log[-1]["mask"]
        assert mask.any(), "must terminate before running out of valid actions"
        action = int(np.flatnonzero(mask)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        log.append({"obs": obs, "mask": info["action_mask"], "reward": reward,
                   "terminated": terminated, "truncated": truncated, "info": info})
        if terminated or truncated:
            return log
    raise AssertionError(f"did not terminate within {max_steps} steps")


class TestObservationSpace:
    def test_shape_and_dtype(self, env):
        obs, _ = env.reset()
        assert obs.shape == (N_ACTIONS, EMBEDDING_WIDTH)
        assert obs.dtype == np.float32

    def test_matches_declared_spaces(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        assert env.action_space.n == N_ACTIONS


class TestActionMask:
    def test_reset_mask_matches_pool_size_when_smaller_than_window(self, env, small_pool):
        _, info = env.reset()
        mask = info["action_mask"]
        assert mask.dtype == bool
        assert mask.shape == (N_ACTIONS,)
        # Four wells, all ready immediately, into an 8-wide window: exactly the
        # first four actions are valid, not more, not fewer.
        assert mask.sum() == len(small_pool)
        assert mask[: len(small_pool)].all()
        assert not mask[len(small_pool):].any()

    def test_padded_rows_are_exactly_zero(self, env, small_pool):
        obs, info = env.reset()
        mask = info["action_mask"]
        for action in range(N_ACTIONS):
            if not mask[action]:
                assert np.array_equal(obs[action], np.zeros(EMBEDDING_WIDTH, dtype=np.float32))

    def test_mask_shrinks_by_one_after_each_valid_step(self, env):
        log = run_to_completion(env)
        valid_counts = [step["mask"].sum() for step in log[:-1]]
        assert valid_counts == sorted(valid_counts, reverse=True)
        assert all(a - b == 1 for a, b in zip(valid_counts, valid_counts[1:]))

    def test_mask_is_all_false_once_terminated(self, env):
        log = run_to_completion(env)
        assert log[-1]["terminated"] is True
        assert not log[-1]["mask"].any()

    def test_taking_a_padded_action_terminates_with_the_invalid_reward(self, env, small_pool):
        _, info = env.reset()
        mask = info["action_mask"]
        padded_action = int(np.flatnonzero(~mask)[0])
        _, reward, terminated, truncated, info = env.step(padded_action)
        assert reward == INVALID_ACTION_REWARD
        assert terminated is True
        assert truncated is False
        assert info["final_npv"] == 0.0  # nothing was ever placed


class TestEpisodeReward:
    def test_reward_sum_equals_final_plan_npv(self, env):
        """The reward is defined as the NPV increment; summed over an episode
        that never hits the invalid-action penalty, it must equal the total."""
        log = run_to_completion(env)
        total_reward = sum(step["reward"] for step in log[1:])
        assert total_reward == pytest.approx(log[-1]["info"]["final_npv"])

    def test_final_plan_contains_every_well(self, env, small_pool):
        log = run_to_completion(env)
        plan = log[-1]["info"]["final_plan"]
        assert {c.well.name for c in plan.well_plans} == {w.name for w in small_pool}


class TestReset:
    def test_reset_is_independent_of_prior_episode_state(self, env):
        run_to_completion(env)  # leave the env in a terminated state
        obs, info = env.reset()
        assert info["action_mask"].any()
        assert env.get_action_mask().sum() == info["action_mask"].sum()

    def test_two_resets_without_stochasticity_give_the_same_first_observation(self, env):
        first, _ = env.reset()
        second, _ = env.reset()
        # trigger_chance=0.0 risk strategy and no exploration in the env itself:
        # reset must be deterministic given the same wells and crews.
        np.testing.assert_array_equal(first, second)
