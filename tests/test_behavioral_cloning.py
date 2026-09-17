"""Behavioural cloning reproduces the greedy policy's action.

``collect_greedy_demonstrations`` translates a greedy well order into action
indices via ``PlanEnv.get_candidate_well_names``; ``pretrain_behavioral_cloning``
then trains the Q-network with supervision to reproduce those labels. This
file checks both halves: the labels are in fact the greedy choices, and
training on them drives prediction accuracy on those same labels to (near)
100% - the point of the whole mechanism.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from hyperactive.env import PlanEnv
from hyperactive.greedy import PlanBuilder
from hyperactive.models import BehavioralCloningConfig, MaskedCDQN
from hyperactive.models.behavioral_cloning import collect_greedy_demonstrations
from hyperactive.planning import ClusterRandomRiskStrategy, TeamManager

START = datetime(2025, 1, 1)
END = START + timedelta(days=3650)
N_ACTIONS = 8


def greedy_order(wells, npv, movement, team_pool, profile):
    plan = PlanBuilder(start=START, end=END, cost_function=npv,
                       production_profile=profile).compile(
        wells=list(wells), manager=TeamManager(team_pool=team_pool, movement=movement),
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
    )
    return [c.well.name for c in plan.well_plans]


@pytest.fixture
def raw_env(small_pool, npv, movement, make_team_pool, linear_profile):
    return PlanEnv(
        wells=list(small_pool), team_pool=make_team_pool(2, 2), movement=movement,
        cost_function=npv, n_actions=N_ACTIONS, start=START, end=END,
        production_profile=linear_profile,
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
    )


@pytest.fixture
def vec_env(raw_env):
    return VecNormalize(DummyVecEnv([lambda: raw_env]), training=True, norm_obs=True,
                        norm_reward=True)


@pytest.fixture
def order(small_pool, npv, movement, make_team_pool, linear_profile):
    return greedy_order(small_pool, npv, movement, make_team_pool(2, 2), linear_profile)


class TestDemonstrationLabels:
    def test_noise_free_demonstration_replays_the_greedy_order_exactly(self, vec_env, order):
        """noise_prob=0.0: the executed action must equal the label at every
        step, and the sequence of executed wells must equal the greedy order."""
        transitions, stats = collect_greedy_demonstrations(
            env=vec_env, greedy_order=order, n_actions=N_ACTIONS, episodes=1,
            max_steps=N_ACTIONS * 2, noise_prob=0.0,
            rng=np.random.default_rng(0),
        )
        assert stats["fallback_actions"] == 0
        assert [t["action"] for t in transitions] == [t["expert_action"] for t in transitions]
        assert len(transitions) == len(order)

    def test_noisy_demonstrations_keep_the_greedy_label_even_when_a_different_action_executes(
        self, vec_env, order
    ):
        transitions, stats = collect_greedy_demonstrations(
            env=vec_env, greedy_order=order, n_actions=N_ACTIONS, episodes=3,
            max_steps=N_ACTIONS * 2, noise_prob=0.9,
            rng=np.random.default_rng(0),
        )
        assert stats["noise_actions"] > 0, "fixture must actually exercise noisy steps"
        for t in transitions:
            if t["noisy_action"]:
                assert t["action"] != t["expert_action"]
            else:
                assert t["action"] == t["expert_action"]


class TestBehavioralCloningTraining:
    def test_pretraining_drives_greedy_action_accuracy_to_one(self, vec_env, order):
        """Train BC alone (no Q-learning at all) on noise-free demonstrations
        and check the Q-network ends up ranking the greedy action first on
        (effectively) every demonstrated state."""
        agent = MaskedCDQN("MlpPolicy", vec_env, policy_kwargs={"net_arch": [32]},
                           buffer_size=200, learning_starts=1000, seed=0, device="cpu")
        agent._setup_learn(total_timesteps=1)

        config = BehavioralCloningConfig(
            enabled=True, demo_episodes=20, demo_noise_prob=0.0,
            epochs=200, val_fraction=0.0, early_stopping_patience=0,
            target_accuracy=0.999, restore_best_weights=True,
        )
        transitions, _ = collect_greedy_demonstrations(
            env=vec_env, greedy_order=order, n_actions=N_ACTIONS, episodes=config.demo_episodes,
            max_steps=N_ACTIONS * 2, noise_prob=0.0, rng=np.random.default_rng(1),
        )
        from hyperactive.models.behavioral_cloning import pretrain_behavioral_cloning

        stats = pretrain_behavioral_cloning(
            model=agent, transitions=transitions, cfg=config,
            normalize_obs=vec_env.normalize_obs, rng=np.random.default_rng(2),
        )
        assert stats["trained"] is True
        assert stats["final_train_accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_hot_start_lowers_exploration_after_a_successful_clone(self, vec_env, order):
        """hot_start applies post_bc_exploration_initial_eps only when cloning
        actually trained; with the default config it must end up lower than
        the pre-clone default of 1.0."""
        agent = MaskedCDQN("MlpPolicy", vec_env, policy_kwargs={"net_arch": [32]},
                           buffer_size=500, learning_starts=8, batch_size=16,
                           exploration_initial_eps=1.0, seed=0, device="cpu")
        agent._setup_learn(total_timesteps=1)
        config = BehavioralCloningConfig(enabled=True, demo_episodes=20, demo_noise_prob=0.1,
                                         epochs=100, target_accuracy=0.99)
        agent.hot_start(greedy_order=order, bc_config=config, prefill_episodes=0, seed=3)
        assert agent.exploration_initial_eps < 1.0
        assert agent.exploration_rate == pytest.approx(agent.exploration_initial_eps)

    def test_disabled_cloning_is_a_no_op(self, vec_env, order):
        agent = MaskedCDQN("MlpPolicy", vec_env, policy_kwargs={"net_arch": [16]},
                           buffer_size=200, learning_starts=8, seed=0, device="cpu")
        agent._setup_learn(total_timesteps=1)
        config = BehavioralCloningConfig(enabled=False)
        result = agent.hot_start(greedy_order=order, bc_config=config, prefill_episodes=0, seed=0)
        assert result["behavioral_cloning"]["trained"] is False
        assert result["behavioral_cloning"]["skip_reason"] == "disabled"
        assert agent.exploration_initial_eps == 1.0  # unchanged
