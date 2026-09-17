"""Smoke tests for loading and running the released policy.

Uses the actual ``models/arrive39`` artifact shipped in the repository, on
the bundled synthetic well pool - these are integration tests (they load real
weights and run a real rollout), not unit tests, and are the closest thing to
"does the release actually work" in this suite.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

from hyperactive.data import load_coordinates, load_wells
from hyperactive.env import PlanEnv
from hyperactive.inference import ModelManifestError, load_policy, plan_with_policy, run_episode
from hyperactive.planning import ClusterRandomRiskStrategy, DistanceTeamMovement
from hyperactive.scenario import default_npv, default_profile, horizon_end, make_team_pool

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "arrive39"


@pytest.fixture(scope="module")
def synthetic_subset():
    wells, rejected = load_wells(ROOT / "data" / "synthetic" / "wells.csv")
    assert rejected == 0
    coordinates = load_coordinates(ROOT / "data" / "synthetic" / "clusters.csv", wells)
    movement = DistanceTeamMovement.from_dicts(coordinates)
    return wells[:16], movement


@pytest.fixture(scope="module")
def bundle():
    return load_policy(MODEL_DIR, device="cpu")


def build_env(wells, movement, bundle, start=None):
    start = start or datetime(2026, 1, 1)
    return PlanEnv(
        wells=list(wells), team_pool=make_team_pool(2, 2), movement=movement,
        cost_function=default_npv(start), n_actions=bundle.n_actions,
        start=start, end=horizon_end(start, 10),
        production_profile=default_profile(),
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
    )


class TestLoadPolicy:
    def test_manifest_fields(self, bundle):
        assert bundle.manifest["feature_set"] == "gen39"
        assert bundle.n_actions == bundle.manifest["action_window"] == 8

    def test_tampered_model_file_is_rejected(self, tmp_path):
        tampered = tmp_path / "arrive39"
        shutil.copytree(MODEL_DIR, tampered)
        with open(tampered / "model.zip", "ab") as handle:
            handle.write(b"corruption")
        with pytest.raises(ModelManifestError):
            load_policy(tampered, device="cpu")

    def test_tampered_manifest_sha_is_rejected(self, tmp_path):
        tampered = tmp_path / "arrive39"
        shutil.copytree(MODEL_DIR, tampered)
        manifest = json.loads((tampered / "manifest.json").read_text())
        manifest["files"]["vec_normalize"]["sha256"] = "0" * 64
        (tampered / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ModelManifestError):
            load_policy(tampered, device="cpu")

    def test_mismatched_feature_set_is_rejected(self, tmp_path):
        tampered = tmp_path / "arrive39"
        shutil.copytree(MODEL_DIR, tampered)
        manifest = json.loads((tampered / "manifest.json").read_text())
        manifest["feature_set"] = "not_a_real_feature_set"
        (tampered / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ModelManifestError):
            load_policy(tampered, device="cpu")


class TestRunEpisode:
    def test_deterministic_episode_produces_a_finite_plan(self, bundle, synthetic_subset):
        wells, movement = synthetic_subset
        env = build_env(wells, movement, bundle)
        plan, info = run_episode(env, bundle, exploration=0.0, seed=0)
        assert 0 < len(plan.well_plans) <= len(wells)
        assert set(c.well.name for c in plan.well_plans) <= {w.name for w in wells}
        assert info["final_npv"] == pytest.approx(plan.total_profit())

    def test_deterministic_episode_is_reproducible(self, bundle, synthetic_subset):
        wells, movement = synthetic_subset
        plan_a, _ = run_episode(build_env(wells, movement, bundle), bundle, seed=0)
        plan_b, _ = run_episode(build_env(wells, movement, bundle), bundle, seed=0)
        assert [c.well.name for c in plan_a.well_plans] == [c.well.name for c in plan_b.well_plans]
        assert plan_a.total_profit() == pytest.approx(plan_b.total_profit())

    def test_every_placed_well_is_a_real_well_from_the_pool(self, bundle, synthetic_subset):
        """A regression guard at the integration level for the same property
        test_masking.py checks at the unit level: the trained policy's own
        rollout never reads a padded (all-zero) observation row as if it were
        a well - every placed well must be a real, named member of the pool."""
        wells, movement = synthetic_subset
        env = build_env(wells, movement, bundle)
        plan, _ = run_episode(env, bundle, exploration=0.0, seed=0)
        pool_names = {w.name for w in wells}
        assert plan.well_plans
        assert all(c.well.name in pool_names for c in plan.well_plans)


class TestPlanWithPolicy:
    def test_returned_plan_is_the_best_among_the_episodes_actually_run(self, bundle,
                                                                        synthetic_subset):
        """``plan_with_policy`` picks the highest-NPV plan among the episodes
        it ran - not "better than some other configuration": with heavy
        exploration a noisy rollout can easily underperform a single
        deterministic one, so that is not a property to assert. What must
        always hold is that the returned NPV equals max(stats["npvs"])."""
        wells, movement = synthetic_subset
        plan, stats = plan_with_policy(
            build_env(wells, movement, bundle), bundle, episodes=5, exploration=0.3)
        assert stats["episodes"] == 5
        assert len(stats["npvs"]) == 5
        assert plan.total_profit() == pytest.approx(max(stats["npvs"]))
        assert 0 <= stats["selected_episode"] < 5

    def test_single_deterministic_episode_matches_run_episode_directly(self, bundle,
                                                                        synthetic_subset):
        wells, movement = synthetic_subset
        via_plan_with_policy, _ = plan_with_policy(
            build_env(wells, movement, bundle), bundle, episodes=1, exploration=0.0)
        via_run_episode, _ = run_episode(build_env(wells, movement, bundle), bundle,
                                         exploration=0.0, seed=0)
        assert via_plan_with_policy.total_profit() == pytest.approx(via_run_episode.total_profit())

    def test_episodes_must_match_the_policys_action_window(self, bundle, synthetic_subset):
        wells, movement = synthetic_subset
        start = datetime(2026, 1, 1)
        wrong_window_env = PlanEnv(
            wells=list(wells), team_pool=make_team_pool(2, 2), movement=movement,
            cost_function=default_npv(start), n_actions=bundle.n_actions + 1,
            start=start, end=horizon_end(start, 10), production_profile=default_profile(),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        with pytest.raises(ValueError):
            run_episode(wrong_window_env, bundle)
