"""Project settings: one economics for every entry point, and no unpriced wells."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from hyperactive.core import WellPlanContext
from hyperactive.data.synthetic import WELL_TYPES
from hyperactive.planning import TeamManager
from hyperactive.scenario import ECONOMICS, horizon_end, load_settings, make_npv, planning_start


def test_defaults_are_the_project_economics():
    settings = load_settings()
    assert settings.economics == ECONOMICS
    assert settings.economics is not ECONOMICS, "overrides must not leak into the defaults"


def test_overrides_merge_by_key_and_replace_the_cost_table_whole(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "economics": {"water_cost_per_tone": 1.0, "build_cost_per_meter": {"ГС": 1.0}},
        "readiness_hour": 8, "days_per_year": 365,
    }), encoding="utf-8")
    settings = load_settings(path)
    assert settings.economics["water_cost_per_tone"] == 1.0
    assert settings.economics["oil_price_per_tone"] == ECONOMICS["oil_price_per_tone"]
    # Replaced, not merged: a legacy table must be able to leave a type unpriced.
    assert settings.economics["build_cost_per_meter"] == {"ГС": 1.0}
    assert (settings.readiness_hour, settings.days_per_year) == (8, 365.0)
    assert ECONOMICS["water_cost_per_tone"] != 1.0


def test_unknown_settings_are_rejected_rather_than_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"economic": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="economic"):
        load_settings(path)


def test_every_well_type_the_generator_emits_is_priced():
    assert set(WELL_TYPES) <= set(ECONOMICS["build_cost_per_meter"])


def test_an_unpriced_well_type_fails_instead_of_drilling_for_free(make_well, make_team_pool,
                                                                  movement, linear_profile):
    start = datetime(2025, 1, 1)
    context = WellPlanContext(well=make_well("w1"), start=start, end=start + timedelta(days=3650))
    TeamManager(team_pool=make_team_pool(1, 1), movement=movement).get_assignments(context)
    linear_profile.compute(context)
    economics = {**ECONOMICS, "build_cost_per_meter": {"DRILLING": 15_000.0}}
    with pytest.raises(KeyError, match="DRILLING\\+GTM"):
        make_npv(start, economics).compute(context)


def test_planning_starts_at_midnight_of_the_earliest_readiness(make_well):
    wells = [make_well("w1", ready_after_days=40), make_well("w2", ready_after_days=10)]
    wells[1].readiness_date = wells[1].readiness_date.replace(hour=8)
    assert planning_start(wells) == datetime(2025, 1, 11)
    assert planning_start([make_well("w3")], default=datetime(2030, 1, 1)) == datetime(2030, 1, 1)


def test_the_economic_horizon_follows_the_benchmark_calendar():
    start = datetime(2026, 7, 9)
    assert horizon_end(start, 20) == start + timedelta(days=round(240 * 365.25 / 12))
