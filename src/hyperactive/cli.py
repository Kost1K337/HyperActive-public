"""Command line: build a drilling plan with a trained policy and compare it with the greedy planner.

Example::

    hyperactive-plan --wells data/synthetic/wells.csv --coordinates data/synthetic/clusters.csv \\
        --model models/arrive39 --start 2026-01-01 --horizon-years 10 --drilling-months 24 \\
        --drilling-crews 3 --gtm-crews 2 --out plan.csv
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from hyperactive.core import Plan
from hyperactive.data import load_coordinates, load_wells
from hyperactive.env import PlanEnv
from hyperactive.greedy import PlanBuilder
from hyperactive.inference import load_policy, plan_with_policy
from hyperactive.planning import ClusterRandomRiskStrategy, DistanceTeamMovement, TeamManager
from hyperactive.scenario import (
    default_npv,
    default_profile,
    horizon_end,
    make_team_pool,
    oil_constraints,
    work_window_end,
)
from hyperactive.tracking import RunTracker, plan_metrics


def schedule_frame(plan: Plan) -> pd.DataFrame:
    """One row per task: order of the well in the plan, crew, dates, move and well NPV."""
    crew_ids: dict[object, str] = {}
    rows = []
    for position, context in enumerate(plan.well_plans, start=1):
        for entry in context.entries:
            crew = crew_ids.setdefault(entry.team, f"{entry.task.name.lower()}-{len(crew_ids) + 1}")
            rows.append({
                "order": position,
                "well": context.well.name,
                "cluster": context.well.cluster,
                "well_type": context.well.well_type,
                "task": entry.task.name,
                "crew": crew,
                "start": entry.start,
                "end": entry.end,
                "travel_days": entry.travel_time.total_seconds() / 86400.0,
                "launch_date": context.launch_date,
                "well_npv": context.cost,
            })
    return pd.DataFrame(rows)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hyperactive-plan", description=__doc__.splitlines()[0])
    p.add_argument("--wells", required=True, help="CSV/XLSX well table (see docs/data-format.md)")
    p.add_argument("--coordinates", default=None, help="CSV/XLSX cluster coordinates, metres")
    p.add_argument("--model", default="models/arrive39", help="directory with manifest.json")
    p.add_argument("--start", default="2025-01-01", help="planning start date, YYYY-MM-DD")
    p.add_argument("--horizon-years", type=int, default=10, help="economic horizon")
    p.add_argument("--drilling-months", type=int, default=None,
                   help="work window; all tasks must finish within it (default: the whole horizon)")
    p.add_argument("--drilling-crews", type=int, default=2)
    p.add_argument("--gtm-crews", type=int, default=1)
    p.add_argument("--oil-cap", type=float, default=None, help="annual oil production cap, t")
    p.add_argument("--max-commissioning-days", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1,
                   help="with --exploration > 0: number of stochastic rollouts, the best is kept")
    p.add_argument("--exploration", type=float, default=0.0)
    p.add_argument("--no-greedy", action="store_true", help="skip the greedy baseline")
    p.add_argument("--out", default=None, help="write the RL schedule to this CSV")
    p.add_argument("--experiment", default="hyperactive-plan", help="MLflow experiment name")
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI; default: $HYPERACTIVE_MLFLOW_TRACKING_URI "
                        "or the local file store file:./mlruns")
    p.add_argument("--no-tracking", action="store_true", help="do not record the run in MLflow")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    tracker = RunTracker(tracking_uri=args.tracking_uri, experiment=args.experiment,
                         enabled=not args.no_tracking)
    with tracker.run(f"{Path(args.model).name}-{Path(args.wells).stem}",
                     tags={"run_type": "plan", "model.key": Path(args.model).name}):
        return _plan(args, tracker)


def _plan(args: argparse.Namespace, tracker: RunTracker) -> int:
    wells, rejected = load_wells(args.wells)
    if not wells:
        raise SystemExit("No valid wells in the input table.")
    movement = DistanceTeamMovement.from_dicts(load_coordinates(args.coordinates, wells))
    start = datetime.fromisoformat(args.start)
    end = horizon_end(start, args.horizon_years)
    end_jobs = work_window_end(start, end, args.drilling_months)

    tracker.log_environment()
    tracker.log_params({key: value for key, value in vars(args).items()
                        if key not in {"tracking_uri", "experiment", "no_tracking"}})
    tracker.log_dataset(args.wells, "wells")
    if args.coordinates:
        tracker.log_dataset(args.coordinates, "coordinates")
    tracker.log_model_artifacts(args.model)

    bundle = load_policy(args.model)
    env = PlanEnv(
        wells=deepcopy(wells),
        team_pool=make_team_pool(args.drilling_crews, args.gtm_crews),
        movement=movement,
        cost_function=default_npv(start),
        n_actions=bundle.n_actions,
        start=start,
        end=end,
        end_jobs=end_jobs,
        production_profile=default_profile(),
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        constraints=oil_constraints(args.oil_cap),
        max_commissioning_days=args.max_commissioning_days,
    )
    plan, stats = plan_with_policy(env, bundle, episodes=args.episodes, exploration=args.exploration)
    rl_npv = plan.total_profit()
    print(f"wells: {len(wells)} valid, {rejected} rejected")
    print(f"RL plan ({bundle.manifest.get('name', bundle.directory.name)}): "
          f"{len(plan.well_plans)} wells, NPV {rl_npv:,.0f}")
    tracker.log_metrics({"rejected_wells": rejected, "wells_total": len(wells),
                         "rl_npv": rl_npv, "rl_wells": len(plan.well_plans),
                         "rl_selected_episode": stats["selected_episode"]})
    window = (start, end_jobs)
    tracker.log_metrics({f"rl.{key}": value
                         for key, value in plan_metrics(plan, window, args.drilling_crews,
                                                        args.gtm_crews).items()})

    if not args.no_greedy:
        greedy_plan = PlanBuilder(
            start=start, end=end, end_jobs=end_jobs,
            cost_function=default_npv(start), production_profile=default_profile(),
            constraints=oil_constraints(args.oil_cap),
            max_commissioning_days=args.max_commissioning_days,
        ).compile(
            wells=deepcopy(wells),
            manager=TeamManager(team_pool=make_team_pool(args.drilling_crews, args.gtm_crews),
                                movement=movement),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        greedy_npv = greedy_plan.total_profit()
        uplift = (rl_npv / greedy_npv - 1.0) * 100.0 if greedy_npv > 0 else float("nan")
        print(f"greedy plan: {len(greedy_plan.well_plans)} wells, NPV {greedy_npv:,.0f} "
              f"(RL vs greedy {uplift:+.2f}%)")
        tracker.log_metrics({"greedy_npv": greedy_npv, "greedy_wells": len(greedy_plan.well_plans),
                             "uplift_percent": uplift})
        tracker.log_metrics({f"greedy.{key}": value
                             for key, value in plan_metrics(greedy_plan, window, args.drilling_crews,
                                                            args.gtm_crews).items()})

    if args.out:
        schedule_frame(plan).to_csv(args.out, index=False)
        print(f"schedule: {args.out}")
        tracker.log_artifact(args.out, artifact_path="outputs")
    if tracker.run_id:
        print(f"mlflow run: {tracker.run_id} (experiment {args.experiment})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
