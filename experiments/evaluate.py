#!/usr/bin/env python3
"""Compare the greedy planner, a trained policy and exhaustive search over well orders.

For every sampled subset three plans are built under the same scheduling,
production and economic model:

* ``greedy`` - :class:`hyperactive.greedy.PlanBuilder` (largest NPV first);
* ``rl`` - the trained policy, one deterministic episode;
* ``exhaustive`` - the best plan over all priority orders of the subset. The
  builder is forced to follow a permutation (``keep_order=True``); a well that
  is not ready yet is skipped until it becomes a candidate. Feasible only for
  small subsets (``n!`` plans).

Example::

    python experiments/evaluate.py --model models/arrive39 --subsets 5 --well-count 6 \\
        --drilling-crews 2 --gtm-crews 1 --horizon-years 10 --drilling-months 24
"""

from __future__ import annotations

import argparse
import math
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyperactive.data import load_coordinates, load_wells  # noqa: E402
from hyperactive.env import PlanEnv  # noqa: E402
from hyperactive.greedy import PlanBuilder  # noqa: E402
from hyperactive.inference import load_policy, plan_with_policy  # noqa: E402
from hyperactive.planning import ClusterRandomRiskStrategy, DistanceTeamMovement, TeamManager  # noqa: E402
from hyperactive.scenario import (  # noqa: E402
    default_npv,
    default_profile,
    horizon_end,
    make_team_pool,
    oil_constraints,
    work_window_end,
)
from hyperactive.tracking import RunTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy vs RL vs exhaustive search on small well subsets.")
    p.add_argument("--wells", default=str(ROOT / "data" / "synthetic" / "wells.csv"))
    p.add_argument("--coordinates", default=str(ROOT / "data" / "synthetic" / "clusters.csv"))
    p.add_argument("--model", default=str(ROOT / "models" / "arrive39"))
    p.add_argument("--subsets", type=int, default=10)
    p.add_argument("--well-count", type=int, default=6)
    p.add_argument("--sampling", choices=("wells", "clusters"), default="clusters")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--horizon-years", type=int, default=10)
    p.add_argument("--drilling-months", type=int, default=None)
    p.add_argument("--drilling-crews", type=int, default=2)
    p.add_argument("--gtm-crews", type=int, default=1)
    p.add_argument("--oil-cap-share", type=float, default=None,
                   help="annual oil cap as a share of the subset's initial yearly rate")
    p.add_argument("--exhaustive-max-wells", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional CSV with per-subset results")
    p.add_argument("--experiment", default="hyperactive-evaluation",
                   help="MLflow experiment name")
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI; default: $HYPERACTIVE_MLFLOW_TRACKING_URI "
                        "or the local file store file:./mlruns")
    p.add_argument("--no-tracking", action="store_true",
                   help="do not record the run in MLflow")
    return p.parse_args()


class Scenario:
    def __init__(self, args: argparse.Namespace, movement: DistanceTeamMovement) -> None:
        self.args = args
        self.movement = movement
        self.start = datetime.fromisoformat(args.start)
        self.end = horizon_end(self.start, args.horizon_years)
        self.end_jobs = work_window_end(self.start, self.end, args.drilling_months)

    def oil_bound(self, wells: list[Any]) -> Optional[float]:
        if self.args.oil_cap_share is None:
            return None
        return self.args.oil_cap_share * sum(float(w.oil_rate) for w in wells) * 365.0

    def builder(self, wells: list[Any]) -> PlanBuilder:
        return PlanBuilder(
            start=self.start, end=self.end, end_jobs=self.end_jobs,
            cost_function=default_npv(self.start), production_profile=default_profile(),
            constraints=oil_constraints(self.oil_bound(wells)),
        )

    def manager(self) -> TeamManager:
        return TeamManager(
            team_pool=make_team_pool(self.args.drilling_crews, self.args.gtm_crews),
            movement=self.movement,
        )

    def greedy(self, wells: list[Any]) -> tuple[float, int]:
        plan = self.builder(wells).compile(
            wells=deepcopy(wells), manager=self.manager(),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        )
        return float(plan.total_profit()), len(plan.well_plans)

    def fixed_order(self, wells: list[Any], order: tuple[int, ...]) -> float:
        ordered = deepcopy([wells[i] for i in order])
        for position, well in enumerate(ordered):
            well.init_entry_date = self.start + timedelta(days=position)
        plan = self.builder(wells).compile(
            wells=ordered, manager=self.manager(),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0), keep_order=True,
        )
        return float(plan.total_profit())

    def rl(self, wells: list[Any], bundle) -> tuple[float, int]:
        env = PlanEnv(
            wells=deepcopy(wells),
            team_pool=make_team_pool(self.args.drilling_crews, self.args.gtm_crews),
            movement=self.movement, cost_function=default_npv(self.start),
            n_actions=bundle.n_actions, start=self.start, end=self.end, end_jobs=self.end_jobs,
            production_profile=default_profile(),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
            constraints=oil_constraints(self.oil_bound(wells)),
        )
        plan, _ = plan_with_policy(env, bundle)
        return float(plan.total_profit()), len(plan.well_plans)


def draw_subset(pool: list[Any], size: int, sampling: str, rng: np.random.Generator) -> list[Any]:
    if sampling == "wells":
        return [pool[int(i)] for i in rng.choice(len(pool), size=size, replace=False)]
    by_cluster: dict[str, list[Any]] = {}
    for well in pool:
        by_cluster.setdefault(str(well.cluster), []).append(well)
    clusters = [c for c, ws in by_cluster.items() if len(ws) >= 2]
    picked: list[Any] = []
    for i in rng.permutation(len(clusters)):
        picked.extend(by_cluster[clusters[int(i)]])
        if len(picked) >= size:
            break
    return picked[:size]


def main() -> int:
    args = parse_args()
    tracker = RunTracker(tracking_uri=args.tracking_uri, experiment=args.experiment,
                         enabled=not args.no_tracking)
    with tracker.run(f"{Path(args.model).name}-w{args.well_count}",
                     tags={"run_type": "evaluation", "model.key": Path(args.model).name}):
        return _evaluate(args, tracker)


def _evaluate(args: argparse.Namespace, tracker: RunTracker) -> int:
    wells, _ = load_wells(args.wells)
    scenario = Scenario(args, DistanceTeamMovement.from_dicts(load_coordinates(args.coordinates, wells)))
    bundle = load_policy(args.model)
    rng = np.random.default_rng(args.seed)

    tracker.log_environment()
    tracker.log_params({key: value for key, value in vars(args).items()
                        if key not in {"tracking_uri", "experiment", "no_tracking"}})
    tracker.log_seeds(subsets=args.seed)
    tracker.log_dataset(args.wells, "wells")
    tracker.log_dataset(args.coordinates, "coordinates")
    tracker.log_model_artifacts(args.model)

    rows = []
    attempts = 0
    while len(rows) < args.subsets and attempts < args.subsets * 50:
        attempts += 1
        subset = draw_subset(wells, args.well_count, args.sampling, rng)
        greedy_npv, greedy_size = scenario.greedy(subset)
        if greedy_size == 0:
            continue  # nothing fits into the work window
        rl_npv, rl_size = scenario.rl(subset, bundle)
        row = {"subset": ";".join(str(w.name) for w in subset),
               "greedy_npv": greedy_npv, "greedy_wells": greedy_size,
               "rl_npv": rl_npv, "rl_wells": rl_size,
               "rl_vs_greedy_percent": (rl_npv / greedy_npv - 1.0) * 100.0 if greedy_npv > 0 else np.nan}
        if len(subset) <= args.exhaustive_max_wells:
            best = max(scenario.fixed_order(subset, order) for order in permutations(range(len(subset))))
            row["exhaustive_npv"] = best
            row["greedy_gap_percent"] = (best - greedy_npv) / abs(best) * 100.0 if best else np.nan
            row["rl_gap_percent"] = (best - rl_npv) / abs(best) * 100.0 if best else np.nan
            row["orders_checked"] = math.factorial(len(subset))
        rows.append(row)
        # The subset index is the step: the series shows the spread over
        # subsets, which a single mean hides.
        tracker.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))},
                            step=len(rows))
        print(f"subset {len(rows):>3}: greedy {greedy_npv:,.0f} | rl {rl_npv:,.0f}"
              + (f" | exhaustive {row['exhaustive_npv']:,.0f}" if "exhaustive_npv" in row else ""),
              flush=True)

    frame = pd.DataFrame(rows)
    if args.out:
        frame.to_csv(args.out, index=False)
    summary_columns = [c for c in ("rl_vs_greedy_percent", "greedy_gap_percent", "rl_gap_percent")
                       if c in frame.columns]
    summary = {f"{column}.{statistic}": value
               for column in summary_columns
               for statistic, value in frame[column].describe().items()}
    tracker.log_metrics(summary)
    tracker.log_metrics({"subsets": len(frame),
                         "rl_wins": float((frame["rl_npv"] > frame["greedy_npv"]).mean())
                         if len(frame) else float("nan")})
    tracker.log_dict(frame.to_dict(orient="records"), "subsets.json")
    if args.out:
        tracker.log_artifact(args.out, artifact_path="outputs")
    print(frame[summary_columns].describe().to_string())
    if tracker.run_id:
        print(f"mlflow run: {tracker.run_id} (experiment {args.experiment})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
