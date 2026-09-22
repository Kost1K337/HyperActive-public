#!/usr/bin/env python3
"""Benchmark a trained policy against the greedy planner on a grid of funds, crews and work windows.

For every grid cell (fund x crews x work window) two plans are built:

* ``rl`` - the policy, ``--episodes`` rollouts with exploration ``--exploration``
  (episode ``k`` is seeded with ``k``), the plan with the highest NPV is kept;
* ``greedy`` - one deterministic greedy pass.

The uplift of the cell is ``NPV_rl / NPV_greedy - 1``.

Funds are discovered in the ``--cases`` directory: every workbook that is not a
coordinates workbook (name contains ``coord``) is a fund, and its directory
must contain exactly one coordinates workbook. The planning start of a fund is
the earliest readiness date of its wells, so that every fund starts its work
window with the first ready well rather than with idle time.

Economics, readiness and calendar conventions are the project's own
(:mod:`hyperactive.scenario`), shared with ``hyperactive-plan`` and training,
and can be overridden with ``--settings``. On top of them the benchmark fixes:

* coordinates are read with the positional contract ``cluster, x, y, z``
  (first four columns after the index rule of ``pandas.read_excel(names=...)``);
* the economic horizon is ``--planning-months`` (default 240 months), the work
  window is the cell's drilling horizon.

Example::

    python benchmark/run_benchmark.py --model models/bc39 --cases benchmark/case_10 \\
        --funds 48 --horizons 12,24 --crews 2x1,3x2 --out runs/benchmark-48.xlsx
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyperactive.data import load_wells  # noqa: E402
from hyperactive.env import PlanEnv  # noqa: E402
from hyperactive.greedy import PlanBuilder  # noqa: E402
from hyperactive.inference import PolicyBundle, load_policy, plan_with_policy  # noqa: E402
from hyperactive.planning import (  # noqa: E402
    ArpsDeclineProductionProfile,
    ClusterRandomRiskStrategy,
    DistanceTeamMovement,
    TeamManager,
)
from hyperactive.scenario import load_settings, make_npv, make_team_pool, work_window_end  # noqa: E402
from hyperactive.tracking import RunTracker, plan_metrics  # noqa: E402

DEFAULT_HORIZONS = "12,18,24,36,48,60"
DEFAULT_CREWS = "2x1,2x2,3x2,3x3,4x3"
MAX_EPISODES = 100

COORD_HINTS = ("coord",)
DATE_COLUMNS = ("init_entry_date", "readiness_date")
# Readiness dates earlier than this year are parsing artefacts (numbers read as dates).
EARLIEST_READINESS_YEAR = 1990


@dataclass(frozen=True)
class Fund:
    name: str
    wells: Path
    coords: Path
    well_count: int
    start: datetime


@dataclass(frozen=True)
class Config:
    model_dir: str
    planning_months: int
    episodes: int
    exploration: float
    economics: dict[str, Any]
    readiness_hour: Optional[int]
    days_per_year: float


# ---------------------------------------------------------------------------
# Funds
# ---------------------------------------------------------------------------


def _normalize_header(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _readiness_dates(values: pd.Series) -> pd.Series:
    """Readiness dates of a column; numbers are not treated as dates."""
    kept = values.map(lambda value: value if isinstance(value, (str, datetime, pd.Timestamp)) else None)
    dates = pd.to_datetime(kept, errors="coerce").dropna()
    return dates[dates.dt.year >= EARLIEST_READINESS_YEAR]


def read_fund_book(path: Path, start_date: Optional[datetime]) -> tuple[int, datetime]:
    """Number of non-empty rows and the planning start of a fund."""
    frame = pd.read_excel(path).dropna(how="all")
    count = int(len(frame))
    if start_date is not None:
        return count, start_date
    columns = {_normalize_header(column): column for column in frame.columns}
    for key in DATE_COLUMNS:
        if key not in columns:
            continue
        dates = _readiness_dates(frame[columns[key]])
        if dates.empty:
            raise ValueError(f"{path}: no readiness dates in column {columns[key]!r}; pass --start-date")
        earliest = dates.min().to_pydatetime()
        return count, datetime(earliest.year, earliest.month, earliest.day)
    raise ValueError(f"{path}: no readiness date column ({', '.join(DATE_COLUMNS)}); pass --start-date")


def discover_funds(root: Path, start_date: Optional[datetime]) -> tuple[list[Fund], dict[str, str]]:
    funds: list[Fund] = []
    problems: dict[str, str] = {}
    for folder in sorted({path.parent for path in root.rglob("*.xlsx")}):
        books = sorted(path for path in folder.glob("*.xlsx") if not path.name.startswith("~"))
        coords = [path for path in books if any(h in path.name.lower() for h in COORD_HINTS)]
        wells = [path for path in books if path not in coords]
        if not wells:
            continue
        base = ".".join(folder.relative_to(root).parts) or root.name
        names = [base if len(wells) == 1 else f"{base}.{book.stem}" for book in wells]
        if len(coords) != 1:
            for name in names:
                problems[name] = f"{folder}: {len(coords)} coordinates workbooks, exactly one is required"
            continue
        for name, book in zip(names, wells):
            try:
                count, start = read_fund_book(book, start_date)
            except ValueError as exc:
                problems[name] = str(exc)
                continue
            funds.append(Fund(name, book, coords[0], count, start))
    return funds, problems


def contract_coordinates(path: Path) -> list[dict[str, Any]]:
    """Coordinates read with the positional ``cluster, x, y, z`` contract of the benchmark."""
    frame = pd.read_excel(path, header=0, names=["cluster", "x", "y", "z"])
    return frame.to_dict(orient="records")


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


_WORKER: dict[str, Any] = {}


def _bundle(model_dir: str) -> PolicyBundle:
    if _WORKER.get("model_dir") != model_dir:
        _WORKER["bundle"] = load_policy(model_dir)
        _WORKER["model_dir"] = model_dir
    return _WORKER["bundle"]


def run_cell(fund: Fund, drilling: int, gtm: int, months: int, cfg: Config) -> dict[str, Any]:
    wells, rejected = load_wells(fund.wells, readiness_hour=cfg.readiness_hour)
    movement = DistanceTeamMovement.from_dicts(contract_coordinates(fund.coords))
    start = fund.start
    days_per_month = cfg.days_per_year / 12.0
    plan_end = start + timedelta(days=int(round(cfg.planning_months * days_per_month)))
    jobs_end = work_window_end(start, plan_end, months)

    row: dict[str, Any] = {
        "fund": fund.name, "fund_wells": len(wells), "rejected_rows": rejected,
        "start": start.date().isoformat(), "crews": f"{drilling}x{gtm}",
        "drilling_crews": drilling, "gtm_crews": gtm, "work_window_months": months,
    }

    began = time.perf_counter()
    bundle = _bundle(cfg.model_dir)
    env = PlanEnv(
        wells=copy.deepcopy(wells), team_pool=make_team_pool(drilling, gtm), movement=movement,
        cost_function=make_npv(start, cfg.economics), n_actions=bundle.n_actions,
        start=start, end=plan_end, end_jobs=jobs_end,
        production_profile=ArpsDeclineProductionProfile(),
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
    )
    rl_plan, stats = plan_with_policy(env, bundle, episodes=cfg.episodes, exploration=cfg.exploration)
    row.update({
        "rl_wells": len(rl_plan.well_plans), "rl_npv": rl_plan.total_profit(),
        "rl_oil": rl_plan.total_oil(), "rl_selected_episode": stats["selected_episode"],
        "rl_seconds": round(time.perf_counter() - began, 1),
    })
    # Crew utilisation, span and per-well spread of both plans: explaining a
    # lost cell starts with how the two plans differ, and recomputing that from
    # the result table alone is impossible.
    window = (start, jobs_end)
    row["rl_plan_metrics"] = plan_metrics(rl_plan, window, drilling, gtm)
    row["rl_episode_npvs"] = stats["npvs"]

    began = time.perf_counter()
    greedy_plan = PlanBuilder(
        start=start, end=plan_end, end_jobs=jobs_end,
        cost_function=make_npv(start, cfg.economics),
        production_profile=ArpsDeclineProductionProfile(),
    ).compile(
        wells=copy.deepcopy(wells),
        manager=TeamManager(team_pool=make_team_pool(drilling, gtm), movement=movement),
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
    )
    row.update({
        "greedy_wells": len(greedy_plan.well_plans), "greedy_npv": greedy_plan.total_profit(),
        "greedy_oil": greedy_plan.total_oil(),
        "greedy_seconds": round(time.perf_counter() - began, 1),
    })
    row["greedy_plan_metrics"] = plan_metrics(greedy_plan, window, drilling, gtm)
    if row["greedy_npv"] > 0:
        row["uplift_percent"] = round((row["rl_npv"] / row["greedy_npv"] - 1.0) * 100.0, 2)
        row["delta_wells"] = row["rl_wells"] - row["greedy_wells"]
    return row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

COLUMNS = ["fund", "fund_wells", "start", "crews", "drilling_crews", "gtm_crews", "work_window_months",
           "rl_wells", "greedy_wells", "delta_wells", "rl_npv", "greedy_npv", "uplift_percent",
           "rl_oil", "greedy_oil", "rl_selected_episode", "rl_seconds", "greedy_seconds"]


NESTED_FIELDS = ("rl_plan_metrics", "greedy_plan_metrics", "rl_episode_npvs")


def write_results(rows: list[dict[str, Any]], out: Path, parameters: list[tuple[str, Any]]) -> None:
    frame = pd.DataFrame([{k: v for k, v in row.items() if k not in NESTED_FIELDS}
                          for row in rows])
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[COLUMNS + [c for c in frame.columns if c not in COLUMNS]]
    if out.suffix.lower() != ".xlsx":
        frame.to_csv(out, index=False)
        return
    done = frame[frame["uplift_percent"].notna()]
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="cells", index=False)
        pd.DataFrame(parameters, columns=["parameter", "value"]).to_excel(
            writer, sheet_name="parameters", index=False)
        if done.empty:
            return
        for column, sheet in (("fund", "by_fund"), ("crews", "by_crews"),
                              ("work_window_months", "by_work_window")):
            grouped = done.groupby(column)["uplift_percent"]
            table = grouped.agg(cells="count", median="median", worst="min", best="max")
            table["win_rate"] = grouped.apply(lambda series: (series > 0).mean())
            table.to_excel(writer, sheet_name=sheet)
        done.pivot_table(index=["fund", "crews"], columns="work_window_months",
                         values="uplift_percent", aggfunc="median").to_excel(writer, sheet_name="uplift_grid")


# ---------------------------------------------------------------------------
# Arguments and main
# ---------------------------------------------------------------------------


def _int_list(raw: str, what: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise SystemExit(f"{what}: positive integers expected, got {raw!r}")
    return sorted(set(values))


def _crew_list(raw: str) -> list[tuple[int, int]]:
    crews = []
    for part in raw.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", part)
        if not match or int(match.group(1)) < 1:
            raise SystemExit(f"--crews: 'drilling x GTM' expected, e.g. 3x2, got {part!r}")
        crews.append((int(match.group(1)), int(match.group(2))))
    return sorted(set(crews))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(ROOT / "models" / "bc39"))
    p.add_argument("--cases", type=Path, default=ROOT / "benchmark" / "case_10")
    p.add_argument("--funds", default="", help="only these funds, comma-separated")
    p.add_argument("--horizons", default=DEFAULT_HORIZONS, help="work windows, months")
    p.add_argument("--crews", default=DEFAULT_CREWS, help="'drilling x GTM' crew configurations")
    p.add_argument("--planning-months", type=int, default=240, help="economic horizon, months")
    p.add_argument("--episodes", type=int, default=10, help=f"RL rollouts per cell, 1..{MAX_EPISODES}")
    p.add_argument("--exploration", type=float, default=0.15, help="probability of a random valid action")
    p.add_argument("--start-date", default=None, help="one start date for all funds, YYYY-MM-DD")
    p.add_argument("--settings", type=Path, default=None,
                   help="JSON overriding the project settings: economics, readiness_hour, "
                        "days_per_year (see hyperactive.scenario.load_settings)")
    p.add_argument("--workers", type=int, default=1, help="cells computed in parallel processes")
    p.add_argument("--out", type=Path, default=None, help=".xlsx or .csv; default runs/benchmark-<model>.xlsx")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--experiment", default="hyperactive-benchmarks",
                   help="MLflow experiment name")
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI; default: $HYPERACTIVE_MLFLOW_TRACKING_URI "
                        "or the local file store file:./mlruns")
    p.add_argument("--no-tracking", action="store_true",
                   help="do not record the suite in MLflow")
    p.add_argument("--suite", default=None,
                   help="suite name for the MLflow run; default: the --cases directory name")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.episodes <= MAX_EPISODES:
        raise SystemExit(f"--episodes: 1..{MAX_EPISODES}")
    if not 0.0 <= args.exploration <= 1.0:
        raise SystemExit("--exploration: 0..1")
    settings = load_settings(args.settings)
    economics = settings.economics
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    cfg = Config(model_dir=str(args.model), planning_months=args.planning_months,
                 episodes=args.episodes, exploration=args.exploration, economics=economics,
                 readiness_hour=settings.readiness_hour, days_per_year=settings.days_per_year)

    found, problems = discover_funds(args.cases.resolve(), start_date)
    wanted = {name.strip() for name in args.funds.split(",") if name.strip()}
    funds = [fund for fund in found if not wanted or fund.name in wanted]
    blocking = {name: text for name, text in problems.items() if not wanted or name in wanted}
    if blocking:
        for name, text in sorted(blocking.items()):
            print(f"cannot run fund {name}: {text}", file=sys.stderr)
        return 2
    if not funds:
        raise SystemExit(f"no funds found in {args.cases}")

    horizons = _int_list(args.horizons, "--horizons")
    crews = _crew_list(args.crews)
    grid = [(fund, drilling, gtm, months) for fund in funds for months in horizons for drilling, gtm in crews]
    model_name = Path(args.model).name
    out = args.out or ROOT / "runs" / f"benchmark-{model_name}.xlsx"
    print(f"model {model_name} | funds {len(funds)} | windows {len(horizons)} | crews {len(crews)} | "
          f"cells {len(grid)}")
    for fund in funds:
        print(f"  {fund.name}: {fund.well_count} rows, start {fund.start.date()}, "
              f"{fund.wells.name} + {fund.coords.name}")
    if args.dry_run:
        return 0

    parameters = [
        ("model", model_name), ("cases", str(args.cases)),
        ("work windows, months", ",".join(map(str, horizons))),
        ("crews", ",".join(f"{d}x{g}" for d, g in crews)),
        ("economic horizon, months", args.planning_months),
        ("rl episodes", args.episodes), ("exploration", args.exploration),
        ("profile", "arps"), ("baseline", "greedy, one deterministic pass"),
        ("economics", json.dumps(economics, ensure_ascii=False)),
        ("readiness hour", "table timestamp" if cfg.readiness_hour is None else cfg.readiness_hour),
        ("days per year", cfg.days_per_year),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    suite = args.suite or args.cases.resolve().name
    tracker = RunTracker(tracking_uri=args.tracking_uri, experiment=args.experiment,
                         enabled=not args.no_tracking)
    rows: list[dict[str, Any]] = []
    started = time.time()
    with tracker.run(f"{model_name}-{suite}",
                     tags={"run_type": "benchmark", "bench.suite": suite,
                           "model.key": model_name}):
        _run_grid(args, cfg, grid, rows, out, parameters, tracker, suite, model_name, started)
    uplift = pd.DataFrame([{k: v for k, v in row.items() if k not in NESTED_FIELDS}
                           for row in rows]).get("uplift_percent")
    if uplift is not None and uplift.notna().any():
        print(f"median uplift {uplift.median():.2f}% | mean {uplift.mean():.2f}% | "
              f"win rate {(uplift > 0).mean():.3f} | cells {int(uplift.notna().sum())}")
    print(f"results: {out}")
    return 0


def _run_grid(args, cfg, grid, rows, out, parameters, tracker, suite, model_name, started) -> None:
    """The grid itself: one nested MLflow run per cell, aggregates on the suite run."""
    tracker.log_environment()
    tracker.log_params({"model": model_name, "suite": suite, "cases": str(args.cases),
                        "planning_months": args.planning_months, "episodes": args.episodes,
                        "exploration": args.exploration, "cells": len(grid),
                        "economics": cfg.economics, "readiness_hour": cfg.readiness_hour,
                        "days_per_year": cfg.days_per_year})
    tracker.log_dict(cfg.economics, "economics.json")
    tracker.log_model_artifacts(args.model)
    for fund in sorted({item[0] for item in grid}, key=lambda f: f.name):
        tracker.log_dataset(fund.wells, f"fund.{fund.name}.wells")
        tracker.log_dataset(fund.coords, f"fund.{fund.name}.coordinates")

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_cell, fund, d, g, m, cfg): (fund, d, g, m) for fund, d, g, m in grid}
        for future in as_completed(futures):
            fund, d, g, m = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # one cell must not stop the grid
                row = {"fund": fund.name, "crews": f"{d}x{g}", "drilling_crews": d, "gtm_crews": g,
                       "work_window_months": m, "error": repr(exc)}
            rows.append(row)
            write_results(rows, out, parameters)
            _track_cell(tracker, row, suite, model_name)
            print(f"[{len(rows)}/{len(grid)}] {fund.name} {d}x{g} {m}m: "
                  f"RL {row.get('rl_wells')} / greedy {row.get('greedy_wells')} wells, "
                  f"uplift {row.get('uplift_percent')}% | {(time.time() - started) / 60:.1f} min",
                  flush=True)
    _track_suite(tracker, rows, out)


def _track_cell(tracker: RunTracker, row: dict[str, Any], suite: str, model_name: str) -> None:
    """One nested run per grid cell: its conditions, both plans and the uplift."""
    cell = f"{row.get('fund')}/{row.get('crews')}/{row.get('work_window_months')}m"
    with tracker.run(cell, nested=True, log_duration=False,
                     tags={"run_type": "benchmark_cell", "bench.suite": suite,
                           "bench.cell": cell, "model.key": model_name,
                           "fund": str(row.get("fund"))}):
        tracker.log_params({k: v for k, v in row.items()
                            if k not in NESTED_FIELDS and not isinstance(v, (dict, list))
                            and k in ("fund", "fund_wells", "start", "crews", "drilling_crews",
                                      "gtm_crews", "work_window_months", "rejected_rows")})
        tracker.log_metrics({k: v for k, v in row.items()
                             if k not in NESTED_FIELDS and isinstance(v, (int, float))})
        tracker.log_metrics({"cell_seconds": float(row.get("rl_seconds") or 0.0)
                             + float(row.get("greedy_seconds") or 0.0)})
        for side in ("rl", "greedy"):
            metrics = row.get(f"{side}_plan_metrics") or {}
            tracker.log_metrics({f"{side}.{key}": value for key, value in metrics.items()})
        for index, npv in enumerate(row.get("rl_episode_npvs") or []):
            # The spread over episodes shows whether the cell was won by the
            # policy or by one lucky exploration rollout.
            tracker.log_metrics({"rl_episode_npv": npv}, step=index)
        if row.get("error"):
            tracker.log_tags({"status": "failed", "error": str(row["error"])[:480]})


def _track_suite(tracker: RunTracker, rows: list[dict[str, Any]], out: Path) -> None:
    """Aggregates of the suite: medians and win rates overall and by axis."""
    frame = pd.DataFrame([{k: v for k, v in row.items() if k not in NESTED_FIELDS}
                          for row in rows])
    tracker.log_artifact(out, artifact_path="outputs")
    if "uplift_percent" not in frame.columns:
        return
    done = frame[frame["uplift_percent"].notna()]
    if done.empty:
        return
    uplift = done["uplift_percent"]
    tracker.log_metrics({
        "cells_total": len(frame), "cells_measured": len(done),
        "uplift_median": uplift.median(), "uplift_mean": uplift.mean(),
        "uplift_worst": uplift.min(), "uplift_best": uplift.max(),
        "win_rate": (uplift > 0).mean(),
        "rl_npv_total": done["rl_npv"].sum(), "greedy_npv_total": done["greedy_npv"].sum(),
        # The aggregate in money, not the mean of percentages: small cells give
        # cheap percentages and would dominate an unweighted mean.
        "uplift_aggregate": (done["rl_npv"].sum() / done["greedy_npv"].sum() - 1.0) * 100.0
        if done["greedy_npv"].sum() > 0 else float("nan"),
    })
    for column, prefix in (("fund", "by_fund"), ("crews", "by_crews"),
                           ("work_window_months", "by_window")):
        for key, group in done.groupby(column)["uplift_percent"]:
            tracker.log_metrics({f"{prefix}.{key}.median": group.median(),
                                 f"{prefix}.{key}.win_rate": (group > 0).mean(),
                                 f"{prefix}.{key}.cells": len(group)})
    tracker.log_dict(done.to_dict(orient="records"), "cells.json")


if __name__ == "__main__":
    raise SystemExit(main())
