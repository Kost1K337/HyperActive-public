#!/usr/bin/env python3
"""Train a masked C-DQN planner on randomised subsets of a well pool.

Why randomise: with a fixed subset of wells all training episodes see the same
set, and the policy memorises an order for that set instead of learning a rule.
Here the set of wells is drawn anew on every environment reset, and the pool is
split upfront into a training part and a held-out part, so the final evaluation
runs on wells the policy has never seen.

Every episode also draws a crew configuration, an economic horizon, a work
window and (optionally) an annual oil cap, so the policy is trained on a
distribution of planning regimes rather than on one configuration.

Pipeline:

1. load the pool, split it into train / hold-out;
2. build ``RandomSubsetPlanEnv -> DummyVecEnv -> VecNormalize``;
3. optional warm start on several subsets (greedy replay prefill and behavioural cloning);
4. Q-learning with the C-DQN loss;
5. deterministic evaluation against the greedy planner on train and hold-out subsets;
6. save ``models/{model.zip, vec_normalize.pkl, manifest.json}``, loadable by
   :func:`hyperactive.inference.load_policy`.

The released ``bc39`` model was trained with the command stored in
``models/bc39/manifest.json`` (on a non-public pool). On the bundled synthetic
data::

    python experiments/train.py --run-name demo --rl-episodes 300 \\
        --crew-mix 2x1,3x2,5x5 --horizon-mix 5,10,25 --drilling-months-mix 12,24,60 \\
        --oil-constraint-prob 0.5 --net-width 64 --weight-decay 0.0001 \\
        --behavioral-cloning 1 --greedy-prefill 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyperactive.data import load_coordinates, load_wells  # noqa: E402
from hyperactive.env import FEATURE_SET, PlanEnv  # noqa: E402
from hyperactive.greedy import PlanBuilder  # noqa: E402
from hyperactive.inference import sha256_of  # noqa: E402
from hyperactive.models import BehavioralCloningConfig, MaskedCDQN  # noqa: E402
from hyperactive.planning import (  # noqa: E402
    ClusterRandomRiskStrategy,
    ConstraintManager,
    DistanceTeamMovement,
    TeamManager,
)
from hyperactive.scenario import (  # noqa: E402
    default_profile,
    horizon_end,
    load_settings,
    make_npv,
    make_team_pool,
    oil_constraints,
    planning_start,
    work_window_end,
)
from hyperactive.tracking import RunTracker  # noqa: E402

# Configuration used where an episode does not draw its own (hot start subsets
# before the first draw, and runs without crew/horizon mixes).
DEFAULT_DRILLING_CREWS = 2
DEFAULT_GTM_CREWS = 2
DEFAULT_HORIZON_YEARS = 25

BUFFER_SIZE = 100_000
GREEDY_PREFILL_REPEATS = 3
MAX_SUBSET_RESAMPLES = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a masked C-DQN planner on randomised well subsets.")
    data = p.add_argument_group("data")
    data.add_argument("--wells-file", default=str(ROOT / "data" / "synthetic" / "wells.csv"))
    data.add_argument("--coordinates-file", default=str(ROOT / "data" / "synthetic" / "clusters.csv"))
    data.add_argument("--output-dir", default=None, help="default: runs/<run-name>")
    data.add_argument("--settings", default=None,
                      help="JSON overriding the project settings: economics, readiness_hour, "
                           "days_per_year (see hyperactive.scenario.load_settings)")

    run = p.add_argument_group("run")
    run.add_argument("--run-name", required=True)
    run.add_argument("--well-count", type=int, default=16, help="size of every training subset")
    run.add_argument("--action-window", type=int, default=8, help="number of candidates per step")
    run.add_argument("--rl-episodes", type=int, default=10_000,
                     help="training budget: total timesteps = rl_episodes * well_count")
    run.add_argument("--seed", type=int, default=20260608)
    # The pool split and evaluation subsets use their own seed so that changing
    # the training seed does not change the task as well.
    run.add_argument("--split-seed", type=int, default=None, help="default: --seed")
    run.add_argument("--device", default="cpu")
    run.add_argument("--holdout-fraction", type=float, default=0.2)
    run.add_argument("--eval-subsets", type=int, default=30)
    # Random wells break the cluster structure: a random subset mostly lands on
    # distinct clusters, crew moves stop depending on the order, and the greedy
    # plan becomes optimal. Sampling whole clusters keeps the structure the
    # planner faces at inference.
    run.add_argument("--sampling", choices=("wells", "clusters"), default="wells")
    # The greedy plan is only a metric: the policy never sees it. It is costly on
    # large subsets, so it can be computed on every k-th episode only.
    run.add_argument("--greedy-every", type=int, default=1)
    run.add_argument("--save-interval", type=int, default=1_000)

    regime = p.add_argument_group("per-episode regime randomisation")
    regime.add_argument("--crew-mix", default=None,
                        help="comma-separated 'drilling x GTM' crew configurations, e.g. 2x1,3x2")
    regime.add_argument("--horizon-mix", default=None, help="economic horizons in years, e.g. 5,10,25")
    regime.add_argument("--horizon-years", type=int, default=None,
                        help="fixed economic horizon when --horizon-mix is not given (default 25)")
    regime.add_argument("--drilling-months-mix", default=None,
                        help="work windows in months, e.g. 12,24,60. The work window bounds when "
                             "tasks may run; the economic horizon bounds how long NPV accumulates")
    # Without a cap the oil_window_* features are constant at 1 during training,
    # and a policy that meets a cap at inference receives out-of-range inputs.
    # The cap is a share of "the whole subset producing at its initial rate for
    # a year", drawn log-uniformly from the given range.
    regime.add_argument("--oil-constraint-prob", type=float, default=0.0,
                        help="share of episodes with an annual oil cap")
    regime.add_argument("--oil-constraint-share", default="0.08:0.35")

    algo = p.add_argument_group("algorithm")
    algo.add_argument("--use-cdqn", type=int, choices=(0, 1), default=1)
    algo.add_argument("--net-width", type=int, default=256, help="hidden width of MaskedQNetwork")
    algo.add_argument("--weight-decay", type=float, default=0.0,
                      help="L2 via AdamW; 0 keeps Adam without regularisation")
    algo.add_argument("--exploration-initial-eps", type=float, default=1.0)
    algo.add_argument("--exploration-final-eps", type=float, default=0.05)
    algo.add_argument("--exploration-fraction", type=float, default=0.8)

    warm = p.add_argument_group("warm start")
    # Warm start on several subsets: cloning the greedy policy on a single set
    # is exactly the memorisation the randomisation avoids.
    warm.add_argument("--hot-start-subsets", type=int, default=8)
    # Cloning pulls the policy towards the greedy one; it can be disabled to
    # check whether it sets the ceiling.
    warm.add_argument("--behavioral-cloning", type=int, choices=(0, 1), default=1)
    warm.add_argument("--bc-patience", type=int, default=8,
                      help="cloning stops after this many epochs without validation improvement")
    warm.add_argument("--greedy-prefill", type=int, choices=(0, 1), default=1)

    track = p.add_argument_group("tracking")
    track.add_argument("--experiment", default="hyperactive-training",
                       help="MLflow experiment name")
    track.add_argument("--tracking-uri", default=None,
                       help="MLflow tracking URI; default: $HYPERACTIVE_MLFLOW_TRACKING_URI "
                            "or the local file store file:./mlruns")
    track.add_argument("--no-tracking", action="store_true",
                       help="do not record the run in MLflow")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    try:
        th.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        th.use_deterministic_algorithms(True)


def parse_share_range(text: str) -> tuple[float, float]:
    values = [float(part) for part in str(text).split(":") if part.strip()]
    if not values:
        raise ValueError("empty oil cap share range")
    low = values[0]
    high = values[1] if len(values) > 1 else values[0]
    if low <= 0 or high < low:
        raise ValueError(f"invalid oil cap share range: {text}")
    return low, high


def parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if not value:
        return None
    return [int(token.strip()) for token in value.split(",")]


def parse_crew_mix(value: Optional[str]) -> Optional[list[tuple[int, int]]]:
    if not value:
        return None
    pairs = []
    for token in value.split(","):
        drilling, gtm = token.strip().split("x")
        pairs.append((int(drilling), int(gtm)))
    return pairs


def build_cluster_index(pool: list[Any]) -> tuple[dict[str, list[Any]], list[str]]:
    """Wells by cluster; single-well clusters carry no grouping and are not sampled."""
    by_cluster: dict[str, list[Any]] = {}
    for well in pool:
        by_cluster.setdefault(str(well.cluster), []).append(well)
    usable = [c for c, ws in by_cluster.items() if len(ws) >= 2]
    return by_cluster, usable


def pick_by_clusters(by_cluster: dict[str, list[Any]], clusters: list[str],
                     size: int, rng: np.random.Generator) -> list[Any]:
    """Take whole clusters in random order until the subset has ``size`` wells."""
    picked: list[Any] = []
    for i in rng.permutation(len(clusters)):
        picked.extend(by_cluster[clusters[int(i)]])
        if len(picked) >= size:
            break
    return picked[:size]


def order_from_plan(plan: Any) -> list[str]:
    return [str(context.well.name) for context in plan.well_plans] if plan is not None else []


def behavioral_cloning_config(enabled: bool, patience: int = 8) -> BehavioralCloningConfig:
    return BehavioralCloningConfig(
        enabled=enabled,
        demo_episodes=10,
        demo_noise_prob=0.15,
        max_steps_multiplier=2,
        epochs=40,
        early_stopping_patience=patience,
        post_bc_exploration_initial_eps=0.3,
    )


class Trainer:
    """Data, movement model and greedy references shared by environment and evaluation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.settings = load_settings(args.settings)
        self.horizon_years = args.horizon_years or DEFAULT_HORIZON_YEARS
        wells, rejected = load_wells(args.wells_file, readiness_hour=self.settings.readiness_hour)
        self.project_start = planning_start(wells)
        self.all_wells = wells
        self.rejected = rejected
        self.movement = DistanceTeamMovement.from_dicts(load_coordinates(args.coordinates_file, wells))
        self._greedy_cache: dict[tuple, float] = {}

    def npv(self):
        return make_npv(self.project_start, self.settings.economics)

    def horizon_end(self, horizon_years: float):
        return horizon_end(self.project_start, horizon_years, self.settings.days_per_year)

    def default_greedy_plan(self, wells: list[Any]):
        """Greedy plan in the default configuration (2x2 crews, no work window, no cap)."""
        builder = PlanBuilder(
            start=self.project_start,
            end=self.horizon_end(self.horizon_years),
            cost_function=self.npv(),
            production_profile=default_profile(),
        )
        return builder.compile(
            wells=deepcopy(wells),
            manager=TeamManager(
                team_pool=make_team_pool(DEFAULT_DRILLING_CREWS, DEFAULT_GTM_CREWS),
                movement=self.movement,
            ),
            risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
            keep_order=False,
        )

    def greedy_npv(self, wells: list[Any], config: Optional[tuple[int, int, int]] = None,
                   drilling_months: Optional[int] = None,
                   oil_bound: Optional[float] = None) -> float:
        """NPV of the greedy plan for the subset, in the same regime as the episode."""
        key = (tuple(sorted(str(w.name) for w in wells)), config, drilling_months, oil_bound)
        cached = self._greedy_cache.get(key)
        if cached is None:
            if config is None:
                cached = float(self.default_greedy_plan(deepcopy(wells)).total_profit())
            else:
                drilling, gtm, horizon_years = config
                end = self.horizon_end(horizon_years)
                builder = PlanBuilder(
                    start=self.project_start, end=end,
                    end_jobs=work_window_end(self.project_start, end, drilling_months),
                    cost_function=self.npv(),
                    production_profile=default_profile(),
                    constraints=oil_constraints(oil_bound),
                )
                plan = builder.compile(
                    wells=deepcopy(wells),
                    manager=TeamManager(team_pool=make_team_pool(drilling, gtm), movement=self.movement),
                    risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
                    keep_order=False,
                )
                cached = float(plan.total_profit())
            self._greedy_cache[key] = cached
        return cached


class RandomSubsetPlanEnv(PlanEnv):
    """``PlanEnv`` that draws a new subset of wells and a new regime on every reset.

    The subset can be pinned temporarily (:meth:`fix`); warm start and
    deterministic evaluation need this because the greedy order must refer to
    the wells actually present in the environment. A pinned subset keeps the
    regime of the previous episode.
    """

    def __init__(self, trainer: Trainer, pool: list[Any], rng: np.random.Generator, **kwargs: Any) -> None:
        args = trainer.args
        self._trainer = trainer
        self._pool = pool
        self._size = int(args.well_count)
        self._oil_prob = float(args.oil_constraint_prob)
        self._oil_share = parse_share_range(args.oil_constraint_share)
        self._episode_oil_bound: Optional[float] = None
        self._rng = rng
        self._sampling = args.sampling
        self._greedy_every = max(1, int(args.greedy_every))
        self._episodes = 0
        self._by_cluster, self._clusters = build_cluster_index(pool)
        if args.sampling == "clusters" and not self._clusters:
            raise ValueError("the pool has no clusters with two or more wells")
        self._crew_mix = parse_crew_mix(args.crew_mix)
        self._horizon_mix = parse_int_list(args.horizon_mix)
        self._drilling_mix = parse_int_list(args.drilling_months_mix)
        self._episode_drilling_months: Optional[int] = None
        self.current_config: Optional[tuple[int, int, int]] = None
        self._fixed: Optional[list[Any]] = None
        self.current_names: tuple[str, ...] = ()
        self.current_greedy: float = float("nan")
        self._chosen: list[Any] = []
        self.degenerate_subsets = 0
        super().__init__(wells=self._pick(), **kwargs)

    def _pick(self) -> list[Any]:
        if self._fixed is not None:
            chosen = self._fixed
        elif self._sampling == "clusters":
            chosen = pick_by_clusters(self._by_cluster, self._clusters, self._size, self._rng)
        else:
            idx = self._rng.choice(len(self._pool), size=self._size, replace=False)
            chosen = [self._pool[int(i)] for i in idx]
        self.current_names = tuple(str(w.name) for w in chosen)
        # The environment works on copies; the greedy reference uses the originals.
        self._chosen = list(chosen)
        return [deepcopy(w) for w in chosen]

    def _apply_work_window(self) -> None:
        self.end_jobs = work_window_end(self.start, self.end, self._episode_drilling_months)

    def _roll_configuration(self) -> None:
        """Draw crews, economic horizon and work window for the episode."""
        if self._fixed is not None or (self._crew_mix is None and self._horizon_mix is None):
            return
        trainer = self._trainer
        drilling, gtm = (
            self._crew_mix[int(self._rng.integers(len(self._crew_mix)))]
            if self._crew_mix else (DEFAULT_DRILLING_CREWS, DEFAULT_GTM_CREWS)
        )
        horizon_years = (
            self._horizon_mix[int(self._rng.integers(len(self._horizon_mix)))]
            if self._horizon_mix else trainer.horizon_years
        )
        self.team_pool = make_team_pool(drilling, gtm)
        self.end = trainer.horizon_end(horizon_years)
        self._episode_drilling_months = (
            self._drilling_mix[int(self._rng.integers(len(self._drilling_mix)))]
            if self._drilling_mix else None
        )
        self._apply_work_window()
        self.current_config = (drilling, gtm, horizon_years)

    def _roll_oil_bound(self) -> None:
        """Draw the annual oil cap of the episode (log-uniform share of the subset's yearly rate)."""
        if self._fixed is not None or self._oil_prob <= 0.0:
            self._episode_oil_bound = None
            return
        if float(self._rng.random()) >= self._oil_prob:
            self._episode_oil_bound = None
            return
        low, high = self._oil_share
        # Log-uniform: binding caps lie in the lower half of the range, and a
        # uniform draw would give them far fewer episodes.
        share = float(np.exp(self._rng.uniform(np.log(low), np.log(high))))
        yearly = sum(self._rate_to_float(well.oil_rate) for well in self.wells) * 365.0
        self._episode_oil_bound = share * yearly if yearly > 0 else None

    def fix(self, wells: Optional[list[Any]]) -> None:
        self._fixed = wells

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self._roll_configuration()
        # Unconditionally: on a pinned subset the draw returns early and the
        # work window would otherwise remain equal to the economic horizon.
        self._apply_work_window()
        # A short work window can make a whole subset infeasible (infrastructure
        # is ready too late); such a subset gives an episode without a single
        # valid action and is redrawn.
        for _ in range(MAX_SUBSET_RESAMPLES):
            self.wells = self._pick()
            # The constraint is set before the base reset, which builds the
            # first candidate window with it.
            self._roll_oil_bound()
            self._constraints = ConstraintManager(oil_constraints(self._episode_oil_bound))
            obs, info = super().reset(seed=seed, options=options)
            if self._fixed is None and not bool(np.asarray(info["action_mask"]).any()):
                self.degenerate_subsets += 1
                continue
            need_greedy = (self._fixed is not None or self._greedy_every == 1
                           or (self._episodes + 1) % self._greedy_every == 1)
            self.current_greedy = (
                self._trainer.greedy_npv(self._chosen, self.current_config,
                                         self._episode_drilling_months, self._episode_oil_bound)
                if need_greedy else float("nan"))
            break
        else:
            raise RuntimeError(
                f"No usable subset found in {MAX_SUBSET_RESAMPLES} attempts: the horizon is too "
                "short for this pool."
            )
        self._episodes += 1
        return obs, info


class Recorder(BaseCallback):
    """Records the NPV of every episode together with the greedy NPV of the same subset."""

    def __init__(self, agent: MaskedCDQN, raw_env: RandomSubsetPlanEnv, env: VecNormalize,
                 output_dir: Path, save_interval: int,
                 tracker: Optional[RunTracker] = None) -> None:
        super().__init__(0)
        self.agent = agent
        self.raw_env = raw_env
        self.env = env
        self.output_dir = output_dir
        self.save_interval = max(1, int(save_interval))
        self.tracker = tracker
        self.rows: list[dict[str, Any]] = []
        self.started = time.perf_counter()
        self.saved = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None or not bool(np.asarray(dones).reshape(-1)[0]):
            return True
        info = dict(infos[0])
        plan = info.get("final_plan")
        npv_rl = float(info.get("final_npv", np.nan))
        # The environment is not reset yet, so the greedy value belongs to the
        # subset the episode ran on.
        npv_greedy = float(self.raw_env.current_greedy)
        self.rows.append({
            "episode": len(self.rows) + 1,
            "env_step": int(self.num_timesteps),
            "elapsed_seconds": time.perf_counter() - self.started,
            "npv_rl": npv_rl,
            "npv_greedy": npv_greedy,
            "uplift_percent": (npv_rl / npv_greedy - 1.0) * 100.0 if npv_greedy else np.nan,
            "beats_greedy": bool(npv_rl > npv_greedy) if np.isfinite(npv_greedy) else None,
            "plan_size": len(plan.well_plans) if plan is not None else 0,
            "epsilon": float(self.agent.exploration_rate),
            "subset": ";".join(self.raw_env.current_names),
        })
        if self.tracker is not None:
            row = self.rows[-1]
            # The episode number is the step of the series: MLflow then draws
            # the learning curve, and a query returns it without the CSV.
            self.tracker.log_metrics(
                {"npv_rl": row["npv_rl"], "npv_greedy": row["npv_greedy"],
                 "uplift_percent": row["uplift_percent"], "plan_size": row["plan_size"],
                 "epsilon": row["epsilon"], "env_step": row["env_step"],
                 **{k: v for k, v in (self.agent.last_update_info or {}).items()}},
                step=row["episode"])
        if len(self.rows) % self.save_interval == 0:
            self.flush()
            self._report()
        return True

    def flush(self) -> None:
        pd.DataFrame(self.rows).to_csv(self.output_dir / "episodes.csv", index=False)
        save_model(self.agent, self.env, self.output_dir / "models")
        self.saved = len(self.rows)

    def _report(self) -> None:
        tail = pd.DataFrame(self.rows[-self.save_interval:])
        measured = tail[tail["uplift_percent"].notna()]
        elapsed = time.perf_counter() - self.started
        if measured.empty:
            print(f"episode {len(self.rows):>6} | eps={self.agent.exploration_rate:.3f} | "
                  f"no greedy references | {elapsed:7.0f} s", flush=True)
            return
        print(
            f"episode {len(self.rows):>6} | eps={self.agent.exploration_rate:.3f} | "
            f"mean uplift {measured['uplift_percent'].mean():+6.2f}% | "
            f"win rate {measured['beats_greedy'].mean() * 100:5.1f}% | "
            f"references {len(measured):>4} | {elapsed:7.0f} s",
            flush=True,
        )


def save_model(agent: MaskedCDQN, env: VecNormalize, model_dir: Path, extra: Optional[dict] = None) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_dir / "model.zip"))
    env.save(str(model_dir / "vec_normalize.pkl"))
    manifest = {
        "feature_set": FEATURE_SET,
        "action_window": int(agent.n_actions),
        "files": {
            "model": {"path": "model.zip", "sha256": sha256_of(model_dir / "model.zip")},
            "vec_normalize": {"path": "vec_normalize.pkl",
                              "sha256": sha256_of(model_dir / "vec_normalize.pkl")},
        },
        **(extra or {}),
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_deterministic(agent: MaskedCDQN, raw_env: RandomSubsetPlanEnv, env: VecNormalize,
                      wells: list[Any], well_count: int) -> tuple[float, float, int]:
    """A deterministic episode on the given subset against the greedy plan of that subset."""
    raw_env.fix(wells)
    try:
        obs = env.reset()
        masks = agent.query_action_masks(env)
        for _ in range(well_count * 2 + 1):
            action, _ = agent.predict(obs, deterministic=True, action_masks=masks)
            obs, _, dones, infos = env.step(np.asarray(action, dtype=np.int64).reshape(-1))
            if bool(dones[0]):
                info = dict(infos[0])
                plan = info.get("final_plan")
                return (float(info.get("final_npv", np.nan)),
                        float(raw_env.current_greedy),
                        len(plan.well_plans) if plan is not None else 0)
            masks = agent.query_action_masks(env)
        return float("nan"), float(raw_env.current_greedy), 0
    finally:
        raw_env.fix(None)


def subset_is_viable(raw_env: RandomSubsetPlanEnv, wells: list[Any]) -> bool:
    """Whether the subset has at least one valid action at the first step."""
    raw_env.fix(wells)
    try:
        _, info = raw_env.reset()
        return bool(np.asarray(info["action_mask"]).any())
    finally:
        raw_env.fix(None)


def sample_subsets(args: argparse.Namespace, pool: list[Any], count: int,
                   rng: np.random.Generator, viable: Any = None) -> list[list[Any]]:
    """Subsets drawn the same way as in training; otherwise evaluation would measure another task."""
    if args.sampling == "clusters":
        by_cluster, clusters = build_cluster_index(pool)
        draw = lambda: pick_by_clusters(by_cluster, clusters, args.well_count, rng)  # noqa: E731
    else:
        draw = lambda: [pool[int(i)] for i in  # noqa: E731
                        rng.choice(len(pool), size=args.well_count, replace=False)]

    out: list[list[Any]] = []
    attempts = 0
    while len(out) < count and attempts < count * MAX_SUBSET_RESAMPLES:
        attempts += 1
        wells = draw()
        if viable is not None and not viable(wells):
            continue
        out.append(wells)
    if len(out) < count:
        raise RuntimeError(f"only {len(out)} of {count} subsets have valid actions")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "runs" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = RunTracker(tracking_uri=args.tracking_uri, experiment=args.experiment,
                         enabled=not args.no_tracking)
    with tracker.run(args.run_name, tags={"run_type": "train",
                                          "feature_set": FEATURE_SET,
                                          "sampling": args.sampling}):
        return _train(args, output_dir, tracker)


def _train(args: argparse.Namespace, output_dir: Path, tracker: RunTracker) -> int:
    trainer = Trainer(args)
    print(f"pool: {len(trainer.all_wells)} valid wells, {trainer.rejected} rejected", flush=True)
    tracker.log_environment()
    tracker.log_params({key: value for key, value in vars(args).items()
                        if key not in {"tracking_uri", "experiment", "no_tracking"}})
    tracker.log_seeds(training=args.seed,
                      split=args.seed if args.split_seed is None else args.split_seed)
    tracker.log_dataset(args.wells_file, "wells")
    tracker.log_dataset(args.coordinates_file, "coordinates")
    if args.settings:
        tracker.log_dataset(args.settings, "settings", attach=True)
    tracker.log_params({"settings": trainer.settings.to_dict()})
    tracker.log_dict(trainer.settings.to_dict(), "settings.json")
    tracker.log_params({"pool.valid_wells": len(trainer.all_wells),
                        "pool.rejected_rows": trainer.rejected})
    set_seed(args.seed)

    split_seed = args.seed if args.split_seed is None else args.split_seed
    order = np.random.default_rng(split_seed).permutation(len(trainer.all_wells))
    n_hold = max(args.well_count * 2, int(round(len(trainer.all_wells) * args.holdout_fraction)))
    holdout_pool = [trainer.all_wells[i] for i in order[:n_hold]]
    train_pool = [trainer.all_wells[i] for i in order[n_hold:]]
    print(f"train pool: {len(train_pool)} wells | hold-out: {len(holdout_pool)}", flush=True)
    tracker.log_params({"pool.train_size": len(train_pool),
                        "pool.holdout_size": len(holdout_pool)})

    started = datetime.now(timezone.utc)
    total_start = time.perf_counter()
    env_rng = np.random.default_rng(args.seed + 1)
    raw_env = RandomSubsetPlanEnv(
        trainer, train_pool, env_rng,
        team_pool=make_team_pool(DEFAULT_DRILLING_CREWS, DEFAULT_GTM_CREWS),
        movement=trainer.movement,
        production_profile=default_profile(),
        cost_function=trainer.npv(),
        n_actions=args.action_window,
        risk_strategy=ClusterRandomRiskStrategy(trigger_chance=0.0),
        start=trainer.project_start,
        end=trainer.horizon_end(trainer.horizon_years),
    )
    env = VecNormalize(DummyVecEnv([lambda: raw_env]), training=True, norm_obs=True, norm_reward=True)

    total_steps = args.rl_episodes * args.well_count
    batch_size = max(2, min(128, max(args.well_count * 4, 8)))
    learning_starts = max(batch_size * 2,
                          min(max(args.well_count * 10, batch_size * 2),
                              max(batch_size * 2, total_steps // 5)))
    max_steps = max(1, args.well_count * 2)

    policy_kwargs: dict[str, Any] = {"net_arch": [args.net_width]}
    if args.weight_decay > 0:
        # AdamW rather than Adam(weight_decay=...): in Adam the L2 term is mixed
        # into the moments and rescaled by the adaptive step.
        policy_kwargs["optimizer_class"] = th.optim.AdamW
        policy_kwargs["optimizer_kwargs"] = {"weight_decay": args.weight_decay}

    agent = MaskedCDQN(
        "MlpPolicy", env, learning_rate=1e-4, buffer_size=BUFFER_SIZE,
        learning_starts=learning_starts, batch_size=batch_size, gamma=0.99,
        train_freq=1, gradient_steps=1,
        target_update_interval=max(100, args.well_count * 10),
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        max_grad_norm=10.0, use_cdqn=bool(args.use_cdqn), policy_kwargs=policy_kwargs,
        seed=args.seed, device=args.device, verbose=0,
    )

    # --- warm start on several subsets -------------------------------------
    hot_rng = np.random.default_rng(args.seed + 2)
    hot_stats: list[dict[str, Any]] = []
    hot_start_begin = time.perf_counter()
    bc_enabled = bool(args.behavioral_cloning)
    prefill_episodes = GREEDY_PREFILL_REPEATS if args.greedy_prefill else 0
    subsets = (sample_subsets(args, train_pool, args.hot_start_subsets, hot_rng)
               if (bc_enabled or prefill_episodes) else [])
    if not subsets:
        print("warm start disabled: the policy learns from scratch", flush=True)
    for k, wells in enumerate(subsets):
        raw_env.fix(wells)
        try:
            stats = agent.hot_start(
                greedy_order=order_from_plan(trainer.default_greedy_plan(deepcopy(wells))),
                bc_config=behavioral_cloning_config(bc_enabled, args.bc_patience),
                prefill_episodes=prefill_episodes,
                max_steps=max_steps, output_dir=None, seed=args.seed + k,
            )
            bc = stats.get("behavioral_cloning") or {}
            hot_stats.append({"subset": k + 1,
                              "val_accuracy": bc.get("final_val_accuracy"),
                              "val_loss": bc.get("final_val_loss")})
        finally:
            raw_env.fix(None)
    hot_start_seconds = time.perf_counter() - hot_start_begin
    tracker.log_metrics({"hot_start_seconds": hot_start_seconds,
                         "hot_start_subsets": len(subsets)})
    if hot_stats:
        tracker.log_dict(hot_stats, "hot_start.json")

    # --- training -----------------------------------------------------------
    recorder = Recorder(agent, raw_env, env, output_dir, args.save_interval, tracker)
    train_begin = time.perf_counter()
    agent.learn(total_timesteps=total_steps, callback=recorder, progress_bar=False)
    if len(recorder.rows) != recorder.saved:
        recorder.flush()
    train_seconds = time.perf_counter() - train_begin

    # --- evaluation on train and hold-out pools -----------------------------
    env.training = False
    env.norm_reward = False
    eval_rng = np.random.default_rng(split_seed + 3)
    evaluation: dict[str, Any] = {}
    for tag, pool in (("train", train_pool), ("holdout", holdout_pool)):
        rows = []
        for wells in sample_subsets(args, pool, args.eval_subsets, eval_rng,
                                    viable=lambda w: subset_is_viable(raw_env, w)):
            npv_rl, npv_greedy, size = run_deterministic(agent, raw_env, env, wells, args.well_count)
            # A non-positive greedy NPV makes the ratio meaningless; such subsets
            # are excluded from the uplift statistics.
            uplift = ((npv_rl / npv_greedy - 1.0) * 100.0
                      if np.isfinite(npv_greedy) and npv_greedy > 0 else np.nan)
            rows.append({"npv_rl": npv_rl, "npv_greedy": npv_greedy, "plan_size": size,
                         "uplift_percent": uplift})
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"eval_{tag}.csv", index=False)
        measurable = df[df["uplift_percent"].notna()]
        # The mean percentage over subsets and the uplift in money differ:
        # small subsets give cheap percentages. The aggregate is the second one.
        aggregate = (float(measurable["npv_rl"].sum() / measurable["npv_greedy"].sum() - 1.0) * 100.0
                     if len(measurable) and measurable["npv_greedy"].sum() > 0 else float("nan"))
        evaluation[tag] = {
            "subsets": int(len(df)),
            "subsets_measurable": int(len(measurable)),
            "mean_uplift_percent": float(measurable["uplift_percent"].mean()),
            "median_uplift_percent": float(measurable["uplift_percent"].median()),
            "aggregate_uplift_percent": aggregate,
            "worst_uplift_percent": float(measurable["uplift_percent"].min()),
            "best_uplift_percent": float(measurable["uplift_percent"].max()),
            "win_rate": float((df["npv_rl"] > df["npv_greedy"]).mean()),
        }
        print(f"evaluation on {tag}: " + json.dumps(evaluation[tag]), flush=True)
        tracker.log_metrics({f"eval.{tag}.{key}": value
                             for key, value in evaluation[tag].items()})
        tracker.log_artifact(output_dir / f"eval_{tag}.csv", artifact_path="outputs")

    config = {key: value for key, value in vars(args).items()
              if key not in {"wells_file", "coordinates_file", "output_dir"}}
    # The resolved settings, not just the --settings path: a model is only
    # interpretable next to the economics its rewards were priced in.
    save_model(agent, env, output_dir / "models",
               extra={"name": args.run_name, "train_args": config,
                      "settings": trainer.settings.to_dict()})

    history = pd.DataFrame(recorder.rows)
    summary = {
        "run_name": args.run_name,
        "args": config,
        "train_pool_size": len(train_pool),
        "holdout_pool_size": len(holdout_pool),
        "rl_episodes_completed": int(len(history)),
        "hot_start_stats": hot_stats,
        "hot_start_seconds": hot_start_seconds,
        "rl_training_seconds": train_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "degenerate_subsets_resampled": int(raw_env.degenerate_subsets),
        "unique_subsets_seen": int(history["subset"].nunique()) if len(history) else 0,
        "evaluation": evaluation,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tracker.log_metrics({"rl_episodes_completed": summary["rl_episodes_completed"],
                         "rl_training_seconds": train_seconds,
                         "total_seconds": summary["total_seconds"],
                         "unique_subsets_seen": summary["unique_subsets_seen"],
                         "degenerate_subsets_resampled": summary["degenerate_subsets_resampled"]})
    tracker.log_dict(summary, "summary.json")
    tracker.log_artifact(output_dir / "episodes.csv", artifact_path="outputs")
    # The trained model goes into the run with the sha256 of its weights: a
    # path does not identify weights, they are replaced in place.
    tracker.log_model_artifacts(output_dir / "models")
    if tracker.run_id:
        print(f"mlflow run: {tracker.run_id} (experiment {args.experiment})", flush=True)
    print(f"artifacts: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
