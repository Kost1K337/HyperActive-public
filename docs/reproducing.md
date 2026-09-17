# Reproducing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # Python 3.10-3.13
pip install -e ".[dev]"          # includes MLflow; ".[tracking]" for tracking only
```

`torch==2.7.1` and `stable-baselines3==2.7.1` are pinned exactly (see
`pyproject.toml`): `stable-baselines3 >= 2.8` requires `torch >= 2.8`, and an
unpinned `torch` on a plain PyPI install can pull a large CUDA build even on a
CPU-only machine. On Linux, installing from the CPU wheel index avoids that
regardless:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
```

## Synthetic data

No field data ships with this repository (see [`data-format.md`](data-format.md)
for why and for the input schema). `data/synthetic/{wells.csv,clusters.csv}`
is the same 64-well, 24-cluster fund as `benchmark/case_10/64` (see
[Reproducing the benchmark](#reproducing-the-benchmark) below), reshaped to
this schema so the examples below have somewhere realistic to run without
proprietary field data.

`hyperactive.data.synthetic` can generate an unrelated, purely procedural
pool of any size instead - useful for stress-testing rather than for
reproducing a specific result:

```bash
python -m hyperactive.data.synthetic --out data/synthetic --clusters 60 --seed 7
```

(this overwrites the bundled fund; re-copy it from `benchmark/case_10/64` to
get it back).

All examples below default to the bundled fund.

## Plan with the released model

`models/arrive39/` is a masked-C-DQN policy trained on an 8-candidate action
window over 16-well subsets (no behavioural cloning, no greedy prefill) —
see `models/arrive39/manifest.json` for the exact training distribution and
command, and [`method.md`](method.md) for what the algorithm does. Build a
plan and compare it with the greedy baseline:

```bash
hyperactive-plan --wells data/synthetic/wells.csv --coordinates data/synthetic/clusters.csv \
    --model models/arrive39 --start 2026-01-01 --horizon-years 10 --drilling-months 24 \
    --drilling-crews 3 --gtm-crews 2 --out plan.csv
```

`plan.csv` gets one row per crew task: well, cluster, crew, schedule, move
time and the well's discounted NPV. Useful flags: `--oil-cap` (annual
production ceiling), `--episodes N --exploration P` (stochastic rollouts,
keep the best by NPV — this is how the benchmark evaluates a model; a single
deterministic rollout is `--episodes 1` and the default), `--no-greedy` to
skip the baseline. `hyperactive-plan --help` lists everything.

Loading a model verifies its `manifest.json` against the artifact files by
sha256 and against the running code's feature encoding
(`hyperactive.env.FEATURE_SET`) — a mismatch raises `ModelManifestError`
rather than silently feeding features into the wrong input positions.

## Training a policy

```bash
python experiments/train.py --run-name demo --rl-episodes 300 \
    --crew-mix 2x1,3x2,5x5 --horizon-mix 5,10,25 --drilling-months-mix 12,24,60 \
    --oil-constraint-prob 0.5 --net-width 64 --weight-decay 0.0001 \
    --behavioral-cloning 0 --greedy-prefill 0
```

Artifacts land in `runs/demo/`: `episodes.csv` (one row per training
episode: NPV, greedy reference, win/loss, ε, plan size), `eval_{train,holdout}.csv`
(deterministic evaluation on held-out subsets), `summary.json`, and
`models/{model.zip,vec_normalize.pkl,manifest.json}` — loadable straight into
`hyperactive.inference.load_policy` or `hyperactive-plan --model runs/demo/models`.

The same run is recorded in MLflow (`./mlruns` by default, `mlflow ui` to
browse): the learning curve as a metric series, the evaluation aggregates, the
trained model, and the provenance needed to repeat the run —
see [`tracking.md`](tracking.md). With the same seed, inputs and library
versions the run reproduces bit for bit; `python experiments/rerun.py --last
--execute` verifies that and re-runs it.

300 episodes is a smoke test, not a trained model — `arrive39` was trained for
10,000. See [`method.md §6`](method.md#6-training-regime-experimentstrainpy) for what the regime
randomisation (`--crew-mix`, `--horizon-mix`, `--drilling-months-mix`,
`--oil-constraint-*`) is for, and `python experiments/train.py --help` for
every flag (`--use-cdqn 0` disables the C-DQN loss for an ablation against
plain masked Double DQN; `--behavioral-cloning 1 --greedy-prefill 1` enables
the warm start).

## Evaluating against exhaustive search

`experiments/evaluate.py` compares the greedy planner, a trained policy, and
(for small subsets) the best plan over *every* well-priority order:

```bash
python experiments/evaluate.py --model models/arrive39 --subsets 10 --well-count 6 \
    --drilling-crews 2 --gtm-crews 1 --horizon-years 10 --drilling-months 24 \
    --exhaustive-max-wells 8
```

Exhaustive search is `n!` plans per subset and is only run when
`well_count <= --exhaustive-max-wells` (default 8; `8! = 40320`, seconds per
subset — do not raise this much). Above that, only the greedy-vs-RL
comparison runs. Output columns: `rl_vs_greedy_percent` always;
`greedy_gap_percent` / `rl_gap_percent` (gap to the exhaustive optimum) when
exhaustive search ran for that subset.

## Reproducing the benchmark

`benchmark/case_10/` ships four well pools (48, 64, 150, 200 wells) used to
validate this repository's port of the RL-vs-greedy benchmark against
numbers previously produced by the full multi-service stack this code was
extracted from — see [`method.md`](method.md) for what's being measured and
`benchmark/run_benchmark.py`'s module docstring for the exact conventions
(readiness at 00:00 rather than 08:00, positional coordinate columns, a
240-month default economic horizon, per-fund planning start = earliest
well readiness).

Every fund's `cluster, x, y, z` coordinates are in metres, matching
`hyperactive.planning.DistanceTeamMovement`'s assumption. Funds `48` and `64`
originally shipped a different, incompatible schema (a local frame in
kilometres plus unrelated per-pad infrastructure fields); those two files were
rewritten to the same four-column, metre-scale format as `150`/`200` so that
every fund's crew movement is computed from real geometry rather than falling
back to a fixed distance for every cross-cluster move.

```bash
python benchmark/run_benchmark.py --model models/arrive39 --cases benchmark/case_10 \
    --funds 48 --horizons 12,18 --crews 2x1,3x2 --out runs/benchmark-48.xlsx
```

Grid cells write to `runs/` after every cell (an `.xlsx` with per-cell,
per-fund, per-crew and per-work-window sheets; `.csv` for a flat table), and to
MLflow as a suite run with one nested run per cell — including the crew
utilisation of both plans and the NPV of each RL rollout
(see [`tracking.md`](tracking.md)). `--dry-run` prints the discovered funds and
grid without computing anything; `--workers N` computes cells in parallel
processes.

Cells on funds `150` and `200` reproduce the reference numbers this port was
checked against bit-for-bit during extraction (their coordinates were never
touched). Cells on funds `48` and `64` now use real crew-movement geometry
instead of the fixed-distance fallback the original coordinate files produced,
so they no longer match the original multi-service stack's numbers exactly —
the drift is small (a few hundredths of a percentage point of uplift on
`48 / 2x1 / 12 months`; about 0.1 point on `48 / 2x2 / 12 months`, where the
plan crosses more clusters) but real, because crew moves now cost a few
hundredths of a day more or less than the flat 90-day floor. Cell
`48 / 2x1 / 12 months` reproduces `NPV_rl ≈ 6.04×10⁹`, `NPV_greedy ≈ 4.86×10⁹`,
uplift `≈ +24.29%`.
