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

## Settings

Economics, the readiness convention and the calendar live in one place,
[`hyperactive.scenario`](../src/hyperactive/scenario.py), and every entry
point uses them — `hyperactive-plan`, `experiments/train.py`,
`experiments/evaluate.py` and the benchmark alike. A plan is therefore priced
and scheduled the same way wherever it is built: `hyperactive-plan` run with
a benchmark cell's fund, crews and work window reproduces that cell's numbers
exactly. The conventions are:

* the economics of `hyperactive.scenario.ECONOMICS` (oil price, opex per
  tonne of oil and water, fixed yearly costs, drilling cost per metre by well
  type, discount rate); a well type without a price is an error, never free;
* readiness at the table's timestamp (00:00 for a plain date);
* planning starts at midnight of the earliest readiness in the pool unless a
  start date is given;
* the economic horizon counts 365.25-day years, the work window 30.4-day months.

`experiments/train.py --settings file.json` and `benchmark/run_benchmark.py
--settings file.json` override `economics`, `readiness_hour` and
`days_per_year` (see `hyperactive.scenario.load_settings`). Every trained model
records the settings it was trained with in its manifest, because a policy's
Q-values are only meaningful in the economics its rewards were priced in.

The released `bc39` was trained on an earlier configuration of the project,
kept in [`models/bc39/training_settings.json`](../models/bc39/training_settings.json):
water at 48.6 rather than 420 per tonne, no price for plain `ННС` wells,
readiness at 08:00 and a 365-day year. Its training command in the manifest
passes that file, so the regime it describes is the one the model actually
learned in; everything it is *evaluated* on — the benchmark included — uses
the project settings above.

## Plan with the released model

`models/bc39/` is a masked-C-DQN policy trained on an 8-candidate action
window over 16-well subsets, starting from a **behavioural-cloning warm
start**: before Q-learning, the network is pretrained on 30 subsets to
reproduce the greedy planner's choice (cross-entropy over masked Q-values,
early stopping with patience 16, mean validation accuracy 0.86), and the
exploration schedule then starts at ε=0.3 rather than 1.0 so the cloned
policy is actually used instead of being overridden by near-random
exploration — see
[`method.md §5`](method.md#5-learning-algorithm-hyperactivemodelsmasked_cdqnmaskedcdqn) for why.
Greedy replay prefill is off. See `models/bc39/manifest.json` for the exact
training distribution, cloning hyperparameters and command. Build a plan and
compare it with the greedy baseline:

```bash
hyperactive-plan --wells data/synthetic/wells.csv --coordinates data/synthetic/clusters.csv \
    --model models/bc39 --horizon-years 10 --drilling-months 24 \
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
    --behavioral-cloning 1 --greedy-prefill 0
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

300 episodes is a smoke test, not a trained model — `bc39` was trained for
10,000. See [`method.md §6`](method.md#6-training-regime-experimentstrainpy) for what the regime
randomisation (`--crew-mix`, `--horizon-mix`, `--drilling-months-mix`,
`--oil-constraint-*`) is for, and `python experiments/train.py --help` for
every flag (`--use-cdqn 0` disables the C-DQN loss for an ablation against
plain masked Double DQN; `--behavioral-cloning 0` turns off the cloning warm
start the released model relies on, `--bc-patience` controls its early
stopping, and `--greedy-prefill 1` additionally seeds the replay buffer with
greedy episodes).

## Evaluating against exhaustive search

`experiments/evaluate.py` compares the greedy planner, a trained policy, and
(for small subsets) the best plan over *every* well-priority order:

```bash
python experiments/evaluate.py --model models/bc39 --subsets 10 --well-count 6 \
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
extracted from — see [`method.md`](method.md) for what's being measured. The
benchmark runs on the project [settings](#settings); on top of them it reads
coordinates by position (`cluster, x, y, z`) and uses a 240-month economic
horizon by default (see `benchmark/run_benchmark.py`'s module docstring).

Every fund's `cluster, x, y, z` coordinates are in metres, matching
`hyperactive.planning.DistanceTeamMovement`'s assumption, so crew movement is
computed from real geometry on all four pools.

```bash
python benchmark/run_benchmark.py --model models/bc39 --cases benchmark/case_10 \
    --funds 48 --horizons 12,18 --crews 2x1,3x2 --out runs/benchmark-48.xlsx
```

Grid cells write to `runs/` after every cell (an `.xlsx` with per-cell,
per-fund, per-crew and per-work-window sheets; `.csv` for a flat table), and to
MLflow as a suite run with one nested run per cell — including the crew
utilisation of both plans and the NPV of each RL rollout
(see [`tracking.md`](tracking.md)). `--dry-run` prints the discovered funds and
grid without computing anything; `--workers N` computes cells in parallel
processes.

What was validated during extraction is the *planner*, not any one policy: the
greedy plan reproduces the multi-service stack's NPV bit for bit on funds `150`
and `200`, in all 30 cells per fund. The RL side is a best-of-`--episodes`
sample rather than a single deterministic pass, so a cell reproduces exactly
only when the same rollouts are drawn; with the same code, inputs and seeds on
one machine it does, and `experiments/rerun.py` checks those preconditions.

The released `bc39` policy's own numbers are not repeated here; the full grid
it produced (120 cells) ships as recorded MLflow runs and is loaded into the
tracking server on first start, where every cell keeps the parameters, plan
metrics and provenance needed to check it — see
[`tracking.md`](tracking.md#the-pre-loaded-benchmark).
