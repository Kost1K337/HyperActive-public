# Experiment tracking

Every experiment script (`train.py`, `evaluate.py`, `run_benchmark.py`) and the
`hyperactive-plan` CLI itself record their run in [MLflow](https://mlflow.org).
The point is not a dashboard: a run holds
everything needed to repeat it later — the command, the seeds, the input data
*by content hash*, the model weights by content hash, the code revision and the
library versions — and [`rerun.py`](#reproducing-a-run) checks those against the
current checkout before re-running.

Tracking never breaks an experiment. A missing package, an unreachable server or
an error inside the tracker is logged and swallowed, and the computation
continues; `--no-tracking` disables it outright.

## Quick start

Tracking is on by default and writes to a local file store — nothing needs to be
running:

```bash
pip install -e ".[tracking]"

python experiments/train.py --run-name demo --rl-episodes 300
mlflow ui --backend-store-uri ./mlruns     # http://localhost:5000
```

Runs land in experiments named `hyperactive-training`, `hyperactive-evaluation`,
`hyperactive-benchmarks` and `hyperactive-plan`; override with `--experiment`.

## A tracking server

For runs that should outlive one laptop — and for benchmark grids computed with
`--workers`, where a file store is a bottleneck — start the server from
[`deploy/mlflow`](../deploy/mlflow) (MLflow + postgres, artifacts served by the
server):

```bash
docker compose -f deploy/mlflow/docker-compose.yml up -d --build
export HYPERACTIVE_MLFLOW_TRACKING_URI=http://localhost:8081
```

The run store is postgres rather than sqlite deliberately: sqlite holds one
writing transaction per file, and parallel benchmark cells run into
`database is locked`. The tracking database is created on startup if absent.

### The pre-loaded benchmark

A fresh server does not come up empty. Once it is healthy, a one-shot `seed`
service replays the released model's full benchmark grid — the suite run and
all 120 cells of `bc39` over `benchmark/case_10` — from
[`deploy/mlflow/seed/`](../deploy/mlflow/seed), so the first thing the UI shows
is how the released policy actually does, without anyone waiting hours for a
grid.

These are the recorded runs themselves, not a summary table: the same
parameters, metrics, tags, input digests and suite/cell nesting the benchmark
wrote. Because they carry the same provenance as any other run
(`run.command`, `env.*`, `dataset.*.sha256`, `model.*.sha256`), a seeded cell
can be verified and recomputed with `rerun.py` exactly like a run of your own:

```bash
export HYPERACTIVE_MLFLOW_TRACKING_URI=http://localhost:8081
python experiments/rerun.py --last --experiment hyperactive-benchmarks
```

That checks all eight fund tables and the model by content hash, and the
runtime against the recorded one. The grid was computed from a clean checkout
(`env.git_dirty` is `false`) immediately before the seed file itself was
written back into the repository, so the recorded commit is the one that
introduced the model rather than the commit you have checked out: expect
`rerun.py` to report that one difference and nothing else.

Seeding is idempotent — each run carries a `seed.id`, and a grid already
present is skipped — so restarting the stack never duplicates it. To start
without it, `docker compose ... up -d postgres mlflow` brings up the server
alone.

## Configuration

| Flag | Environment variable | Default |
|---|---|---|
| `--tracking-uri` | `HYPERACTIVE_MLFLOW_TRACKING_URI` | `file:./mlruns` |
| `--experiment` | `HYPERACTIVE_MLFLOW_EXPERIMENT` | per script (see above) |
| `--no-tracking` | `HYPERACTIVE_MLFLOW_ENABLED=false` | tracking enabled |

## What is recorded

**Provenance, in every run** (`RunTracker.log_environment`):

* `run.command` — the exact command line; the full `argv` is also in the
  `environment.json` artifact;
* `env.python`, `env.torch`, `env.stable_baselines3`, `env.gymnasium`,
  `env.numpy`, `env.pandas`, `env.pydantic`, `env.mlflow`, `env.platform`;
* `env.git_sha` and `env.git_dirty` (also as tags `code.git_sha` /
  `code.git_dirty`) — a commit identifies the code only if the tree was clean;
* `seed.*` — every seed the result depends on (training seed, pool split seed);
* `dataset.<name>.sha256` for every input file, and the same files registered as
  MLflow *inputs* with their digest. A file name does not identify a dataset —
  the same name holds different pools over time — so the hash is what matters;
* `model.*` — for runs that load or produce a model: the manifest fields, the
  sha256 of the weights and of the normalisation statistics, and the model
  directory as an artifact. A path does not identify weights: they are replaced
  in place under the same name.

**`experiments/train.py`** — one run per training:

| | |
|---|---|
| Parameters | every CLI argument, pool sizes, the train/hold-out split |
| Metric series (step = episode) | `npv_rl`, `npv_greedy`, `uplift_percent`, `plan_size`, `epsilon`, and the C-DQN diagnostics `loss`, `loss_dqn`, `loss_msbe`, `msbe_active_fraction`, `q_mean`, `q_abs_max`, `grad_norm` |
| Final metrics | `eval.train.*` and `eval.holdout.*` (mean/median/aggregate uplift, win rate, worst/best cell), episode count, timings, resampled degenerate subsets |
| Artifacts | `episodes.csv`, `eval_train.csv`, `eval_holdout.csv`, `summary.json`, `hot_start.json`, and the trained `model/` (weights, `vec_normalize.pkl`, `manifest.json`) |

**`benchmark/run_benchmark.py`** — a suite run with one nested run per grid cell:

| | |
|---|---|
| Suite parameters | model, suite, economics, grid size, episodes, exploration, economic horizon |
| Suite metrics | `uplift_median`, `uplift_mean`, `uplift_worst`, `uplift_best`, `win_rate`, `uplift_aggregate` (in money, not the mean of percentages), and the breakdowns `by_fund.*`, `by_crews.*`, `by_window.*` |
| Cell parameters | fund, well count, start date, crews, work window |
| Cell metrics | `rl_npv`, `greedy_npv`, `uplift_percent`, well counts, oil, timings; the plan metrics of **both** sides (`rl.*`, `greedy.*`): crew utilisation and busy/travel days, clusters touched, plan span, per-well NPV spread; and `rl_episode_npv` as a series over rollouts, which shows whether a cell was won by the policy or by one lucky exploration episode |
| Artifacts | the result table, `cells.json`, `economics.json`, the model |

**`experiments/evaluate.py`** — one run per evaluation: the per-subset series
(`rl_npv`, `greedy_npv`, `exhaustive_npv`, gaps), the `describe()` statistics of
each comparison column, and the result table.

**`hyperactive-plan`** — one run per invocation, same shape as a single
benchmark cell: `rl_npv`, `greedy_npv`, `uplift_percent`, well counts,
`rejected_wells`, the plan metrics of both sides (`rl.*`, `greedy.*`), and the
written `--out` schedule as an artifact when given.

## Reproducing a run

```bash
python experiments/rerun.py --run-id <id>            # inspect and verify
python experiments/rerun.py --last --execute         # verify, then re-run it
```

The script reads the run back and checks the three things that silently change a
result:

* **inputs** — locates every recorded file and compares the sha256 of its
  content, not its name;
* **code** — the current commit against the recorded one, and refuses to call a
  dirty-tree run reproducible;
* **libraries** — `torch`, `stable-baselines3`, `gymnasium`, `numpy` and the
  Python version, which decide the arithmetic.

Mismatches are listed and the exit code is non-zero; `--execute` runs the
recorded command anyway only with `--force`.

## What "reproducible" means here

With the same seeds, the same inputs and the same library versions on the same
machine, a training run is reproducible **bit for bit**: the episode table, the
policy weights and the evaluation summary come out identical. Seeding covers
`random`, `numpy`, `torch` and the environment's own generators, and
`torch.use_deterministic_algorithms(True, warn_only=True)` is set.

Across *different* machines, expect agreement to the last few digits rather than
exact equality: BLAS implementations differ between platforms, and a
near-tie between two candidate Q-values can resolve the other way. In a
cross-platform check of the benchmark, 5 of 6 cells matched exactly and the
sixth differed by 3·10⁻⁴ relative in RL NPV, with the same well count — the same
code on one machine reproduced each other's numbers exactly.
