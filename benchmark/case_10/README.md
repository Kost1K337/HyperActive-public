# case_10 benchmark (HyperActive's own bundled/native dataset)

## What this is

`benchmark/case_10/` ships four synthetic well pools (48, 64, 150 and 200
wells, in the `48/`, `64/`, `150/` and `200/` subdirectories) that were
extracted with the rest of this repository from the multi-service stack
HyperActive was built for -- see [`docs/method.md`](../../docs/method.md) and
[`docs/reproducing.md`](../../docs/reproducing.md#reproducing-the-benchmark).
Unlike the Egg Model, SPE9, SPE10 and Volve benchmarks (which are external,
open-industry reservoir datasets adapted into HyperActive's schema by this
project), `case_10` is HyperActive's **own native dataset**: its wells,
coordinates and calendar were already produced in HyperActive's own
`wells.xlsx` / `coordinates.xlsx` format, and `benchmark/run_benchmark.py`
(the project's own purpose-built harness) was written specifically to run
against it.

It is included in the paper's benchmark comparison table alongside the
external benchmarks specifically so the comparison is not limited to
externally sourced data -- HyperActive's own bundled dataset is a real
comparison point in its own right, not a placeholder.

## Real pipeline run

Environment: existing `.venv_real` (repository root), same environment used
for every other benchmark in this repo.

Command actually run (real subprocess, not simulated):

```bash
.venv_real/bin/python benchmark/run_benchmark.py \
  --model models/bc39 --cases benchmark/case_10 \
  --funds 48,64,150,200 --horizons 12 --crews 2x1 \
  --planning-months 240 --episodes 10 --exploration 0.15 \
  --workers 4 --no-tracking \
  --out benchmark/case_10/results_bc39.xlsx
```

This mirrors the exact per-cell parameters used for the Egg Model and SPE9
real-pipeline runs (US-F2/US-F4): 12-month drilling work window, 240-month
(20-year) economic horizon, 2 drilling crews x 1 GTM crew, 10 RL rollouts at
exploration 0.15, best-of-10 kept. One grid cell was run per fund size (a
4-cell grid, not the harness's much larger 120-cell default grid of 6 work
windows x 5 crew configs x 4 funds), since only a single, representative
comparison point per fund size was needed here.

Real wall-clock: **6m50.6s** for all 4 funds combined (4 parallel worker
processes, `--workers 4`).

**Real console output (verbatim):**

```
model bc39 | funds 4 | windows 1 | crews 1 | cells 4
  150: 150 rows, start 2026-07-01, wells_150.xlsx + coordinates_150.xlsx
  200: 200 rows, start 2027-03-01, wells_200.xlsx + coordinates_200.xlsx
  48: 48 rows, start 2026-07-09, wells_native.xlsx + coordinates_native.xlsx
  64: 64 rows, start 2026-07-01, demo_64.xlsx + demo_coordinates.xlsx
[1/4] 48 2x1 12m: RL 10 / greedy 6 wells, uplift 39.58% | 3.6 min
[2/4] 200 2x1 12m: RL 10 / greedy 6 wells, uplift 26.39% | 4.2 min
[3/4] 64 2x1 12m: RL 10 / greedy 7 wells, uplift 15.12% | 4.7 min
[4/4] 150 2x1 12m: RL 9 / greedy 8 wells, uplift 8.12% | 6.6 min
median uplift 20.75% | mean 22.30% | win rate 1.000 | cells 4
results: benchmark/case_10/results_bc39.xlsx
```

## Real results (from `results_bc39.xlsx`, `cells` sheet)

| fund | wells | rl (bc39) wells scheduled | greedy wells scheduled | bc39 NPV | greedy NPV | uplift vs. greedy |
|---:|---:|---:|---:|---:|---:|---:|
| 48  | 48  | 10 | 6 | 6,784,442,227  | 4,860,649,333  | **+39.58%** |
| 64  | 64  | 10 | 7 | 12,206,853,099 | 10,603,736,532 | **+15.12%** |
| 150 | 150 | 9  | 8 | 8,817,538,841  | 8,155,481,707  | **+8.12%**  |
| 200 | 200 | 10 | 6 | 8,784,582,880  | 6,950,302,884  | **+26.39%** |

Every number above is copied verbatim from the `cells` sheet of
`benchmark/case_10/results_bc39.xlsx`, produced by the real command above
(not invented or estimated). bc39 beats the greedy `PlanBuilder` baseline on
all four fund sizes at this work window/crew configuration (median uplift
20.75%, mean 22.30%, win rate 4/4). A machine-readable summary is in
[`results_summary.csv`](results_summary.csv).

This is a single grid cell per fund (one work window, one crew
configuration), not an exhaustive sweep -- `results_bc39.xlsx` also has
`by_fund`, `by_crews`, `by_work_window` and `uplift_grid` sheets that would
be populated with medians/win-rates across a larger grid if `--horizons`
and `--crews` were widened; that fuller grid was not run here since a single
representative comparison point per fund was sufficient for this story.

## Baseline fairness correction (bug-fix pass)

The numbers above (US-F5) are kept as recorded. Overview of the whole pass:
[`../BASELINE_FAIRNESS_CORRECTION.md`](../BASELINE_FAIRNESS_CORRECTION.md).

**What was wrong.** The greedy baseline `PlanBuilder` ranks candidates by
`NPV - drill_team_penalty`; `PlanEnv`, which builds the candidate set bc39
acts on, ranks by NPV alone, so the two were not compared under the same
ranking rule. **Fix.** `run_benchmark.py` now also runs
`PlanBuilder(use_drill_team_penalty=False)` in every cell and writes
`greedy_nopenalty_wells`, `greedy_nopenalty_npv` and
`uplift_nopenalty_percent` next to the existing (penalised) `greedy_*` and
`uplift_percent` columns. The NPV-only greedy is the PRIMARY baseline; the
penalised one is kept as secondary.

Rerun with the identical command above (wall-clock 3.5 min, 4 workers);
`results_bc39.xlsx` now holds both baselines:

| fund | bc39 wells / NPV | greedy NPV only (PRIMARY) wells / NPV | greedy NPV - penalty wells / NPV | bc39 vs primary | bc39 vs penalised |
|---:|---:|---:|---:|---:|---:|
| 48  | 10 / 6,784,442,227  | 6 / 4,860,649,333  | 6 / 4,860,649,333  | **+39.58%** | +39.58% |
| 64  | 10 / 12,206,853,099 | 7 / 10,603,736,532 | 7 / 10,603,736,532 | **+15.12%** | +15.12% |
| 150 | 9 / 8,817,538,841   | 8 / 8,155,481,707  | 8 / 8,155,481,707  | **+8.12%**  | +8.12%  |
| 200 | 10 / 8,784,582,880  | 6 / 6,950,302,884  | 6 / 6,950,302,884  | **+26.39%** | +26.39% |

No change on case_10: both greedy rules give the same plan in every cell,
and the bc39 runs reproduce the earlier NPVs exactly.

Feature-scale check (disclosed limitation of bc39 on out-of-range inputs,
see the overview): candidate rows with any feature at the VecNormalize
+/-10 clip bound, row-0 rollout: fund 48 2.1%, 64 12.5%, 150 6.2%, 200
2.3%, all on `job_window_share` only; none of the absolute-scale economic
features saturate. This is bc39's own data distribution, so the effect is
minor here.

## Search-budget fairness correction

The bc39 numbers above are best of 10 rollouts at exploration 0.15, picked
by true NPV; the greedy baselines were one deterministic pass. That unequal
search budget is corrected in
[`../SEARCH_BUDGET_CORRECTION.md`](../SEARCH_BUDGET_CORRECTION.md) (rows
`CORRECTED2 (search-budget fairness pass)` in
[`../results_real_pipeline.csv`](../results_real_pipeline.csv)). Two
comparisons replace the single one: **deterministic** (bc39 1 episode,
exploration 0, vs one greedy pass; the primary, it measures the policy) and
**matched budget** (both sides best of 10 at exploration 0.15, the greedy
taking a random candidate among the top 8 with the same probability and
seeds; `--greedy-matched-budget`). Same parameters as before.

| | bc39 | greedy NPV only | greedy NPV - penalty | bc39 vs NPV-only greedy (primary) | bc39 vs penalised greedy |
|---|---:|---:|---:|---:|---:|
| fund 48, deterministic | 5,715,132,492 (8) | 4,860,649,333 (6) | 4,860,649,333 (6) | **+17.58%** | +17.58% |
| fund 48, best of 10 | 6,784,442,227 (10) | 6,041,438,472 (8) | 6,041,438,472 (8) | **+12.30%** | +12.30% |
| fund 64, deterministic | 12,318,920,597 (10) | 10,603,736,532 (7) | 10,603,736,532 (7) | **+16.18%** | +16.18% |
| fund 64, best of 10 | 12,206,853,099 (10) | 10,704,025,147 (7) | 10,704,025,147 (7) | **+14.04%** | +14.04% |
| fund 150, deterministic | 7,237,860,235 (7) | 8,155,481,707 (8) | 8,155,481,707 (8) | **-11.25%** | -11.25% |
| fund 150, best of 10 | 8,817,538,841 (9) | 8,649,496,154 (9) | 8,649,496,154 (9) | **+1.94%** | +1.94% |
| fund 200, deterministic | 8,147,946,592 (9) | 6,950,302,884 (6) | 6,950,302,884 (6) | **+17.23%** | +17.23% |
| fund 200, best of 10 | 8,784,582,880 (10) | 8,114,297,337 (9) | 8,114,297,337 (9) | **+8.26%** | +8.26% |

NPV, wells scheduled in parentheses.

case_10 is bc39's own training distribution. Deterministically bc39 wins funds 48, 64 and 200 (+17.58%, +16.18%, +17.23%) and loses fund 150 (-11.25%). With a matched budget it wins all four, but by much less than reported before: +39.58/+15.12/+8.12/+26.39% become +12.30/+14.04/+1.94/+8.26%. Outputs: `results_bc39_deterministic.xlsx`, `results_bc39_matched_budget.xlsx`.
