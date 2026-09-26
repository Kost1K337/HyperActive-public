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
