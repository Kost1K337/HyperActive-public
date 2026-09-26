# Baseline fairness correction

A diagnostic pass over the eight real-pipeline benchmarks (Egg Model, SPE9,
SPE10, Volve, case_10, Norne, NLOG, Drogon) found two bugs that made the
published bc39-vs-greedy comparison invalid, and one limitation of the bc39
checkpoint. The two bugs are fixed and every benchmark was rerun with the
same parameters as before. The limitation is disclosed, not corrected. The
earlier numbers stay in `results_real_pipeline.csv` (rows `HISTORY` /
`CURRENT (US-Fx)`); the corrected rows are tagged `CORRECTED (bug-fix pass)`.

## 1. Greedy and PlanEnv ranked candidates by different rules (all benchmarks)

- `src/hyperactive/greedy.py` (`PlanBuilder._select_best_candidate`): the
  greedy baseline takes the candidate with the largest
  `NPV - drill_team_penalty` (`use_drill_team_penalty=True`, the default and
  what `hyperactive-plan` and `run_benchmark.py` used).
- `src/hyperactive/env/plan_env.py` (`PlanEnv._limit_candidates`): the
  environment that builds bc39's candidate set sorts candidates by NPV
  alone.

So "greedy" and "the environment's best-ranked action" were different
policies. Where candidate NPVs tie or the penalty reorders them, they take
different wells: on SPE9 (25 producers with one rate) the NPV-only greedy
scores 43.81e9 and the penalised greedy 58.97e9, purely from tie-breaking.

**Fix (reporting level; bc39, its checkpoint and its environment are
unchanged):**

- `hyperactive-plan --greedy-penalty {on,off,both}` (default `on`, the old
  behaviour). `both` runs both greedy variants in one invocation.
- `benchmark/run_benchmark.py` always computes both and writes
  `greedy_nopenalty_wells`, `greedy_nopenalty_npv`,
  `uplift_nopenalty_percent` beside the existing `greedy_*` /
  `uplift_percent` columns.
- `tests/test_env.py::TestGreedyRankingMatchesPlanBuilder`: always taking
  `PlanEnv`'s row 0 reproduces `PlanBuilder(use_drill_team_penalty=False)`
  exactly (same well order and NPV, with tied candidates).

The PRIMARY baseline is the NPV-only greedy: it is the ranking rule bc39
acts under, so "bc39 vs primary greedy" measures what the learned policy
adds over always taking the environment's top-ranked candidate. The
penalised greedy is kept as a SECONDARY baseline: it is the rule
`sections/method.tex` describes, and it is a legitimate, sometimes stronger,
heuristic. Both are reported; the primary number is not to be quoted
without the secondary one.

## 2. Volve `init_entry_date` predated the planning window (Volve only)

`benchmark/volve/build_hyperactive_input.py` used the wells' historical
completion dates (2007-2014) as `init_entry_date`; every run starts on
2026-01-01, so `ready_in_days` was about -6110 days for every candidate and
sat at the VecNormalize clip bound on 100% of candidate rows. Now all Volve
wells use `init_entry_date = 2026-01-01`, the convention of the other
adapters. Clipped rows on Volve dropped from 100% to 5.9%. Details:
`volve/README.md`.

## Corrected comparison

All runs: `--model models/bc39 --start 2026-01-01 --horizon-years 20
--drilling-months 12 --drilling-crews 2 --gtm-crews 1 --episodes 10
--exploration 0.15` (`hyperactive-plan --greedy-penalty both`); case_10 via
`run_benchmark.py --horizons 12 --crews 2x1 --planning-months 240`. The bc39
runs are deterministic (episode k seeded with k) and reproduced the earlier
bc39 NPVs exactly on every benchmark whose input did not change (all but
Volve); the penalised-greedy runs reproduced the earlier greedy NPVs exactly.

| benchmark | wells | bc39 NPV | greedy NPV only (PRIMARY) | greedy NPV - penalty (secondary) | bc39 vs primary | bc39 vs secondary | previously reported |
|---|---:|---:|---:|---:|---:|---:|---:|
| Egg Model | 12 | 1,722,483,028 (9) | 1,710,121,760 (10) | 1,529,277,304 (12) | **+0.72%** | +12.63% | +12.63% |
| SPE9 | 26 | 47,324,417,883 (13) | 43,808,870,706 (12) | 58,971,812,911 (16) | **+8.02%** | -19.75% | -19.75% |
| SPE10 | 5 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) | **0.00%** | 0.00% | 0.00% |
| Volve | 12 | 6,423,330,891 (12) | 6,429,779,953 (12) | 6,429,779,953 (12) | **-0.10%** | -0.10% | -0.76% |
| case_10 / 48 | 48 | 6,784,442,227 (10) | 4,860,649,333 (6) | 4,860,649,333 (6) | **+39.58%** | +39.58% | +39.58% |
| case_10 / 64 | 64 | 12,206,853,099 (10) | 10,603,736,532 (7) | 10,603,736,532 (7) | **+15.12%** | +15.12% | +15.12% |
| case_10 / 150 | 150 | 8,817,538,841 (9) | 8,155,481,707 (8) | 8,155,481,707 (8) | **+8.12%** | +8.12% | +8.12% |
| case_10 / 200 | 200 | 8,784,582,880 (10) | 6,950,302,884 (6) | 6,950,302,884 (6) | **+26.39%** | +26.39% | +26.39% |
| Norne | 36 | 401,689,185,673 (8) | 440,744,130,579 (8) | 440,744,130,579 (8) | **-8.86%** | -8.86% | -8.86% |
| NLOG | 16 | 10,136,820,303 (15) | 10,259,729,536 (15) | 10,259,729,536 (15) | **-1.20%** | -1.20% | -1.20% |
| Drogon | 6 | 296,650,035,339 (6) | 296,650,884,980 (6) | 296,650,884,980 (6) | **-0.0003%** | -0.0003% | -0.0003% |

Wells scheduled in parentheses.

**What changed.** Only Egg Model, SPE9 and Volve move; on the other five
benchmarks both greedy rules produce the same plan.

- Egg Model: the earlier +12.63% was mostly a weak baseline. Against the
  NPV-only greedy, bc39 gains +0.72%.
- SPE9: -19.75% becomes +8.02% against the NPV-only greedy, but the
  penalised greedy remains 24.6% above bc39 (58.97e9 vs 47.32e9). bc39
  improves on its own ranking rule and still loses clearly to a simple
  heuristic that breaks ties differently.
- Volve: the input fix moves bc39 from -0.76% to -0.10%; still a loss.

**Honest summary.** Against the primary baseline, bc39 wins on case_10 (all
four funds, +8.12% to +39.58%), SPE9 (+8.02%) and Egg Model (+0.72%, small);
ties SPE10 and Drogon (0.00%, -0.0003%); and loses on Volve (-0.10%), NLOG
(-1.20%) and Norne (-8.86%). The large wins are on case_10, bc39's own data
distribution. On the external benchmarks the picture is mixed-to-negative,
and on SPE9 the penalised greedy beats bc39 by a wide margin.

## 3. Disclosed limitation: feature-scale saturation (not corrected)

`src/hyperactive/env/features.py` feeds `oil_rate`, `liq_rate`, `cost_npv`,
`cash_flow`, `capex` and `travel_cost` as absolute values. bc39's
VecNormalize statistics were fit on its training distribution and clip at
+/-10 (`clip_obs=10.0`, `src/hyperactive/inference/policy.py`). Where a
benchmark's scale is far outside that distribution, the clipped features no
longer distinguish candidates.

Share of valid candidate rows with at least one feature at the clip bound
(row-0 rollout under the benchmark parameters above):

| benchmark | rows | clipped | features at the bound |
|---|---:|---:|---|
| Drogon | 21 | **100.0%** | liq_rate 100%, cost_npv 100%, cash_flow 100%, oil_rate 48% |
| Norne | 51 | **96.1%** | oil_rate, liq_rate, cost_npv, cash_flow 96% each; job_window_share 4% |
| SPE9 | 87 | **86.2%** | cost_npv 86%, cash_flow 86%, oil_rate 25%, job_window_share 5% |
| case_10 / 64 | 56 | 12.5% | job_window_share only |
| case_10 / 150 | 64 | 6.2% | job_window_share only |
| Volve (after fix) | 68 | 5.9% | length 6%, capex 3% (was 100%, ready_in_days) |
| case_10 / 200 | 44 | 2.3% | job_window_share only |
| case_10 / 48 | 47 | 2.1% | job_window_share only |
| Egg Model | 65 | 0.0% | -- |
| SPE10 | 15 | 0.0% | -- |
| NLOG | 99 | 0.0% | -- |

On Drogon, Norne and SPE9, bc39 is evaluated largely outside the input range
it was trained on. No retraining and no inference-time rescaling was applied:
either would change the model's behaviour after the fact without proper
training and validation. This belongs in the paper's limitations, together
with the fact that per-benchmark scale normalisation of the economic
features (or retraining on a wider distribution) is the fix, still to be
done and validated.
