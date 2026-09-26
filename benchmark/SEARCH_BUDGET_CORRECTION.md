# Search-budget fairness correction

Follow-up to [`BASELINE_FAIRNESS_CORRECTION.md`](BASELINE_FAIRNESS_CORRECTION.md).
That pass fixed the ranking-rule mismatch between the greedy baseline and
`PlanEnv`. This pass fixes a second, separate unfairness in the same
comparison: **the two sides had different search budgets.** Earlier rows in
`results_real_pipeline.csv` (`HISTORY`, `CURRENT (US-Fx)`,
`CORRECTED (bug-fix pass)`) are kept unchanged; the rows of this pass are
tagged `CORRECTED2 (search-budget fairness pass)`.

## The bug

Every bc39 number reported so far came from
`--episodes 10 --exploration 0.15`:

- `src/hyperactive/inference/policy.py`, `run_episode`: at every step, with
  probability 0.15 (per-episode `np.random.default_rng(seed)`), a uniformly
  random valid action is taken instead of the highest-Q one. The valid
  actions are `PlanEnv`'s candidate window: the top 8 candidates by NPV
  (`action_window = 8` in `models/bc39/manifest.json`).
- `plan_with_policy`: runs episodes `k = 0..9` with seed `k` and keeps the
  plan with the highest **true NPV** (`final_npv`), i.e. best-of-10 random
  restarts, selected by the objective itself. None of the 10 episodes is the
  pure deterministic policy (each has exploration 0.15).
- `hyperactive-plan` (`src/hyperactive/cli.py`) and
  `benchmark/run_benchmark.py` passed these two flags to `plan_with_policy`
  only. The greedy baseline (`PlanBuilder.compile`, both ranking variants)
  was a single deterministic pass with no randomness and no restarts.

So every "bc39 vs greedy" figure compared *the policy plus a 10-restart
random search judged by true NPV* against *the heuristic, run once*. Part
of each reported win could come from the search, not from the learned
scorer.

## The fix

`PlanBuilder` (`src/hyperactive/greedy.py`) gets the same search mechanism,
applied to the greedy ranking instead of the network:

- `compile(..., exploration=p, seed=k, window=w)`: at each step, with
  probability `p` (`np.random.default_rng(k)`), take a uniformly random
  candidate among the `w` best-ranked candidates (by the builder's own key:
  NPV, or NPV - drill_team_penalty) instead of the best one. `p = 0` is the
  old deterministic pass, bit for bit.
- `compile_best_of(wells, make_manager, ..., episodes, exploration, window)`:
  the greedy counterpart of `plan_with_policy` - episodes `0..N-1` seeded
  with `k`, best by total NPV, ties to the larger plan.
- `hyperactive-plan --greedy-matched-budget` and
  `run_benchmark.py --greedy-matched-budget`: both greedy variants get the
  policy's `--episodes`, `--exploration` and action window (8). Without the
  flag the greedy is one deterministic pass, as before.
- Tests: `tests/test_greedy.py::TestGreedySearchBudget` (zero budget equals
  the deterministic pass; window 1 at exploration 1.0 is still greedy;
  seeded reproducibility and best-by-NPV selection).

bc39, its checkpoint and `PlanEnv` are unchanged.

## Reported comparisons

- **PRIMARY - deterministic, zero search on either side.** bc39 with
  `--episodes 1 --exploration 0` (always the valid action with the highest
  Q-value) vs one deterministic greedy pass. This isolates the skill of the
  learned scorer.
- **SECONDARY - matched search budget.** bc39 best of 10 at exploration
  0.15 (the previously reported runs, unchanged) vs greedy best of 10 at
  exploration 0.15 over the same top-8 window, same seeds, same selection
  by true NPV. This asks whether the learned scorer beats an equal amount
  of random search around the heuristic.

Each is reported against both greedy rankings: NPV only (primary baseline,
the rule `PlanEnv` ranks bc39's candidates by) and NPV - drill_team_penalty
(secondary baseline).

Commands (all runs: `--model models/bc39 --start 2026-01-01
--horizon-years 20 --drilling-months 12 --drilling-crews 2 --gtm-crews 1`,
the same parameters as the CORRECTED pass):

```bash
# deterministic (primary)
.venv_real/bin/hyperactive-plan --wells benchmark/<b>/hyperactive_input/wells.xlsx \
  --coordinates benchmark/<b>/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 --drilling-months 12 \
  --drilling-crews 2 --gtm-crews 1 --no-tracking --greedy-penalty both \
  --episodes 1 --exploration 0 --out benchmark/<b>/plan_bc39_deterministic.csv
# matched budget (secondary)
.venv_real/bin/hyperactive-plan <same inputs and parameters> --no-tracking --greedy-penalty both \
  --episodes 10 --exploration 0.15 --greedy-matched-budget
# case_10: the CORRECTED-pass run_benchmark.py command (--funds 48,64,150,200 --horizons 12
# --crews 2x1 --planning-months 240 --workers 4 --no-tracking) with
#   --episodes 1 --exploration 0          --out benchmark/case_10/results_bc39_deterministic.xlsx
#   --episodes 10 --exploration 0.15 --greedy-matched-budget
#                                         --out benchmark/case_10/results_bc39_matched_budget.xlsx
```

Reproducibility check: the bc39 best-of-10 runs reproduced the CORRECTED
bc39 NPVs exactly on all 11 cells, and the deterministic greedy runs
reproduced the CORRECTED greedy NPVs exactly.

## Corrected comparison

NPV, wells scheduled in parentheses. Uplift = bc39 / greedy - 1.

### Primary: deterministic bc39 vs deterministic greedy (zero search budget)

| benchmark | wells | bc39 deterministic | greedy NPV only | greedy NPV - penalty | **vs NPV-only greedy** | vs penalised greedy |
|---|---:|---:|---:|---:|---:|---:|
| Egg Model | 12 | 1,482,494,720 (12) | 1,710,121,760 (10) | 1,529,277,304 (12) | **-13.31%** | -3.06% |
| SPE9 | 26 | 43,361,745,196 (12) | 43,808,870,706 (12) | 58,971,812,911 (16) | **-1.02%** | -26.47% |
| SPE10 | 5 | 1,734,414,147 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) | **-1.69%** | -1.69% |
| Volve | 12 | 6,390,103,682 (12) | 6,429,779,953 (12) | 6,429,779,953 (12) | **-0.62%** | -0.62% |
| case_10 / 48 | 48 | 5,715,132,492 (8) | 4,860,649,333 (6) | 4,860,649,333 (6) | **+17.58%** | +17.58% |
| case_10 / 64 | 64 | 12,318,920,597 (10) | 10,603,736,532 (7) | 10,603,736,532 (7) | **+16.18%** | +16.18% |
| case_10 / 150 | 150 | 7,237,860,235 (7) | 8,155,481,707 (8) | 8,155,481,707 (8) | **-11.25%** | -11.25% |
| case_10 / 200 | 200 | 8,147,946,592 (9) | 6,950,302,884 (6) | 6,950,302,884 (6) | **+17.23%** | +17.23% |
| NLOG | 16 | 9,671,363,420 (14) | 10,259,729,536 (15) | 10,259,729,536 (15) | **-5.73%** | -5.73% |
| Drogon | 6 | 296,650,035,339 (6) | 296,650,884,980 (6) | 296,650,884,980 (6) | **-0.0003%** | -0.0003% |
| Norne | 36 | 338,556,579,765 (6) | 440,744,130,579 (8) | 440,744,130,579 (8) | **-23.19%** | -23.19% |

### Secondary: bc39 best of 10 vs greedy best of 10 (matched search budget, exploration 0.15)

| benchmark | bc39 best of 10 | greedy NPV only, best of 10 | greedy NPV - penalty, best of 10 | **vs NPV-only greedy** | vs penalised greedy | CORRECTED pass (vs 1-pass NPV-only greedy) |
|---|---:|---:|---:|---:|---:|---:|
| Egg Model | 1,722,483,028 (9) | 1,710,121,760 (10) | 1,704,866,230 (10) | **+0.72%** | +1.03% | +0.72% |
| SPE9 | 47,324,417,883 (13) | 51,729,831,812 (14) | 58,971,812,911 (16) | **-8.52%** | -19.75% | +8.02% |
| SPE10 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) | **0.00%** | 0.00% | 0.00% |
| Volve | 6,423,330,891 (12) | 6,429,779,953 (12) | 6,429,779,953 (12) | **-0.10%** | -0.10% | -0.10% |
| case_10 / 48 | 6,784,442,227 (10) | 6,041,438,472 (8) | 6,041,438,472 (8) | **+12.30%** | +12.30% | +39.58% |
| case_10 / 64 | 12,206,853,099 (10) | 10,704,025,147 (7) | 10,704,025,147 (7) | **+14.04%** | +14.04% | +15.12% |
| case_10 / 150 | 8,817,538,841 (9) | 8,649,496,154 (9) | 8,649,496,154 (9) | **+1.94%** | +1.94% | +8.12% |
| case_10 / 200 | 8,784,582,880 (10) | 8,114,297,337 (9) | 8,114,297,337 (9) | **+8.26%** | +8.26% | +26.39% |
| NLOG | 10,136,820,303 (15) | 10,259,692,325 (15) | 10,259,692,325 (15) | **-1.20%** | -1.20% | -1.20% |
| Drogon | 296,650,035,339 (6) | 296,650,884,980 (6) | 296,650,884,980 (6) | **-0.0003%** | -0.0003% | -0.0003% |
| Norne | 401,689,185,673 (8) | 440,744,130,579 (8) | 440,744,130,579 (8) | **-8.86%** | -8.86% | -8.86% |

Note on the matched greedy: none of its 10 episodes is the pure
deterministic pass (the same holds for bc39), so best-of-10 greedy can end
marginally below the one-pass greedy - on NLOG it does (10,259,692,325 vs
10,259,729,536, -0.0004%). Everywhere else it equals or exceeds the
one-pass greedy.

## What changed vs the CORRECTED (bug-fix pass) numbers

- **Deterministic bc39 loses on every external benchmark** (7/7): Egg Model
  -13.31%, SPE9 -1.02%, SPE10 -1.69%, Volve -0.62%, NLOG -5.73%, Drogon
  -0.0003%, Norne -23.19% against the NPV-only greedy. The small external
  "wins" of the CORRECTED pass (Egg Model +0.72%, SPE9 +8.02%) and the SPE10
  tie came from the best-of-10 search, not from the scorer.
- On case_10 (bc39's own data distribution) the deterministic policy wins 3
  of 4 funds (+17.58%, +16.18%, +17.23%) and loses fund 150 (-11.25%).
- With a matched budget, SPE9 flips from +8.02% to -8.52%: ten random
  restarts around the NPV-only greedy find 51.73e9, more than bc39's 47.32e9.
  The penalised greedy stays far ahead (58.97e9, -19.75%).
- case_10 wins shrink under a matched budget: +39.58/+15.12/+8.12/+26.39%
  become +12.30/+14.04/+1.94/+8.26%. bc39 still wins all four, i.e. on its
  training distribution the learned scorer guides search better than random
  perturbation of the greedy does.
- Egg Model stays +0.72% under a matched budget (the NPV-only greedy gains
  nothing from perturbation; the penalised greedy gains, to 1.705e9, and bc39
  leads it by +1.03%). But deterministically bc39 is -13.31%, so this gain is
  a property of policy-guided search, not of the policy's own choices.
- Volve, NLOG, Drogon, Norne: the matched-budget figures equal the CORRECTED
  ones (random restarts do not improve the greedy there); bc39 still loses.

## Honest headline

With zero search on either side, bc39 **does not beat the NPV-only greedy
on any of the seven external benchmarks** (Egg Model, SPE9, SPE10, Volve,
NLOG, Drogon, Norne); it wins 3 of 4 case_10 funds, its own training
distribution, by +16% to +18% and loses the fourth by -11%. With an equal
search budget, bc39 wins all four case_10 funds (+1.9% to +14.0%), is
marginally ahead on Egg Model (+0.72%), ties SPE10, and loses SPE9 (-8.52%),
Volve, NLOG, Drogon and Norne.

This is an in-distribution vs out-of-distribution result: the learned
scorer carries real signal on data like its training set and does not
transfer to the external fields, where it is also partly evaluated on
saturated inputs (see the feature-scale limitation in
`BASELINE_FAIRNESS_CORRECTION.md`, section 3: Drogon 100%, Norne 96%, SPE9
86% of candidate rows clipped). Earlier claims of external wins rested on
unequal search budgets and should not be repeated. Paper figures must state
the search budget of both sides; the deterministic comparison is the one
that measures the policy.
