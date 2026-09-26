# bc39v2: retraining with pool-relative features and singleton-cluster sampling

Follow-up to [`SEARCH_BUDGET_CORRECTION.md`](SEARCH_BUDGET_CORRECTION.md).
That pass made the bc39-vs-greedy comparison fair and found that bc39 beats
the greedy baseline only on `case_10`. This pass fixes two defects in the
model's own training setup, retrains a new checkpoint (`models/bc39v2/`),
and re-runs the same 8-benchmark evaluation zero-shot. `models/bc39/` is
unchanged and still reproduces its published numbers bit for bit. Rows of
this pass in `results_real_pipeline.csv` are tagged
`BC39V2 (retrain with fixed features + singleton-cluster sampling)`.

## 1. The two defects

### Defect 1: rate and money features in absolute units

At commit `cecee82`, `src/hyperactive/env/features.py:195-204`
(`_relative_embedding`) put `oil_rate`, `liq_rate`, `cost_npv`
(`well_context.cost`), `cash_flow`, `capex` and `travel_cost` into the
observation in t/day and money. `VecNormalize` then standardises them with
statistics fit on the training pool and clips at +/-10. On fields with a
different scale most rows sat at the clip: Drogon 100%, Norne 96.1%, SPE9
86.2% of candidate rows (`BASELINE_FAIRNESS_CORRECTION.md`, section 3). The
network could not tell those candidates apart on these features.

### Defect 2: singleton clusters never sampled

At commit `cecee82`, `experiments/train.py:216-222` (`build_cluster_index`)
kept only clusters with `len(ws) >= 2` for `--sampling clusters`, the mode
bc39 was trained with. SPE10 (5/5 clusters are singletons), Drogon (6/6) and
Norne (25/30) have a topology that training excluded by construction.

## 2. What changed (commit `d900a69`)

- `src/hyperactive/env/features.py:246` `_pool_relative`: each of the six
  values is divided by its largest magnitude among the current candidate
  window (`self.candidates`, at most 8 rows), so it lies in [-1, 1] on any
  field. It follows the convention of `value_density_norm` (normalise
  within the window, fall back to the row itself) but divides by the maximum
  magnitude instead of min-max, so that the sign of NPV and cash flow is
  kept and zero stays zero. Width (39), names and column order are
  unchanged; `FEATURE_SET` is still `"gen39"`.
- Keeping bc39 reproducible: both encodings have the same 39 columns, so
  the manifest now records which one a model expects. `experiments/train.py`
  writes `"feature_scaling": "pool_relative"` (line 532);
  `hyperactive.inference.run_episode` sets `env.pool_relative_scale` from
  it (`src/hyperactive/inference/policy.py:137`). A manifest without the key
  (bc39) gets the old absolute encoding (`features.py:233`
  `_absolute_scale`). Without this, bc39 would silently receive
  pool-relative inputs, which `load_policy`'s docstring warns about for
  same-width feature sets. Verified: bc39 deterministic on SPE10
  (1,734,414,147) and Volve (6,390,103,682) matches the CORRECTED2 rows
  exactly, and `tests/test_golden.py` (bc39 golden plan) passes.
- `experiments/train.py:216-225` `build_cluster_index`: the `len(ws) >= 2`
  filter is removed; every cluster is sampled. `pick_by_clusters` needed
  no change.
- `docs/method.md`, section 3: one sentence on the new scaling.
- Tests: `tests/test_env.py::TestPoolRelativeScale` (values in [-1, 1];
  invariance to scaling all rates by 50x; the absolute encoding is still
  produced when the flag is off). Suite: 85 passed (82 before + 3 new).

### Limitation of the Defect 2 fix on this training pool

**No public pool in this repository has a single-well cluster.** The
training pool `benchmark/case_10/200/wells_200.xlsx` has 30 clusters of 4-8
wells. After the 160/40 train/hold-out split (seed 8), the 160 training
wells form 30 clusters of 3-8 wells: zero singletons. The other case_10
pools (48, 64, 150) and `data/synthetic` also have none. So the Defect 2
fix is correct but **a no-op for bc39v2**: its training never saw a
singleton cluster either. Any change on SPE10, Drogon or Norne comes from
Defect 1 and from the different training pool, not from singleton
sampling. Exercising the fix needs a training pool with singleton clusters.

## 3. Training run

```bash
.venv_real/bin/python experiments/train.py --run-name bc39v2 \
  --wells-file benchmark/case_10/200/wells_200.xlsx \
  --coordinates-file benchmark/case_10/200/coordinates_200.xlsx \
  --settings models/bc39/training_settings.json \
  --seed 8 --well-count 16 --action-window 8 --rl-episodes 4000 \
  --sampling clusters --net-width 64 --weight-decay 0.0001 \
  --behavioral-cloning 1 --greedy-prefill 0 --hot-start-subsets 30 \
  --bc-patience 16 --greedy-every 50 \
  --crew-mix 2x1,2x1,2x1,2x5,3x2,3x4,4x1,4x3,5x2,5x5 \
  --horizon-mix 5,8,10,15,20,25 --drilling-months-mix 12,12,12,18,18,24,36,48,60 \
  --oil-constraint-prob 0.5 --oil-constraint-share 0.08:0.35 \
  --save-interval 500 --no-tracking
```

Every flag is identical to bc39's `manifest.json` command except:

| | bc39 | bc39v2 | why |
|---|---|---|---|
| training pool | non-public pool | public `case_10/200` (200 wells, 160 train / 40 hold-out) | the largest public pool |
| `--rl-episodes` | 10,000 (160,000 timesteps) | **4,000 (64,000 timesteps, 40%)** | time box, see below |
| feature scaling | absolute | pool-relative | Defect 1 |
| singleton clusters | excluded | included (none exist in the pool) | Defect 2 |

Scaling-down rationale: a 100-episode smoke run measured about 16
timesteps/s on this machine (4 CPU cores, shared with other work), so
160,000 timesteps would take about 2.8 h of RL alone. 64,000 timesteps was
chosen to fit the ~60-90 min time box. The exploration schedule is defined
as a fraction (0.8) of the total, so it has the same shape as bc39's,
compressed.

Wall clock (`/usr/bin/time`): **1:35:12** total, 2026-09-26 18:09:11Z to
19:44:23Z. Warm start (behavioural cloning on 30 subsets) 1,144 s; RL
4,524 s (75.4 min); in-training evaluation the rest.

**Convergence status:** the run **completed its full 64,000-timestep
schedule** (10,494 episodes; epsilon reached its final 0.05 around episode
8,500). It was not cut short. But it had 40% of bc39's training budget, and
the in-training learning curve is noisy (500-episode windows against greedy
references every 50th episode: mean uplift swings between -12% and +193%,
10 references per window), so there is no evidence of a plateau. Treat
bc39v2 as a budget-limited checkpoint, not a converged one. Warm-start
behavioural-cloning validation accuracy: mean 0.89 over the 30 subsets
(bc39: 0.858).

In-training evaluation (`runs/bc39v2/summary.json`, 30 random 16-well
subsets each, vs the default 2x2 greedy): train pool aggregate uplift
+2.60%, win rate 46.7%; hold-out pool aggregate -1.02%, win rate 36.7%.

Artifacts: `models/bc39v2/{model.zip, vec_normalize.pkl, manifest.json,
training_settings.json, training_summary.json}`. Training economics are
bc39's `training_settings.json`. Evaluation uses the same default economics
as every bc39 evaluation. `runs/` is git-ignored.

## 4. Input saturation after the fix

Share of valid candidate rows with at least one feature at the `VecNormalize`
clip (row-0 rollout, same method and parameters as
`BASELINE_FAIRNESS_CORRECTION.md` section 3; the method reproduces bc39's
published table exactly):

| benchmark | rows (bc39) | bc39 clipped | rows (bc39v2) | bc39v2 clipped | bc39v2 features at the bound |
|---|---:|---:|---:|---:|---|
| Drogon | 21 | **100.0%** | 21 | 9.5% | oil_rate, cost_npv, cash_flow, decline_year 10% |
| Norne | 51 | **96.1%** | 51 | 5.9% | job_window_share 4%; oil_rate, cash_flow, cost_npv 2% |
| SPE9 | 87 | **86.2%** | 87 | 4.6% | job_window_share 5% |
| Egg Model | 65 | 0.0% | 65 | 9.2% | oil_rate, cost_npv, cash_flow, decline_year 9% |
| SPE10 | 15 | 0.0% | 15 | 6.7% | oil_rate, cost_npv, cash_flow, decline_year 7% |
| Volve | 68 | 5.9% | 68 | 7.4% | oil_rate, cost_npv, cash_flow, decline_year 4% |
| NLOG | 99 | 0.0% | 99 | 15.2% | has_ready_date 15% |
| case_10 / 48 | 47 | 2.1% | 47 | 2.1% | job_window_share |
| case_10 / 64 | 56 | 12.5% | 56 | 12.5% | job_window_share |
| case_10 / 150 | 64 | 6.2% | 64 | 7.8% | job_window_share |
| case_10 / 200 | 44 | 2.3% | 44 | 2.3% | job_window_share |

Defect 1 is fixed where it was severe (Drogon, Norne, SPE9 fall from
86-100% to 5-10%). There is a new, smaller residual: bc39v2's training
pool gives pool-relative rates and NPVs a narrow range, so candidates with
negative or near-zero values relative to the window (and `decline_year`,
`has_ready_date`, which were never scale-normalised) can reach the clip on
some fields. This is a training-distribution coverage issue, not a scale
issue.

## 5. Before/after comparison (zero-shot, all 8 benchmarks)

The protocol, commands and parameters are exactly those of
`SEARCH_BUDGET_CORRECTION.md`, with `--model models/bc39v2`. bc39v2 was not
fine-tuned on any benchmark. All greedy numbers reproduced the CORRECTED2
greedy numbers exactly (deterministic and matched budget, both rankings).
NPV, wells scheduled in parentheses. Uplift = policy / greedy - 1.

**Caveat on case_10:** bc39v2 was trained on 160 wells of `case_10/200`,
so the `case_10 / 200` cell is in-sample for bc39v2. The 48/64/150 pools
share no well names with the 200 pool but come from the same generator.
bc39 was trained on a non-public pool, so for bc39 no case_10 cell is
in-sample.

### Primary: deterministic (1 episode, exploration 0) vs one deterministic greedy pass

| benchmark | wells | greedy NPV only | greedy NPV - penalty | bc39 | bc39v2 | **bc39 vs NPV-only** | **bc39v2 vs NPV-only** | bc39 vs penalised | bc39v2 vs penalised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Egg Model | 12 | 1,710,121,760 (10) | 1,529,277,304 (12) | 1,482,494,720 (12) | 1,622,994,861 (11) | -13.31% | **-5.09%** | -3.06% | +6.13% |
| SPE9 | 26 | 43,808,870,706 (12) | 58,971,812,911 (16) | 43,361,745,196 (12) | 51,423,206,404 (14) | -1.02% | **+17.38%** | -26.47% | -12.80% |
| SPE10 | 5 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,734,414,147 (5) | 1,764,158,272 (5) | -1.69% | **0.00%** | -1.69% | 0.00% |
| Volve | 12 | 6,429,779,953 (12) | 6,429,779,953 (12) | 6,390,103,682 (12) | 6,423,241,676 (12) | -0.62% | **-0.10%** | -0.62% | -0.10% |
| case_10 / 48 | 48 | 4,860,649,333 (6) | 4,860,649,333 (6) | 5,715,132,492 (8) | 6,602,539,535 (9) | +17.58% | **+35.84%** | +17.58% | +35.84% |
| case_10 / 64 | 64 | 10,603,736,532 (7) | 10,603,736,532 (7) | 12,318,920,597 (10) | 9,574,742,217 (7) | +16.18% | **-9.70%** | +16.18% | -9.70% |
| case_10 / 150 | 150 | 8,155,481,707 (8) | 8,155,481,707 (8) | 7,237,860,235 (7) | 8,094,592,021 (8) | -11.25% | **-0.75%** | -11.25% | -0.75% |
| case_10 / 200 (in-sample for v2) | 200 | 6,950,302,884 (6) | 6,950,302,884 (6) | 8,147,946,592 (9) | 6,501,407,574 (7) | +17.23% | **-6.46%** | +17.23% | -6.46% |
| NLOG | 16 | 10,259,729,536 (15) | 10,259,729,536 (15) | 9,671,363,420 (14) | 10,259,283,839 (15) | -5.73% | **-0.004%** | -5.73% | -0.004% |
| Drogon | 6 | 296,650,884,980 (6) | 296,650,884,980 (6) | 296,650,035,339 (6) | 296,650,884,980 (6) | -0.0003% | **0.00%** | -0.0003% | 0.00% |
| Norne | 36 | 440,744,130,579 (8) | 440,744,130,579 (8) | 338,556,579,765 (6) | 380,773,599,660 (8) | -23.19% | **-13.61%** | -23.19% | -13.61% |

### Secondary: best of 10 at exploration 0.15, both sides (matched search budget)

| benchmark | greedy NPV only, best of 10 | greedy NPV - penalty, best of 10 | bc39 best of 10 | bc39v2 best of 10 | **bc39 vs NPV-only** | **bc39v2 vs NPV-only** | bc39 vs penalised | bc39v2 vs penalised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Egg Model | 1,710,121,760 (10) | 1,704,866,230 (10) | 1,722,483,028 (9) | 1,703,855,913 (10) | +0.72% | **-0.37%** | +1.03% | -0.06% |
| SPE9 | 51,729,831,812 (14) | 58,971,812,911 (16) | 47,324,417,883 (13) | 51,729,831,812 (14) | -8.52% | **0.00%** | -19.75% | -12.28% |
| SPE10 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) | 0.00% | **0.00%** | 0.00% | 0.00% |
| Volve | 6,429,779,953 (12) | 6,429,779,953 (12) | 6,423,330,891 (12) | 6,426,361,214 (12) | -0.10% | **-0.05%** | -0.10% | -0.05% |
| case_10 / 48 | 6,041,438,472 (8) | 6,041,438,472 (8) | 6,784,442,227 (10) | 7,201,770,445 (10) | +12.30% | **+19.21%** | +12.30% | +19.21% |
| case_10 / 64 | 10,704,025,147 (7) | 10,704,025,147 (7) | 12,206,853,099 (10) | 11,778,857,296 (8) | +14.04% | **+10.04%** | +14.04% | +10.04% |
| case_10 / 150 | 8,649,496,154 (9) | 8,649,496,154 (9) | 8,817,538,841 (9) | 8,579,214,991 (9) | +1.94% | **-0.81%** | +1.94% | -0.81% |
| case_10 / 200 (in-sample for v2) | 8,114,297,337 (9) | 8,114,297,337 (9) | 8,784,582,880 (10) | 7,176,242,810 (7) | +8.26% | **-11.56%** | +8.26% | -11.56% |
| NLOG | 10,259,692,325 (15) | 10,259,692,325 (15) | 10,136,820,303 (15) | 10,259,281,111 (15) | -1.20% | **-0.004%** | -1.20% | -0.004% |
| Drogon | 296,650,884,980 (6) | 296,650,884,980 (6) | 296,650,035,339 (6) | 296,650,884,980 (6) | -0.0003% | **0.00%** | -0.0003% | 0.00% |
| Norne | 440,744,130,579 (8) | 440,744,130,579 (8) | 401,689,185,673 (8) | 429,722,954,535 (9) | -8.86% | **-2.50%** | -8.86% | -2.50% |

On SPE9, bc39v2's best of 10 equals the best-of-10 NPV-only greedy exactly
(51,729,831,812, 14 wells): the two searches reached the same plan.

## 6. Reading the result

Primary comparison (deterministic, vs the NPV-only greedy):

- **External benchmarks (7): bc39v2 wins 1, ties 2 exactly, loses 4.**
  bc39 lost all 7. The win is **SPE9 +17.38%** (bc39: -1.02%). The exact
  ties are SPE10 and Drogon, two of the three singleton-cluster fields
  (bc39: -1.69%, -0.0003%). It loses Egg Model -5.09%, Volve -0.10%,
  NLOG -0.004% and Norne -13.61%, each by less than bc39 (-13.31%, -0.62%,
  -5.73%, -23.19%). On all 7 external benchmarks bc39v2's deterministic NPV
  is equal to or higher than bc39's.
- The SPE9 gain is not a win over the stronger baseline: the penalised
  greedy (58.97e9) still beats bc39v2 by 12.80%. On Egg Model the ordering
  flips the other way: bc39v2 beats the penalised greedy (+6.13%) but loses
  to the NPV-only one.
- **case_10: bc39v2 is worse than bc39.** It wins only fund 48 (+35.84%,
  bc39 +17.58%) and loses 64 (-9.70%), 150 (-0.75%) and 200 (-6.46%, which
  is **in its own training pool**). bc39 won 3 of 4. So "retrain on the
  public case_10 pool" did not reproduce bc39's in-distribution advantage.
  The likely causes, none isolated here: 40% of the training budget, a
  160-well training pool (versus bc39's non-public pool), and the loss of
  the absolute-scale signal. Pool-relative features cannot tell a field of
  uniformly weak wells from one of uniformly strong wells, and bc39 may
  have been using that on case_10-like data.
- With a matched search budget, bc39v2 is ahead only on case_10/48 and
  case_10/64 (+19.21%, +10.04%). It ties SPE10, Drogon and SPE9 (the last
  via an identical plan), and is slightly behind on Egg Model (-0.37%),
  Volve, NLOG, case_10/150 and Norne (-2.50%). It is clearly behind on
  case_10/200 (-11.56%).

A diagnostic, not a claim: the deterministic bc39v2 picks the highest-NPV
candidate (row 0) more often than bc39 on most external fields (bc39 vs
bc39v2 row-0 picks: Egg 10/12 vs 5/11, SPE9 1/12 vs 4/14, SPE10 3/5 vs 4/5,
Volve 1/12 vs 6/12, NLOG 6/14 vs 5/15, Drogon 2/6 vs 5/6, Norne 0/6 vs 3/8).
On fields where the NPV-only greedy is hard to beat, bc39v2 behaves more
like it than bc39 did. This explains much of the gap closing to near-zero
on Volve, NLOG, SPE10 and Drogon.

## 7. Honest headline

Fixing the feature scaling removes the input saturation that made bc39
blind on Drogon, Norne and SPE9. The retrained checkpoint:

- no longer loses to the NPV-only greedy on 3 of the 7 external fields
  (a clear deterministic win on SPE9, exact ties on SPE10 and Drogon);
- loses by much less on Egg Model and Norne;
- is never worse than bc39 on an external field;
- is worse than bc39 on its own case_10 distribution (1 of 4 funds won
  instead of 3 of 4).

It **does not** beat the greedy baseline across the external benchmarks:
deterministically it still loses Egg Model, Volve, NLOG and Norne. On SPE9
it still loses to the penalised greedy. The singleton-cluster fix could
not be exercised, because no public training pool has singleton clusters.
bc39v2 had 40% of bc39's training budget and shows no evidence of
convergence. A full-budget run on a pool that contains singleton clusters
is the next step before any claim that the fixes, rather than the
retraining itself, cause the observed changes.
