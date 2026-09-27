# bc39v2: retraining with pool-relative features and singleton-cluster sampling

Follow-up to [`SEARCH_BUDGET_CORRECTION.md`](SEARCH_BUDGET_CORRECTION.md).
That pass made the bc39-vs-greedy comparison fair and found that bc39 beats
the greedy baseline only on `case_10`. This pass fixes two defects in the
model's own training setup, retrains a new checkpoint (`models/bc39v2/`),
and re-runs the same 8-benchmark evaluation zero-shot. `models/bc39/` is
unchanged and still reproduces its published numbers bit for bit. Rows of
this pass in `results_real_pipeline.csv` are tagged
`BC39V2 (retrain with fixed features + singleton-cluster sampling)`.

**Follow-up:** [section 8](#8-follow-up-bc39v2-full-full-160000-timestep-budget)
retrains the same setup to bc39's full 160,000-timestep budget
(`models/bc39v2-full/`) and gives the three-way comparison bc39 /
bc39v2 (64k) / bc39v2-full.
[Section 9](#9-multi-seed-variance-study-at-the-full-160k-budget) retrains
the bc39v2-full setup with 3 more seeds (`models/bc39v2-seed{1,2,3}/`) to
measure seed-to-seed variance.

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

## 8. Follow-up: bc39v2-full, full 160,000-timestep budget

bc39v2 above had 40% of bc39's budget. This follow-up trains the identical
setup to bc39's full 160,000 timesteps, as `models/bc39v2-full/`.
`models/bc39/` and `models/bc39v2/` are unchanged. Rows in
`results_real_pipeline.csv` are tagged
`BC39V2FULL (full 160k-timestep retrain)`. Every earlier row is kept as is.

### 8.1 Training run

The 64k run could not be resumed. `experiments/train.py` has no resume
option. The periodic checkpoints (`--save-interval`) hold only the network
and the `VecNormalize` statistics, not the replay buffer or the RNG state.
The epsilon schedule is also defined as a fraction of `total_timesteps`
(`exploration_fraction` 0.8), so a continued run would not follow the
schedule of a 160k run. bc39v2-full was therefore trained from scratch.

Budget: `experiments/train.py:660` sets `total_steps = rl_episodes *
well_count`. The 64k run confirms this: `--rl-episodes 4000`, 16 wells,
last `env_step` 63,999 in `runs/bc39v2/episodes.csv`. So
`--rl-episodes 10000` gives 160,000. Note that `--rl-episodes` counts
units of `well_count` steps, not environment episodes. The real episodes
average about 6.1 steps, so the 64k run had 10,494 of them and this run had
26,494.

```bash
.venv_real/bin/python experiments/train.py --run-name bc39v2-full \
  --wells-file benchmark/case_10/200/wells_200.xlsx \
  --coordinates-file benchmark/case_10/200/coordinates_200.xlsx \
  --settings models/bc39/training_settings.json \
  --seed 8 --well-count 16 --action-window 8 --rl-episodes 10000 \
  --sampling clusters --net-width 64 --weight-decay 0.0001 \
  --behavioral-cloning 1 --greedy-prefill 0 --hot-start-subsets 30 \
  --bc-patience 16 --greedy-every 50 \
  --crew-mix 2x1,2x1,2x1,2x5,3x2,3x4,4x1,4x3,5x2,5x5 \
  --horizon-mix 5,8,10,15,20,25 --drilling-months-mix 12,12,12,18,18,24,36,48,60 \
  --oil-constraint-prob 0.5 --oil-constraint-share 0.08:0.35 \
  --save-interval 500 --no-tracking
```

Only `--run-name` and `--rl-episodes` differ from the bc39v2 command.

- **Timesteps reached: 160,000** (last `env_step` 159,999; 26,494 episodes).
  The run completed its full schedule and exited with status 0.
- **Wall clock** (`/usr/bin/time`): **2:58:41**, from 2026-09-26
  19:59:17Z to 22:57:58Z.
  - Warm start: 500 s. bc39v2 took 1,144 s on a more loaded machine.
  - RL: 10,190 s (169.8 min, 15.7 timesteps/s).
  - In-training evaluation: about 530 s.
- The warm start is the same as bc39v2's, since both use the same seed.
  Mean behavioural-cloning validation accuracy is 0.894 in both runs. The
  RL phases differ because the epsilon schedule is stretched over 160k
  steps. Epsilon reached its final 0.05 at episode 21,050, at
  128,000 steps (0.8 of the total).
- In-training evaluation (`runs/bc39v2-full/summary.json`, 30 subsets per
  pool, vs the default 2x2 greedy):

  | pool | aggregate uplift | win rate |
  |---|---:|---:|
  | train, bc39v2-full | +2.30% | 70.0% |
  | train, bc39v2 | +2.60% | 46.7% |
  | hold-out, bc39v2-full | +0.61% | 56.7% |
  | hold-out, bc39v2 | -1.02% | 36.7% |

### 8.2 Convergence assessment

The in-training curve uses 1 greedy reference per 50 episodes. Those
episodes are run epsilon-greedy. The data is in
`runs/bc39v2-full/episodes.csv`. Grouped into 20,000-timestep bins:

| timesteps | references | mean epsilon | median uplift | win rate | aggregate uplift (sum NPV_rl / sum NPV_greedy - 1) |
|---|---:|---:|---:|---:|---:|
| 0-20k | 66 | 0.28 | -13.81% | 43.9% | -8.75% |
| 20-40k | 66 | 0.24 | +8.86% | 51.5% | -10.75% |
| 40-60k | 65 | 0.20 | +12.95% | 55.4% | +3.49% |
| 60-80k | 66 | 0.16 | -15.59% | 45.5% | -12.55% |
| 80-100k | 65 | 0.12 | -19.02% | 43.1% | -14.19% |
| 100-120k | 65 | 0.09 | +14.45% | 52.3% | -0.24% |
| 120-140k | 69 | 0.05 | -12.19% | 47.8% | -6.80% |
| 140-160k | 67 | 0.05 | +4.36% | 52.2% | +10.85% |

**No convergence is visible, and neither is any improvement.**
- The win rate stays between 43% and 55% in every bin, from the first to
  the last.
- The aggregate uplift swings between -14% and +11% with no monotone trend.
  The last bin (+10.85%) is the best, but the bin before it has the same
  epsilon and gave -6.80%.
- A linear fit over the second half (per-reference uplift, clipped at
  +/-100%) has a slope of +0.10 percentage points per 1,000 steps. That is
  small next to the noise.

With about 65 noisy references per bin, this signal cannot separate "has
plateaued" from "is still slowly improving". The honest summary: the curve
is flat within its noise for the whole run. Tripling the RL budget did not
produce a trend visible in this signal. The 30-subset evaluation after
training (section 8.1) improved over the 64k run on win rate and hold-out
uplift, but not on train aggregate uplift.

### 8.3 Three-way comparison (zero-shot, all 8 benchmarks)

The protocol, commands and parameters are exactly those of section 5, with
`--model models/bc39v2-full`. There is no per-benchmark fine-tuning. All
greedy numbers again reproduced the CORRECTED2 greedy numbers exactly, in
both protocols and for both rankings. Outputs:
- `benchmark/<b>/plan_bc39v2full_deterministic.csv` and
  `plan_bc39v2full.csv`;
- `benchmark/case_10/results_bc39v2full_{deterministic,matched_budget}.xlsx`.

NPV, with wells scheduled in parentheses. Uplift = policy / greedy - 1.
case_10/200 is in-sample for both bc39v2 checkpoints (section 5 caveat).

#### Primary: deterministic (1 episode, exploration 0) vs one deterministic greedy pass

| benchmark | greedy NPV only | greedy NPV - penalty | bc39v2-full NPV | **bc39 vs NPV-only** | **bc39v2 (64k) vs NPV-only** | **bc39v2-full vs NPV-only** | bc39v2-full vs penalised |
|---|---:|---:|---:|---:|---:|---:|---:|
| Egg Model | 1,710,121,760 (10) | 1,529,277,304 (12) | 1,707,095,723 (10) | -13.31% | -5.09% | **-0.18%** | +11.63% |
| SPE9 | 43,808,870,706 (12) | 58,971,812,911 (16) | 58,971,812,911 (16) | -1.02% | +17.38% | **+34.61%** | 0.00% (same NPV) |
| SPE10 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,709,757,963 (5) | -1.69% | 0.00% | **-3.08%** | -3.08% |
| Volve | 6,429,779,953 (12) | 6,429,779,953 (12) | 6,427,252,520 (12) | -0.62% | -0.10% | **-0.04%** | -0.04% |
| NLOG | 10,259,729,536 (15) | 10,259,729,536 (15) | 10,259,729,536 (15) | -5.73% | -0.004% | **0.00% (same NPV)** | 0.00% |
| Drogon | 296,650,884,980 (6) | 296,650,884,980 (6) | 296,650,578,690 (6) | -0.0003% | 0.00% | **-0.0001%** | -0.0001% |
| Norne | 440,744,130,579 (8) | 440,744,130,579 (8) | 377,415,893,237 (7) | -23.19% | -13.61% | **-14.37%** | -14.37% |
| case_10 / 48 | 4,860,649,333 (6) | 4,860,649,333 (6) | 4,860,649,333 (6) | +17.58% | +35.84% | **0.00% (same NPV)** | 0.00% |
| case_10 / 64 | 10,603,736,532 (7) | 10,603,736,532 (7) | 11,006,110,358 (8) | +16.18% | -9.70% | **+3.79%** | +3.79% |
| case_10 / 150 | 8,155,481,707 (8) | 8,155,481,707 (8) | 7,399,346,900 (7) | -11.25% | -0.75% | **-9.27%** | -9.27% |
| case_10 / 200 (in-sample for v2, v2-full) | 6,950,302,884 (6) | 6,950,302,884 (6) | 8,843,770,238 (10) | +17.23% | -6.46% | **+27.24%** | +27.24% |

#### Secondary: best of 10 at exploration 0.15, both sides (matched search budget)

| benchmark | greedy NPV only, best of 10 | greedy NPV - penalty, best of 10 | bc39v2-full best of 10 | **bc39 vs NPV-only** | **bc39v2 (64k) vs NPV-only** | **bc39v2-full vs NPV-only** | bc39v2-full vs penalised |
|---|---:|---:|---:|---:|---:|---:|---:|
| Egg Model | 1,710,121,760 (10) | 1,704,866,230 (10) | 1,801,823,597 (9) | +0.72% | -0.37% | **+5.36%** | +5.69% |
| SPE9 | 51,729,831,812 (14) | 58,971,812,911 (16) | 58,971,812,911 (16) | -8.52% | 0.00% | **+14.00%** | 0.00% (same NPV) |
| SPE10 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) | 0.00% | 0.00% | **0.00%** | 0.00% |
| Volve | 6,429,779,953 (12) | 6,429,779,953 (12) | 6,428,242,807 (12) | -0.10% | -0.05% | **-0.02%** | -0.02% |
| NLOG | 10,259,692,325 (15) | 10,259,692,325 (15) | 10,259,692,325 (15) | -1.20% | -0.004% | **0.00% (same NPV)** | 0.00% |
| Drogon | 296,650,884,980 (6) | 296,650,884,980 (6) | 296,650,578,690 (6) | -0.0003% | 0.00% | **-0.0001%** | -0.0001% |
| Norne | 440,744,130,579 (8) | 440,744,130,579 (8) | 380,734,886,167 (7) | -8.86% | -2.50% | **-13.62%** | -13.62% |
| case_10 / 48 | 6,041,438,472 (8) | 6,041,438,472 (8) | 6,021,978,785 (8) | +12.30% | +19.21% | **-0.32%** | -0.32% |
| case_10 / 64 | 10,704,025,147 (7) | 10,704,025,147 (7) | 11,500,491,261 (8) | +14.04% | +10.04% | **+7.44%** | +7.44% |
| case_10 / 150 | 8,649,496,154 (9) | 8,649,496,154 (9) | 8,670,178,821 (9) | +1.94% | -0.81% | **+0.24%** | +0.24% |
| case_10 / 200 (in-sample for v2, v2-full) | 8,114,297,337 (9) | 8,114,297,337 (9) | 9,774,227,843 (11) | +8.26% | -11.56% | **+20.46%** | +20.46% |

#### Win / tie / loss counts vs the NPV-only greedy

"Tie" means an identical NPV. Every other difference counts as a win or a
loss, however small (Drogon's -0.0001% is a loss).

| | bc39 | bc39v2 (64k) | bc39v2-full |
|---|---|---|---|
| deterministic, 7 external | 0 / 0 / 7 | 1 / 2 / 4 | 1 / 1 / 5 |
| deterministic, case_10 (4) | 3 / 0 / 1 | 1 / 0 / 3 | 2 / 1 / 1 |
| matched budget, 7 external | 1 / 1 / 5 | 0 / 3 / 4 | 2 / 2 / 3 |
| matched budget, case_10 (4) | 4 / 0 / 0 | 2 / 0 / 2 | 3 / 0 / 1 |

### 8.4 Reading the result

**Deterministic (primary), 7 external benchmarks.**
- bc39v2-full beats the NPV-only greedy on **SPE9 only (+34.61%)**. There
  its plan has exactly the penalised greedy's NPV (58,971,812,911, 16
  wells), so on SPE9 it matches the stronger baseline and does not beat it.
- It ties NLOG exactly.
- It loses the other five:
  - Egg Model -0.18%;
  - SPE10 -3.08%;
  - Volve -0.04%;
  - Drogon -0.0001%;
  - Norne -14.37%.
- Four of those losses are small. SPE10's -3.08% is the largest SPE10 loss
  of the three checkpoints: bc39 lost -1.69% and bc39v2 tied.
- Norne is essentially unchanged from bc39v2 (-13.61%) and still clearly
  lost.

**The larger budget is not a uniform improvement over bc39v2 (64k).**
Deterministically, bc39v2-full is:
- better on Egg Model, SPE9, Volve, NLOG, case_10/64 and case_10/200;
- worse on SPE10, Drogon, Norne, case_10/48 and case_10/150.

The external win count does not improve: 1 in both checkpoints. The
external tie count drops from 2 to 1. It is also no longer true that the
retrained model is never worse than bc39 on an external field: SPE10 is
worse (-3.08% vs -1.69%).

**case_10 partly recovers, mostly in-sample.**
- bc39v2-full wins case_10/64 (+3.79%, bc39 +16.18%) and case_10/200
  (+27.24%, bc39 +17.23%).
- It ties case_10/48 exactly. bc39v2 had +35.84% there and bc39 +17.58%.
- It loses case_10/150 (-9.27%).
- The only case_10 cell where it beats bc39 is 200, which is in its
  training pool. On the three funds it did not train on, it is below bc39
  on 48 and 64 and above bc39 only on 150, where both lose.

So bc39v2-full does not recover bc39's out-of-sample case_10 advantage.

**Matched budget (secondary).**
- bc39v2-full's best-of-10 wins Egg Model (+5.36%, the best of the three
  checkpoints) and SPE9 (+14.00%, again with the same NPV as the penalised
  greedy).
- It ties SPE10 and NLOG.
- It loses Volve (-0.02%), Drogon (-0.0001%) and Norne (-13.62%). On Norne,
  bc39v2's -2.50% becomes a much larger loss, worse than bc39's -8.86%.
- On case_10 it wins 64, 150 and 200 and loses 48 by 0.32%.

**Single-seed caveat.** Each checkpoint is one training run with one seed.
The cell-level differences between bc39v2 and bc39v2-full have mixed signs
and are often large: case_10/48 moves from +35.84% to 0.00%, case_10/200
from -6.46% to +27.24%. That suggests large run-to-run variance. We did
not measure seed variance, so these differences cannot be attributed to
the training budget alone.

### 8.5 Honest headline (bc39v2-full)

Training the fixed-feature setup to bc39's full 160,000-timestep budget
gives a checkpoint that:

- still does **not** beat the greedy baseline across the external
  benchmarks. Deterministically, against the NPV-only greedy, it wins 1 of
  7 (SPE9), ties 1 (NLOG) and loses 5. Against the penalised greedy it
  wins Egg Model and ties SPE9 and NLOG.
- keeps the SPE9 gain from the feature fix, and on SPE9 reaches but does
  not exceed the penalised greedy.
- is close to the greedy on Egg Model, Volve, NLOG and Drogon (within
  0.2%), loses SPE10 by 3%, and still loses Norne by about 14%.
- wins 2 of 4 case_10 funds deterministically, up from 1 for bc39v2. One
  of the two is its in-sample training pool. It does not recover bc39's
  3-of-4 out-of-sample case_10 result.
- shows no convergence trend in its training curve (section 8.2). More
  budget did not yield a consistent improvement over the 64k checkpoint.

The comparison supports the claim that the feature-scaling fix removes the
input saturation and the large bc39 losses on SPE9 and Egg Model. It does
**not** support a claim that the learned scorer beats a simple NPV-ranked
greedy zero-shot on external fields. The singleton-cluster fix is still
unexercised: no public training pool has singleton clusters. Multi-seed
training is needed before any budget effect can be claimed.

## 9. Multi-seed variance study at the full 160k budget

Sections 5 and 8 each rest on one training run with seed 8. The cell-level
changes between the 64k and 160k checkpoints were large and had mixed
signs, so they could be a budget effect or ordinary seed-to-seed RL
variance. This section trains the bc39v2-full setup with 3 more seeds and
evaluates each checkpoint with the same protocol.

**Status: in progress.** Seed 1 is done. Seeds 2 and 3 are training
sequentially; this section is updated as each one finishes.

### 9.1 Seed choice and run plan

- **Seeds 1, 2, 3**: the first three positive integers other than 8. They
  were fixed before any result was seen. There is no other reason for the
  choice.
- **Setup:** identical to bc39v2-full (section 8.1) except `--seed S` and
  `--run-name bc39v2-seedS`. Same budget: `--rl-episodes 10000`, which is
  160,000 timesteps. Checkpoints are in `models/bc39v2-seed{1,2,3}/`.
  `models/bc39/`, `models/bc39v2/` and `models/bc39v2-full/` are unchanged.
- The `--seed` flag also sets the train/hold-out split unless `--split-seed`
  is given. `split_seed` is null here, as for bc39v2-full, so the split
  changes with the seed. Each seed therefore holds out a different 40 of
  the 200 `case_10/200` wells. This is part of what "seed variance" means in
  this study. case_10/200 is in-sample for every seed either way.
- **Runs were sequential, not parallel.** The machine has 4 cores. One
  training run used about 190% CPU (`top`, 5 s sample during the RL phase),
  and the machine was about 41% idle with it running. A second run would
  have needed more than the free capacity, so running two at once would
  have slowed both.
- Evaluation of each seed ran right after its training and before the next
  seed started, so no evaluation overlapped a training run.

### 9.2 Training runs

| seed | wall clock (`/usr/bin/time`) | start - end (UTC) | warm start | RL | RL timesteps/s | episodes | BC val. acc. (mean of 30) | in-training eval, train pool: aggregate uplift / win rate | hold-out pool |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 8 (bc39v2-full) | 2:58:41 | 2026-09-26 19:59:17 - 22:57:58 | 500 s | 10,190 s | 15.7 | 26,494 | 0.894 | +2.30% / 70.0% | +0.61% / 56.7% |
| 1 | 3:09:52 | 2026-09-27 07:43:06 - 10:52:59 | 458 s | 10,810 s | 14.8 | 26,522 | 0.885 | +1.86% / 50.0% | -0.45% / 36.7% |

Every run completed the full 160,000-timestep schedule and exited with
status 0. `episodes.csv` logs finished episodes only. For seed 1 the last
logged episode ends at env step 159,996; `learn()` stops at 160,000 in the
middle of the next episode.

The in-training evaluation columns are not comparable across seeds. The
train and hold-out pools, and the 30 evaluation subsets
(`experiments/train.py:728`, `split_seed + 3`), both depend on the seed.

### 9.3 Per-seed results (interim: seeds 8 and 1)

Same protocol, commands and parameters as sections 5 and 8.3, with
`--model models/bc39v2-seedS`. Outputs:
`benchmark/<b>/plan_bc39v2seedS_deterministic.csv`, `plan_bc39v2seedS.csv`
and `benchmark/case_10/results_bc39v2seedS_{deterministic,matched_budget}.xlsx`.
Every greedy number reproduced the CORRECTED2 greedy exactly, in both
protocols and for both rankings. Seed 8 is bc39v2-full (section 8.3).
NPV, wells scheduled in parentheses. A tie is an NPV within 1 unit
(rounding in the case_10 workbooks); any other difference is a win or a
loss.
#### NPV per seed, deterministic (1 episode, exploration 0)

| benchmark | greedy NPV only | greedy NPV - penalty | seed 8 | seed 1 |
|---|---:|---:|---:|---:|
| Egg Model | 1,710,121,760 (10) | 1,529,277,304 (12) | 1,707,095,723 (10) | 1,896,517,681 (8) |
| SPE9 | 43,808,870,706 (12) | 58,971,812,911 (16) | 58,971,812,911 (16) | 43,808,870,706 (12) |
| SPE10 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,709,757,963 (5) | 1,764,158,272 (5) |
| Volve | 6,429,779,953 (12) | 6,429,779,953 (12) | 6,427,252,520 (12) | 6,417,439,261 (12) |
| NLOG | 10,259,729,536 (15) | 10,259,729,536 (15) | 10,259,729,536 (15) | 10,259,705,249 (15) |
| Drogon | 296,650,884,980 (6) | 296,650,884,980 (6) | 296,650,578,690 (6) | 296,650,884,980 (6) |
| Norne | 440,744,130,579 (8) | 440,744,130,579 (8) | 377,415,893,237 (7) | 440,670,628,357 (8) |
| case_10 / 48 | 4,860,649,333 (6) | 4,860,649,333 (6) | 4,860,649,333 (6) | 7,816,076,935 (11) |
| case_10 / 64 | 10,603,736,532 (7) | 10,603,736,532 (7) | 11,006,110,358 (8) | 10,698,541,555 (7) |
| case_10 / 150 | 8,155,481,707 (8) | 8,155,481,707 (8) | 7,399,346,900 (7) | 8,155,615,432 (8) |
| case_10 / 200 (in-sample) | 6,950,302,884 (6) | 6,950,302,884 (6) | 8,843,770,238 (10) | 7,539,042,365 (7) |

#### NPV per seed, best of 10 at exploration 0.15 (matched budget)

| benchmark | greedy NPV only | greedy NPV - penalty | seed 8 | seed 1 |
|---|---:|---:|---:|---:|
| Egg Model | 1,710,121,760 (10) | 1,704,866,230 (10) | 1,801,823,597 (9) | 1,896,517,681 (8) |
| SPE9 | 51,729,831,812 (14) | 58,971,812,911 (16) | 58,971,812,911 (16) | 51,729,831,812 (14) |
| SPE10 | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) | 1,764,158,272 (5) |
| Volve | 6,429,779,953 (12) | 6,429,779,953 (12) | 6,428,242,807 (12) | 6,418,232,131 (12) |
| NLOG | 10,259,692,325 (15) | 10,259,692,325 (15) | 10,259,692,325 (15) | 10,259,684,303 (15) |
| Drogon | 296,650,884,980 (6) | 296,650,884,980 (6) | 296,650,578,690 (6) | 296,650,884,980 (6) |
| Norne | 440,744,130,579 (8) | 440,744,130,579 (8) | 380,734,886,167 (7) | 440,670,628,357 (8) |
| case_10 / 48 | 6,041,438,472 (8) | 6,041,438,472 (8) | 6,021,978,785 (8) | 7,816,076,935 (11) |
| case_10 / 64 | 10,704,025,147 (7) | 10,704,025,147 (7) | 11,500,491,261 (8) | 11,389,241,800 (8) |
| case_10 / 150 | 8,649,496,154 (9) | 8,649,496,154 (9) | 8,670,178,821 (9) | 8,649,629,879 (9) |
| case_10 / 200 (in-sample) | 8,114,297,337 (9) | 8,114,297,337 (9) | 9,774,227,843 (11) | 8,794,134,914 (9) |

#### Uplift vs the NPV-only greedy (primary baseline), deterministic (1 episode, exploration 0)

| benchmark | seed 8 | seed 1 | mean | std (pp) | range | W / T / L |
|---|---:|---:|---:|---:|---|---|
| Egg Model | -0.18% | +10.90% | +5.36% | 7.83 | -0.18% .. +10.90% | 1 / 0 / 1 |
| SPE9 | +34.61% | 0.00% | +17.31% | 24.47 | 0.00% .. +34.61% | 1 / 1 / 0 |
| SPE10 | -3.08% | 0.00% | -1.54% | 2.18 | -3.08% .. 0.00% | 0 / 1 / 1 |
| Volve | -0.04% | -0.19% | -0.12% | 0.11 | -0.19% .. -0.04% | 0 / 0 / 2 |
| NLOG | 0.00% | -0.0002% | -0.0001% | 0.00 | -0.0002% .. 0.00% | 0 / 1 / 1 |
| Drogon | -0.0001% | 0.00% | -0.0001% | 0.00 | -0.0001% .. 0.00% | 0 / 1 / 1 |
| Norne | -14.37% | -0.02% | -7.19% | 10.15 | -14.37% .. -0.02% | 0 / 0 / 2 |
| case_10 / 48 | 0.00% | +60.80% | +30.40% | 42.99 | 0.00% .. +60.80% | 1 / 1 / 0 |
| case_10 / 64 | +3.79% | +0.89% | +2.34% | 2.05 | +0.89% .. +3.79% | 2 / 0 / 0 |
| case_10 / 150 | -9.27% | +0.0016% | -4.63% | 6.56 | -9.27% .. +0.0016% | 1 / 0 / 1 |
| case_10 / 200 (in-sample) | +27.24% | +8.47% | +17.86% | 13.27 | +8.47% .. +27.24% | 2 / 0 / 0 |

#### Uplift vs the penalised greedy, deterministic (1 episode, exploration 0)

| benchmark | seed 8 | seed 1 | mean | std (pp) | range | W / T / L |
|---|---:|---:|---:|---:|---|---|
| Egg Model | +11.63% | +24.01% | +17.82% | 8.76 | +11.63% .. +24.01% | 2 / 0 / 0 |
| SPE9 | 0.00% | -25.71% | -12.86% | 18.18 | -25.71% .. 0.00% | 0 / 1 / 1 |
| SPE10 | -3.08% | 0.00% | -1.54% | 2.18 | -3.08% .. 0.00% | 0 / 1 / 1 |
| Volve | -0.04% | -0.19% | -0.12% | 0.11 | -0.19% .. -0.04% | 0 / 0 / 2 |
| NLOG | 0.00% | -0.0002% | -0.0001% | 0.00 | -0.0002% .. 0.00% | 0 / 1 / 1 |
| Drogon | -0.0001% | 0.00% | -0.0001% | 0.00 | -0.0001% .. 0.00% | 0 / 1 / 1 |
| Norne | -14.37% | -0.02% | -7.19% | 10.15 | -14.37% .. -0.02% | 0 / 0 / 2 |
| case_10 / 48 | 0.00% | +60.80% | +30.40% | 42.99 | 0.00% .. +60.80% | 1 / 1 / 0 |
| case_10 / 64 | +3.79% | +0.89% | +2.34% | 2.05 | +0.89% .. +3.79% | 2 / 0 / 0 |
| case_10 / 150 | -9.27% | +0.0016% | -4.63% | 6.56 | -9.27% .. +0.0016% | 1 / 0 / 1 |
| case_10 / 200 (in-sample) | +27.24% | +8.47% | +17.86% | 13.27 | +8.47% .. +27.24% | 2 / 0 / 0 |

#### Uplift vs the NPV-only greedy (primary baseline), best of 10 at exploration 0.15 (matched budget)

| benchmark | seed 8 | seed 1 | mean | std (pp) | range | W / T / L |
|---|---:|---:|---:|---:|---|---|
| Egg Model | +5.36% | +10.90% | +8.13% | 3.92 | +5.36% .. +10.90% | 2 / 0 / 0 |
| SPE9 | +14.00% | 0.00% | +7.00% | 9.90 | 0.00% .. +14.00% | 1 / 1 / 0 |
| SPE10 | 0.00% | 0.00% | 0.00% | 0.00 | 0.00% .. 0.00% | 0 / 2 / 0 |
| Volve | -0.02% | -0.18% | -0.10% | 0.11 | -0.18% .. -0.02% | 0 / 0 / 2 |
| NLOG | 0.00% | -0.0001% | -0.0000% | 0.00 | -0.0001% .. 0.00% | 0 / 1 / 1 |
| Drogon | -0.0001% | 0.00% | -0.0001% | 0.00 | -0.0001% .. 0.00% | 0 / 1 / 1 |
| Norne | -13.62% | -0.02% | -6.82% | 9.62 | -13.62% .. -0.02% | 0 / 0 / 2 |
| case_10 / 48 | -0.32% | +29.37% | +14.53% | 21.00 | -0.32% .. +29.37% | 1 / 0 / 1 |
| case_10 / 64 | +7.44% | +6.40% | +6.92% | 0.73 | +6.40% .. +7.44% | 2 / 0 / 0 |
| case_10 / 150 | +0.24% | +0.0015% | +0.12% | 0.17 | +0.0015% .. +0.24% | 2 / 0 / 0 |
| case_10 / 200 (in-sample) | +20.46% | +8.38% | +14.42% | 8.54 | +8.38% .. +20.46% | 2 / 0 / 0 |

#### Uplift vs the penalised greedy, best of 10 at exploration 0.15 (matched budget)

| benchmark | seed 8 | seed 1 | mean | std (pp) | range | W / T / L |
|---|---:|---:|---:|---:|---|---|
| Egg Model | +5.69% | +11.24% | +8.46% | 3.93 | +5.69% .. +11.24% | 2 / 0 / 0 |
| SPE9 | 0.00% | -12.28% | -6.14% | 8.68 | -12.28% .. 0.00% | 0 / 1 / 1 |
| SPE10 | 0.00% | 0.00% | 0.00% | 0.00 | 0.00% .. 0.00% | 0 / 2 / 0 |
| Volve | -0.02% | -0.18% | -0.10% | 0.11 | -0.18% .. -0.02% | 0 / 0 / 2 |
| NLOG | 0.00% | -0.0001% | -0.0000% | 0.00 | -0.0001% .. 0.00% | 0 / 1 / 1 |
| Drogon | -0.0001% | 0.00% | -0.0001% | 0.00 | -0.0001% .. 0.00% | 0 / 1 / 1 |
| Norne | -13.62% | -0.02% | -6.82% | 9.62 | -13.62% .. -0.02% | 0 / 0 / 2 |
| case_10 / 48 | -0.32% | +29.37% | +14.53% | 21.00 | -0.32% .. +29.37% | 1 / 0 / 1 |
| case_10 / 64 | +7.44% | +6.40% | +6.92% | 0.73 | +6.40% .. +7.44% | 2 / 0 / 0 |
| case_10 / 150 | +0.24% | +0.0015% | +0.12% | 0.17 | +0.0015% .. +0.24% | 2 / 0 / 0 |
| case_10 / 200 (in-sample) | +20.46% | +8.38% | +14.42% | 8.54 | +8.38% .. +20.46% | 2 / 0 / 0 |


The 4-seed aggregate and its reading are added in sections 9.4 and 9.5
once seeds 2 and 3 finish.
