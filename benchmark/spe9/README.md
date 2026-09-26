# SPE9 benchmark (9th SPE Comparative Solution Project)

## Section A: proxy-objective AttentionDuelingQuantileQ work (superseded/alternative, kept for history)

Everything below this line up to "Section B" is the ORIGINAL US-E2/US-E3
work: a custom, standalone candidate-set evaluator
(`benchmark/common/sequencing_env.py`) with an analytic proxy objective (not
simulated NPV) and a from-scratch attention-based dueling quantile Q-network
(`benchmark/common/attention_dueling_quantile_q.py`). It is **superseded** by
Section B (below the divider near the end of this file), which runs this
same dataset through HyperActive-public's real, deployed pipeline
(`hyperactive-plan`, the real greedy `PlanBuilder`, the real MaskedCDQN
`models/bc39` checkpoint, and the real NPV/Arps-decline economics). Section A
is kept, unmodified in substance, as an honest ablation/alternative-
architecture trail -- do not read its numbers as HyperActive NPV; they use a
different, proxy objective on a different (non-HyperActive) evaluator.

## Source

- Official SPE benchmark: "The 9th SPE Comparative Solution Project" — water-flooding
  black-oil case, 26 wells (1 injector + 25 producers), with well controls that
  change over the course of the simulated schedule.
- Open input files: `github.com/OPM/opm-data`, path `spe9/` (also mirrored/used by
  `github.com/OPM/opm-tests`).
  Repository: https://github.com/OPM/opm-data
  Commit acquired from: `eaa2261683a97027e057c2bc49612ad1c86390b3` (opm-data `master`,
  authored 2023-05-05 by Atgeirr Flø Rasmussen).

## License

**Open Database License (ODbL) v1.0.** Confirmed from the opm-data repository root
`README.md`:

> Unless stated otherwise, decks are now provided under the open database license
> available in the file odbl-10.txt, and new decks should also be provided under
> the same or a compatible license. Similarly, data contained in or pointed to by
> the decks is licensed under the open database content license available in the
> file dbcl-10.txt.

Full license text: https://github.com/OPM/opm-data/blob/master/odbl-10.txt
(database content license: https://github.com/OPM/opm-data/blob/master/dbcl-10.txt)

## Exact commands used

The full `opm-data` repository is a multi-benchmark collection; only the `spe9/`
subdirectory was pulled via a git sparse checkout, avoiding vendoring the other
benchmarks (norne, sleipner, etc.) into this repo:

```bash
mkdir opm-data-sparse && cd opm-data-sparse
git init -q
git remote add origin https://github.com/OPM/opm-data.git
git config core.sparseCheckout true
echo "spe9/*" > .git/info/sparse-checkout
git pull --depth 1 origin master
```

Result: only `spe9/*` was checked out (verified with `find . -maxdepth 2`), at
commit `eaa2261683a97027e057c2bc49612ad1c86390b3`. The `spe9/` directory was then
copied verbatim into `benchmark/spe9/raw/` in this repository.

## File listing

```
$ find benchmark/spe9/raw -type f
benchmark/spe9/raw/PERMVALUES.DATA
benchmark/spe9/raw/README
benchmark/spe9/raw/SPE9_CP.DATA
benchmark/spe9/raw/SPE9_CP_GROUP.DATA
benchmark/spe9/raw/SPE9_CP_SHORT.DATA
benchmark/spe9/raw/SPE9_CP_SHORT_RESTART.DATA
benchmark/spe9/raw/SPE9.DATA
benchmark/spe9/raw/SPE9.GRDECL
benchmark/spe9/raw/TOPSVALUES.DATA
```

```
$ ls -la benchmark/spe9/raw
142483  PERMVALUES.DATA
   593  README
 15580  SPE9_CP.DATA
 16661  SPE9_CP_GROUP.DATA
 15588  SPE9_CP_SHORT.DATA
 15754  SPE9_CP_SHORT_RESTART.DATA
 15927  SPE9.DATA
805951  SPE9.GRDECL
 10996  TOPSVALUES.DATA

$ du -sh benchmark/spe9/raw
1.1M
```

9 files, 1.1 MB total. `SPE9.DATA` is the primary Eclipse-format deck; `SPE9.GRDECL`
is the corner-point grid geometry include file; `SPE9_CP*.DATA` are corner-point
variants of the deck (per the upstream `README` in this directory, which notes
`TOPSVALUES.DATA` is not needed for the corner-point geometry variants but is kept
here since it is used by the non-CP `SPE9.DATA` deck). All files are well under any
reasonable git size threshold (largest is `SPE9.GRDECL` at ~806 KB).

## Scope note

Data acquisition (US-E2) is described above. The rest of this document
(US-E3) covers the candidate-set adapter, the real baseline runs, and the
attention-network training status.

## Candidate-set adapter (`adapter.py`)

`adapter.py` parses `raw/SPE9.DATA` (+ its `PERMVALUES.DATA` include) with
the same standalone Eclipse-deck parser used for the Egg Model
(`benchmark/common/eclipse_parse.py`) to build one candidate-set case with
all 26 wells (1 injector + 25 producers) as candidates; a "plan" is the
order in which they are brought online.

SPE9 has something the Egg Model deck doesn't: a genuine, dated
control-change schedule. `WCONPROD` is reissued three times over the
900-day run (`WCONINJE` for the single injector stays constant):

| day | all-producer oil-rate target (STB/d) |
|----:|--------------------------------------:|
|   0 |                                  1500 |
| 300 |                                   100 |
| 360 |                                  1500 |

The adapter keeps all three rate targets as separate per-well features
(rather than collapsing them to one number), since this is real schedule
structure from the source deck, not something invented for the benchmark.

### Proxy objective disclosure

**This is not a reservoir simulation.** As with the Egg Model, no
Eclipse/OPM Flow run is performed or available offline. The value used to
compare orderings is an explicit, documented analytic proxy:

```
injector value_index = 0.3 * injection_rate_target * local_permeability / 1000
producer value_index = local_permeability * mean(phase_rate_targets) / 100 / 1000
```

- `injection_rate_target` (5000 STBW/d) and the three `phase_rate_targets`
  (1500, 100, 1500 STB/d) are read directly from the deck's `WCONINJE` /
  `WCONPROD` records.
- `local_permeability` is the mean `PERMX` (mDarcy) over each well's
  completed layer range (`COMPDAT` K1:K2) at its (I, J) location, from
  `PERMVALUES.DATA`.
- The same multi-resource, travel-aware, 10%/year-discounted evaluator as
  the Egg Model is used (`benchmark/common/sequencing_env.py`) -- see
  "Duration proxy and resource/travel model" below for what changed and
  why.

This is a transparent proxy for value, not simulated cumulative
production; the numbers below are not physical NPV.

### Duration proxy and resource/travel model (fixes a flat-landscape bug)

**Superseded design note:** an earlier version of this evaluator gave every
well the same fixed `duration_days` (10) and processed all wells through a
single shared resource. Under that design the discounted objective reduces
exactly to a rearrangement-inequality problem (sort by value density,
descending), which is *provably* optimal for greedy value-density -- no
local search or learned policy could ever beat it. That reproduces the
paper's own prior reviewer complaint (r5LL: "improvements are very small
... on cases where the landscape is nearly flat"); repeating it on a new
benchmark would be a wasted effort. This has been fixed:

- **Variable duration by well type + local permeability** (not uniform):

  ```
  injector duration_days = 15 + 15 * (1 - clamp(local_permeability / 500, 0, 1))
  producer duration_days = 15 + 15 *      clamp(local_permeability / 500, 0, 1)
  ```

  (the single injector is modeled as slower to bring online in tighter
  rock; producers slower in higher-permeability rock -- the same pattern
  as the Egg Model adapter, scaled to SPE9's local-permeability range).
  This is an honest, DOCUMENTED PROXY, not a measured duration.
- **3 parallel bring-online resource slots**, not 1, using
  earliest-available-resource assignment (the same rule used by the
  private repo's Oklahoma OCC benchmark). With 26 wells and 3 slots,
  resource contention genuinely binds.
- **Grid-distance travel/setup term**: a resource pays `distance / 2.0`
  extra days between consecutive wells it services, using each well's
  (I, J) grid position -- the same distance-based-travel-time mechanism
  HyperActive's Oklahoma OCC benchmark and its `PlanBuilder` already use.

### Baselines run (real, real output)

```bash
python3 benchmark/spe9/run_benchmark.py \
    --seeds 17,29,43,71,101 --random-samples 500 --local-search-evaluations 2000 \
    --out benchmark/spe9/results.csv
```

Produces `results.csv` with one row per (method, seed): 1 greedy + 5 random
seeds + 5 local-search seeds = 11 rows, plus 1
`attention_dueling_quantile_q_rl` row (see "Training status" below) = 12
rows total. **These numbers supersede the old flat-landscape numbers from
before the duration/resource/travel fix; the old numbers are no longer
valid and are not reproduced here.**

| method                                | n | mean metric_value | min     | max     |
|-----------------------------------------|--:|---------------------:|--------:|--------:|
| `greedy_value_density`                  | 1 |               174.513 | 174.513 | 174.513 |
| `local_swap_search_2000`                | 5 |               174.533 | 174.518 | 174.539 |
| `random_500_best`                       | 5 |               174.119 | 173.846 | 174.289 |
| `attention_dueling_quantile_q_rl`       | 1 |               174.461 | 174.461 | 174.461 |

**Win/loss vs. `greedy_value_density`, per seed** (the number that
actually matters -- confirming the landscape is no longer trivially flat):

| method                            | wins | losses | ties | n |
|------------------------------------|-----:|-------:|-----:|--:|
| `local_swap_search_2000`           |    5 |      0 |    0 | 5 |
| `random_500_best`                  |    0 |      5 |    0 | 5 |
| `attention_dueling_quantile_q_rl`  |    0 |      1 |    0 | 1 |

**Honest reading of this result:** local-swap search now beats greedy on
all 5 seeds -- the fix worked; the landscape genuinely has exploitable
structure that a single-pass value-density heuristic misses.
Random-500-best still loses every time (500 samples is a weak baseline
against a 26!-permutation space with real resource/travel structure). The
trained `attention_dueling_quantile_q_rl` policy also currently loses to
greedy (see "Training status" below for why, honestly).

### Training status: TRAINED (real, small budget, honestly reported)

PyTorch **is** installed in the benchmark venv
(`/home/dmitrii/.venvs/hyperactive-benchmark`, `torch==2.14.0+cpu`) as of
this pass, so `train_attention.py` was actually run end-to-end (a prior
pass hit `ModuleNotFoundError` in a different, non-venv environment and
reported that honestly as a blocker; that blocker no longer applies here).

Training method is identical to the Egg Model's (see that README for the
full explanation): sequential-construction policy over
`q_values(observations, remaining_mask)`, trained with REINFORCE against a
running-mean baseline plus a quantile-regression term on `value_index`,
evaluated via the exact same `evaluate_order` as the baselines.

```bash
$ python3 benchmark/spe9/train_attention.py
episode=0   loss=3.158982  reward=172.892 baseline=172.892
episode=80  loss=49.478836 reward=174.249 baseline=173.342
episode=200 loss=-7.630608 reward=174.186 baseline=174.382
episode=320 loss=0.713815  reward=174.431 baseline=174.428
episode=399 loss=-0.074339 reward=174.441 baseline=174.459
trained 400 episodes in 92.84s, final_loss=-0.074339
appended 1 attention_dueling_quantile_q_rl row to benchmark/spe9/results.csv
```

`feature_dim=10`, retrained completely from scratch on SPE9 features only
(no weight transfer from the Egg Model or any other benchmark). Unlike the
Egg Model run, the reward here visibly converges toward the baseline
within 400 episodes (SPE9 has a single fixed case, vs. 12 cases sampled
per episode for the Egg Model, so the policy has an easier, more
stationary target) -- but the final greedy rollout (174.461) still falls
just short of `greedy_value_density` (174.513) and `local_swap_search_2000`
(174.518-174.539). This is reported as-is: the training loop is real and
converges, but does not yet beat the baselines on this benchmark within
this budget.

### Bootstrap ablation: expert-bootstrapped attention (EBA)

**Motivation:** same as the Egg Model's bootstrap ablation (see that
README for the full rationale) -- the architecture's "EBA"
(expert-bootstrapped attention) naming implies imitation pretraining from a
strong expert before RL fine-tuning, which the 400-episode REINFORCE-only
run above never did. `train_attention_bootstrap.py` adds that missing
bootstrap phase and scales the fine-tuning budget 7.5x (3000 episodes vs.
400).

**Phase 1 -- expert bootstrap (behavioral cloning):** SPE9 has a single
case, so demonstration diversity comes entirely from `local_swap_search`
runs from different starting points: the 5 canonical seeds already used
for the reported `local_swap_search_2000` baseline (2000 evaluations,
greedy start) plus 10 additional runs from independent random starting
orders (800 evaluations each) -- 15 expert trajectories, 390 (state,
action) supervised pairs (26 construction steps each). Pretrained with the
same cross-entropy-on-`q_values()` imitation loss as the Egg Model:

```
$ python3 benchmark/spe9/train_attention_bootstrap.py
bootstrap: 15 demo trajectories, 390 (state,action) pairs, 60 epochs in 18.82s
  bootstrap_epoch=0  cross_entropy_loss=2.334538
  bootstrap_epoch=15 cross_entropy_loss=1.851089
  bootstrap_epoch=30 cross_entropy_loss=1.818661
  bootstrap_epoch=59 cross_entropy_loss=1.717079
```

With only 15 trajectories (390 pairs) over a 26-way action space, the
imitation loss drops from 2.33 (near the ln(26)=3.26 random baseline) to
1.72 -- a smaller relative improvement than the Egg Model's (12-way, 1440
pairs), consistent with SPE9's much larger action space and thinner
demonstration set, but still a real, non-trivial reduction confirming
genuine learning signal.

**Phase 2 -- REINFORCE fine-tuning from bootstrapped weights:** identical
REINFORCE + quantile-regression loop as the no-bootstrap run, starting from
the bootstrapped weights at a lower learning rate (3e-4 vs. the bootstrap
phase's 1e-3, chosen upfront, not tuned after seeing results), for 3000
episodes (7.5x the no-bootstrap run's 400):

```
episode=0    loss=3.117737 reward=174.374 baseline=174.374
episode=300  loss=0.448701 reward=174.508 baseline=174.505
episode=600  loss=0.153074 reward=174.511 baseline=174.506
episode=900  loss=0.024242 reward=174.501 baseline=174.510
episode=1200 loss=0.112777 reward=174.514 baseline=174.511
episode=1500 loss=0.076261 reward=174.515 baseline=174.511
episode=1800 loss=0.057375 reward=174.515 baseline=174.513
episode=2100 loss=0.024511 reward=174.509 baseline=174.511
episode=2400 loss=0.149537 reward=174.513 baseline=174.512
episode=2700 loss=0.087731 reward=174.519 baseline=174.512
episode=2999 loss=0.077820 reward=174.507 baseline=174.512
fine-tuned 3000 episodes in 385.64s, final_loss=0.077820
total wall-clock (bootstrap + fine-tune): 404.46s
```

**Wall-clock time:** 404.46s (~6.7 minutes) total (18.82s bootstrap +
385.64s fine-tuning), well within the 15-20 minute check-in budget.

**Final evaluation** (greedy/argmax rollout, exact same `evaluate_order`
protocol as every baseline):

| method                                             | n | metric_value |
|-------------------------------------------------------|--:|---------------:|
| `greedy_value_density`                                 | 1 |      174.512585 |
| `local_swap_search_2000` (range across 5 seeds)        | 5 | 174.517925-174.539197 |
| `attention_dueling_quantile_q_rl` (no bootstrap)       | 1 |      174.460969 |
| `attention_dueling_quantile_q_bootstrap_rl`            | 1 |      174.515295 |

**Win/loss vs. `greedy_value_density` and vs. `local_swap_search_2000`
(per seed):**

| method                                              | vs greedy (n=1)       | vs local_swap_search_2000 (n=5) |
|----------------------------------------------------------|:-----------------------:|:----------------------------------:|
| `attention_dueling_quantile_q_rl` (no bootstrap)          | 0 wins / 1 loss / 0 ties | 0 wins / 5 losses / 0 ties |
| `attention_dueling_quantile_q_bootstrap_rl`               | **1 win** / 0 losses / 0 ties | 0 wins / 5 losses / 0 ties |

**Honest interpretation:** on SPE9, the bootstrap-then-fine-tune policy
(174.515295) genuinely edges past `greedy_value_density` (174.512585, a
0.0016% margin) -- a real, unmanufactured win, the first case in either
benchmark where the trained network beats a baseline. It still loses to
every one of the 5 `local_swap_search_2000` seeds (174.517925-174.539197),
by an equally small margin (0.0015%-0.0125%). This is a narrow result and
should be read as such: it confirms bootstrap pretraining plus a larger
fine-tuning budget can close the entire gap to greedy on this benchmark
(unlike the Egg Model, where a gap remains), but it does not establish an
advantage over the stronger `local_swap_search_2000` baseline. The
fine-tuning curve here is also far more stable than the Egg Model's (loss
and reward both settle smoothly after ~episode 300, rather than swinging
by hundreds), plausibly because SPE9's single, stationary case gives
REINFORCE an easier target to preserve the bootstrap's imitation gains on,
consistent with the note in the no-bootstrap run above about SPE9 being an
"easier, more stationary target" than the Egg Model's 12-case setting.

---

## Section B: real HyperActive pipeline with MaskedCDQN/bc39 (current, US-F2 + US-F4)

This section supersedes Section A above as the current benchmark for this
dataset: it runs the same 26 real SPE9 wells through HyperActive-public's
actual, deployed planner -- not the proxy evaluator described in Section A.
It was originally written with a one-cluster-per-well mapping (US-F2); US-F4
below replaces that mapping with real spatial-proximity clustering as the
default and is the current, primary result -- the one-cluster-per-well
numbers are kept as dated history further down.

### Real-pipeline adapter (`build_hyperactive_input.py`)

`build_hyperactive_input.py` parses the same vendored deck (`raw/SPE9.DATA`)
with the same shared parser (`benchmark/common/eclipse_parse.py`) as
`adapter.py`, but writes the real HyperActive schema documented in
[`docs/data-format.md`](../../docs/data-format.md) instead:
`benchmark/spe9/hyperactive_input/wells.xlsx` and `coordinates.xlsx`. Full
mapping detail and disclosures are in `build_hyperactive_input.py`'s module
docstring; in short (see "Real spatial-proximity clustering (US-F4)" below
for the current cluster mapping; this list uses the original US-F2
one-cluster-per-well mapping, kept as history):

* one cluster per well (no pad grouping in the archive) -- **history,
  superseded by US-F4 below**;
* `well_type = "ГС"` for all 26 wells;
* `x`/`y` from real (I, J) WELSPECS positions x the deck's real 300 ft
  (91.44 m) cell size; `length` from each well's real COMPDAT K1:K2
  completion range x the deck's real 20 ft layer thickness (producers 60 ft
  = 18.29 m, injector INJE1 100 ft = 30.48 m);
* producers: `oil_rate` = the deck's REAL day-0 WCONPROD target (1500 STB/d,
  converted to t/d with the deck's real 720.6 kg/m3 oil density) -- unlike
  the Egg Model, this is not a permeability proxy. `liq_rate` adds an
  assumed 5% water cut (proxy, not measured -- the deck gives no separate
  liquid-rate target). The deck's later rate-cut phase (100 STB/d at day
  300) and recovery (1500 STB/d at day 360) are real and documented but NOT
  representable in HyperActive's schema (`ArpsDeclineProductionProfile`
  models one starting rate that declines monotonically, with no scheduled
  rate-change mechanism) -- this loses real information from the source
  deck, disclosed rather than silently dropped;
* injector INJE1: `liq_rate` = the deck's real 5000 STB/d WCONINJE target
  (converted with real water density 1009.5 kg/m3), `oil_rate = 0`.

**Same limitation as the Egg Model:** HyperActive's economics have no
injection-support concept, so INJE1 shows negative well NPV in the results
below -- disclosed, not a bug.

### Real pipeline run (US-F2 history: one-cluster-per-well mapping)

```bash
python benchmark/spe9/build_hyperactive_input.py

.venv_real/bin/hyperactive-plan \
  --wells benchmark/spe9/hyperactive_input/wells_one_cluster_per_well.xlsx \
  --coordinates benchmark/spe9/hyperactive_input/coordinates_one_cluster_per_well.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking \
  --out benchmark/spe9/plan_bc39.csv
```

(At the time this subsection was written, `wells.xlsx`/`coordinates.xlsx`
themselves held the one-cluster-per-well mapping; US-F4 below repointed those
default filenames at the new spatial-binning mapping and moved the original
one-cluster-per-well output to the `_one_cluster_per_well` filenames shown
above, so this command is now written against those filenames to remain
reproducible.)

One command runs both the real greedy `PlanBuilder` baseline and the real
deployed `models/bc39` MaskedCDQN policy (10 stochastic rollouts, best kept).

**Real output (verbatim console output of the command above):**

```
wells: 26 valid, 0 rejected
RL plan (bc39): 6 wells, NPV 21,876,851,268
greedy plan: 6 wells, NPV 21,876,851,268 (RL vs greedy +0.00%)
```

RL and greedy selected the identical 6-well plan here (all 25 producers have
identical mapped oil/liq rates, so with no readiness-date differentiation
among wells there is little for a policy to gain over greedy by well
choice). As with the Egg Model, only 6 of 26 wells fit the 12-month/2-crew
work window because of the `min_days_between_clusters = 90` day
rig-down/rig-up floor charged between every pair of the one-cluster-per-well
mapping -- see the Egg Model README's Section B for the full explanation;
the same effect applies here unchanged.

Full RL schedule: [`plan_bc39.csv`](plan_bc39.csv).

### Real spatial-proximity clustering (US-F4, current default)

Same fix and same reasoning as the Egg Model (see that README's "Real
spatial-proximity clustering (US-F4)" section for the full method
writeup): `build_hyperactive_input.py` now groups SPE9's 26 wells into
clusters using real spatial proximity from the deck's own (I, J) grid
positions, via fixed-size spatial binning. SPE9's 25 producers happen to sit
on a single connected diagonal chain across the grid (each well's nearest
real neighbour is only 1.4-4.1 grid cells away, no natural gap to cut at),
so nearest-neighbour/agglomerative (single-linkage) clustering was
deliberately **not** used here: any single-linkage threshold above ~4.1
cells would chain the *entire* producer set into one degenerate all-in-one
cluster. Fixed-size binning avoids this because bins are bounded squares,
not a walk along nearest-neighbour edges.

Each well's (I, J) is quantized into a 700 m square bin (`n = round(700 /
91.44) = 8` grid cells, `DX = DY = 300 ft = 91.44` m); wells sharing a bin
become one cluster; a cluster's coordinate is the centroid of its member
wells. This gives **7 clusters of size [2, 2, 2, 4, 4, 6, 6]** (mean 3.7
wells/cluster) from the 26 wells -- no singleton cluster, no cluster holding
more than 6/26 wells. 700 m was chosen after trying smaller bins that stayed
close to one-per-well (300 m: 22 clusters, mostly singletons, mean 1.18;
500 m: 16 clusters, mean 1.62), fixed *before* looking at any NPV outcome
(see `build_hyperactive_input.py`'s module docstring for the full writeup).
This mapping is now written to the default `wells.xlsx` /
`coordinates.xlsx`; the original one-cluster-per-well mapping is preserved
unchanged at `wells_one_cluster_per_well.xlsx` /
`coordinates_one_cluster_per_well.xlsx` for reproducibility of the numbers
in the subsection above.

```bash
python benchmark/spe9/build_hyperactive_input.py

.venv_real/bin/hyperactive-plan \
  --wells benchmark/spe9/hyperactive_input/wells.xlsx \
  --coordinates benchmark/spe9/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking \
  --out benchmark/spe9/plan_bc39_v2.csv
```

**Real output (verbatim console output of the command above):**

```
wells: 26 valid, 0 rejected
RL plan (bc39): 13 wells, NPV 47,324,417,883
greedy plan: 16 wells, NPV 58,971,812,911 (RL vs greedy -19.75%)
```

**Wells scheduled: greedy 6/26 -> 16/26; bc39 6/26 -> 13/26.** A materially
less degenerate result on well count, same direct effect as the Egg Model:
most well-to-well moves now land within a cluster instead of paying the
90-day inter-cluster floor on every move.

**NPV, reported honestly:** with the new clustering, greedy beats bc39 by
**-19.75%** (bc39: 47,324,417,883 vs. greedy: 58,971,812,911) -- unlike the
Egg Model, bc39 does *not* come out ahead here. This is reported as-is: the
700 m bin threshold was fixed by the nearest-neighbour/cluster-size
reasoning above, before this command was run or its output looked at, and it
was not revisited after seeing this result. Greedy also schedules more
wells than bc39 here (16 vs. 13) as well as more NPV, unlike the Egg Model
case where bc39 won on NPV with fewer wells -- SPE9's producers are close to
homogeneous (identical mapped oil/liq rates across all 25 producers, see
"Real pipeline run" above), so there is less for a learned policy to
differentiate on here than on the Egg Model's mixed injector/producer,
permeability-varying pool.

Full RL schedule: [`plan_bc39_v2.csv`](plan_bc39_v2.csv).

### Exhaustive comparison

Not run for SPE9: with 26 wells (`26! `, astronomically infeasible) and no
well-count-limited PRD requirement for this dataset (unlike the Egg Model's
12), `experiments/evaluate.py --exhaustive-max-wells 8` would only ever
exercise a random 8-well subset that says nothing about SPE9-scale planning;
the Egg Model README documents that comparison instead, where it was
explicitly requested.

### Results table

See [`benchmark/results_real_pipeline.csv`](../results_real_pipeline.csv)
for the combined Egg Model + SPE9 real-pipeline results table.


## Baseline fairness correction (bug-fix pass)

The earlier numbers in this README (Section B, US-F4: bc39 -19.75% vs greedy) are kept as recorded; this
section supersedes them for the bc39-vs-greedy comparison. Overview of the
whole pass: [`../BASELINE_FAIRNESS_CORRECTION.md`](../BASELINE_FAIRNESS_CORRECTION.md).

**What was wrong.** The greedy baseline `PlanBuilder` ranks candidates by
`NPV - drill_team_penalty` (its default, used by `hyperactive-plan`), while
`PlanEnv` -- which builds the candidate set bc39 sees and acts on -- ranks
candidates by NPV alone. "Greedy" and "the environment's best-ranked
action" were therefore two different policies, and wherever candidate NPVs
tie (identical rates) or the penalty reorders them, the two rules take
different wells.

**Fix.** Both greedy variants are now reported. The PRIMARY baseline is
`PlanBuilder(use_drill_team_penalty=False)` (`--greedy-penalty off`), the
ranking rule bc39 acts under; a regression test (`tests/test_env.py`,
`TestGreedyRankingMatchesPlanBuilder`) checks that always taking
`PlanEnv`'s row 0 reproduces it exactly. The penalised greedy is kept as a
SECONDARY baseline: it is the rule the paper's method section describes,
and it can be the stronger heuristic. bc39, its checkpoint and its
environment are unchanged; the bc39 number below is the same deterministic
10-episode run as before.

Command (same parameters as before, both greedy variants):

```bash
.venv_real/bin/hyperactive-plan \
  --wells benchmark/spe9/hyperactive_input/wells.xlsx \
  --coordinates benchmark/spe9/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking --greedy-penalty both \
  --out benchmark/spe9/plan_bc39_v2.csv
```

```
wells: 26 valid, 0 rejected
RL plan (bc39): 13 wells, NPV 47,324,417,883
greedy plan (NPV only): 12 wells, NPV 43,808,870,706 (RL vs greedy +8.02%)
greedy plan (NPV - drill_team_penalty): 16 wells, NPV 58,971,812,911 (RL vs greedy -19.75%)
```

| | wells | NPV |
|---|---:|---:|
| bc39 | 13 | 47,324,417,883 |
| greedy, NPV only (PRIMARY) | 12 | 43,808,870,706 |
| greedy, NPV - drill_team_penalty (secondary, previously the only baseline) | 16 | 58,971,812,911 |

bc39 vs primary greedy: **+8.02%**; vs penalised greedy:
-19.75% (previously reported: -19.75%).

The sign flips. 25 producers share one rate, so tie-breaking decides the
plan: the NPV-only greedy scores 43.81e9, bc39 beats it by +8.02%. But the
penalised greedy, which breaks the ties differently, scores 58.97e9 and
still beats bc39 by 19.75%. Read honestly: bc39 improves on the ranking rule
it acts under, but on SPE9 a simple heuristic (NPV - drill_team_penalty) is
clearly better than bc39; the +8.02% must not be reported without that.

**Disclosed limitation: feature-scale saturation in the bc39 checkpoint.**
`src/hyperactive/env/features.py` feeds `oil_rate`, `liq_rate`, `cost_npv`,
`cash_flow`, `capex` and `travel_cost` to the network as absolute values;
bc39's VecNormalize statistics were fit on its training distribution and
clip normalised features at +/-10 (`clip_obs=10.0`). On this benchmark
**86.2% of candidate rows** have at least one feature at the clip bound
(row-0 rollout, 87 valid candidate rows; per feature: cost_npv 86%, cash_flow 86%, oil_rate 25%, job_window_share 5%). On the
clipped features the network cannot tell those candidates apart, so bc39 is
here being evaluated outside the input range it was trained on. This is
disclosed, not corrected: no retraining and no inference-time rescaling was
applied, because either would change the model after the fact without
proper training and validation.
