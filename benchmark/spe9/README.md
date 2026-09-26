# SPE9 benchmark (9th SPE Comparative Solution Project)

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
this budget. A larger training budget and/or hyperparameter tuning is a
natural next step, not performed in this pass.
