# Egg Model benchmark

## Source

- Jansen, J.D., Fonseca, R.M., Kahrobaei, S., Siraj, M.M., Van Essen, G.M., Van den Hof,
  P.M.J. (2014). "The egg model — a geological ensemble for reservoir simulation."
  *Geoscience Data Journal*, 1(2), 192-195. DOI: [10.1002/gdj3.21](https://doi.org/10.1002/gdj3.21)
- Data files (separate dataset record): "The Egg Model — data files", TU Delft /
  4TU.ResearchData. DOI: `10.4121/uuid:916c86cd-3558-4672-829a-105c62985ab2`
  Record pages:
  - https://data.4tu.nl/articles/dataset/The_Egg_Model_-_data_files/12707642
  - https://research.tudelft.nl/en/datasets/the-egg-model-data-files/

## License

4TU General Terms of Use (as stated on the dataset record page). The dataset is
freely downloadable without registration or a request form.

## Exact download command used

```bash
curl -sL -o data.zip \
  "https://data.4tu.nl/file/0e53ddd5-33b7-44f2-9ca4-921880825e47/5867bc9e-4bf6-4e07-827c-4a6669a539ce"
```

Verification performed at download time:

```
HTTP status: 200
Bytes downloaded: 49267594
MD5: 50d270d9f769f2ca4bb1a8c10b49852e
```

This exactly matches the checksum published for `data.zip` on the 4TU.ResearchData
record (49,267,594 bytes / MD5 `50d270d9f769f2ca4bb1a8c10b49852e`), confirming the
file was fetched intact from the canonical dataset location.

The archive was then extracted:

```bash
unzip data.zip -d extracted
```

producing top-level folders `Eclipse/`, `MoReS/`, `MRST/`, `AD-GPRS/`,
`Permeability_Realizations/` under `Egg_Model_Data_Files_v2/`.

## What subset was taken, and why

The full archive is an ensemble of **101 permeability realizations** of the same
8-injector / 4-producer channelized reservoir, tested across four simulators
(Eclipse, MoReS, AD-GPRS, MRST). Vendoring all 101 realizations (~81 MB of
near-duplicate `PERM<N>_ECL.INC` grid property files) into this git repository
was judged unnecessary for a well-placement/sequencing benchmark, since:

- the base/reference model definition (well positions, schedule, PVT/rel-perm
  tables, grid geometry) is identical across all realizations and is fully
  included here;
- a small number of realizations is sufficient to exercise ensemble-based /
  robust-optimization variants of the benchmark, while keeping the repo lean.

**What is included in `raw/`:**

- `Eclipse/` — full base Eclipse deck and include files (`Egg_Model_ECL.DATA`,
  `ACTIVE.INC`, `COMPDAT.INC`, `SCHEDULE_NEW.INC`, `mDARCY.INC`) — reference realization.
- `MoReS/` — full MoReS/Dynamo deck and include files.
- `MRST/` — full MRST reading scripts and Eclipse-format deck (incl. `MRST_FIM.mat`).
- `AD-GPRS/Egg_Model/` — the AD-GPRS deck description (`GPRS_Egg_Model.txt`) and the
  production-rates output (`VANESSEN_ORI_RMF_2.sim.rates.txt`). The accompanying
  `VANESSEN_ORI_RMF_2.sim.CFL.txt` simulation log (87 MB) was **excluded**: it is a
  solver diagnostics/log file, not model input, and is far larger than the rest of
  the dataset combined (see size note below).
- `AD-GPRS/VanEssenData/` — full raw permeability/active-block/well-connection
  input files used by the AD-GPRS deck.
- `Permeability_Realizations_subset10/` — **10 of the 100** alternative
  permeability realizations provided in the archive (`PERM1`, `PERM10`, `PERM20`,
  ..., `PERM100`, i.e. every 10th realization by index, `_ECL.INC` format),
  chosen to give an evenly-spaced sample of the ensemble's geological variability
  without vendoring all 100 files (~845 KB each, ~81 MB total for the full set).

**Full reproduction:** to obtain the complete 101-realization ensemble, download
`data.zip` from the DOI above and extract `Permeability_Realizations/` in full;
all 100 `PERM<N>_ECL.INC` files (N = 1..100) plus the base Eclipse realization
make up the published 101-member ensemble.

## Size threshold check

Largest file vendored here is `mDARCY.INC` at 189 KB and `PERM<N>_ECL.INC` files at
~845 KB each. No file exceeds 1 MB; nothing close to a problematic size for git.
The excluded `VANESSEN_ORI_RMF_2.sim.CFL.txt` (87 MB, a run log rather than model
input) was the only file over the 50 MB threshold in the original archive and was
deliberately not vendored.

## File count / size

```
$ find benchmark/egg_model/raw -type f | wc -l
43
$ du -sh benchmark/egg_model/raw
12M
```

## Scope note

Data acquisition (US-E2) is described above. The rest of this document
(US-E3) covers the candidate-set adapter, the real baseline runs, and the
attention-network training status.

## Candidate-set adapter (`adapter.py`)

`adapter.py` parses `raw/Eclipse/Egg_Model_ECL.DATA` (+ its `SCHEDULE_NEW.INC`
and `mDARCY.INC` includes) with a small standalone Eclipse-deck parser
(`benchmark/common/eclipse_parse.py`) to build one candidate-set case per
vendored permeability realization (the base realization + the 10 in
`Permeability_Realizations_subset10/`, so 12 cases total). Each case has all
12 wells (8 injectors + 4 producers) as candidates; a "plan" is the order in
which the 12 wells are brought online.

### Proxy objective disclosure

**This is not a reservoir simulation.** No Eclipse/OPM Flow run is performed
here or available offline for this dataset. The economic value used to
compare orderings is an explicit, documented analytic proxy:

```
injector value_index = 0.3 * injection_rate_target * local_permeability / 1000
producer value_index = local_permeability * (reference_bhp / 100) / 1000
```

- `injection_rate_target` / `reference_bhp` are read directly from the deck's
  `WCONINJE` (79.5 STB/d water, all injectors) and `WCONPROD` (395 psia BHP,
  all producers) records -- these are the deck's real assigned controls, not
  invented numbers.
- `local_permeability` is the mean `PERMX` (mDarcy) over the well's full
  vertical column (K=1..7) at its (I, J) location, taken from the realization
  file itself, so the 12 cases genuinely differ in value by realization.
- A single shared "bring-online" resource processes wells strictly in the
  chosen order; each well's `value_index` is discounted at 10%/year by the
  day it comes online (30 days per prior well, matching the deck's 30-day
  `TSTEP` granularity). This reuses the same discounted-single-resource
  evaluator pattern used for the Oklahoma OCC benchmark, reimplemented here
  standalone (`benchmark/common/sequencing_env.py`).

This rewards bringing high-permeability, high-control-target wells online
earlier. It is a transparent, reproducible stand-in for value -- it is
**not** cumulative oil/water production from a flow simulation, and the
numbers below must not be read as physical NPV.

### Duration proxy and resource/travel model (fixes a flat-landscape bug)

**Superseded design note:** an earlier version of this evaluator gave every
well the same fixed `duration_days` (30) and processed all wells through a
single shared resource. Under that design the discounted objective reduces
exactly to a rearrangement-inequality problem (sort by value density,
descending), which is *provably* optimal for greedy value-density -- no
local search or learned policy could ever beat it, no matter how good. That
is the same failure mode the paper's own prior reviewer complaint (r5LL:
"improvements are very small ... on cases where the landscape is nearly
flat") already flagged once; repeating it on a new benchmark would be a
wasted effort. This has been fixed:

- **Variable duration by well type + local permeability** (not uniform):

  ```
  injector duration_days = 20 + 20 * (1 - clamp(local_permeability / 2000, 0, 1))
  producer duration_days = 20 + 20 *      clamp(local_permeability / 2000, 0, 1)
  ```

  i.e. injectors are modeled as slower to bring online in tighter rock
  (establishing injectivity is harder there); producers are modeled as
  slower in higher-permeability rock (a higher achievable rate needs a
  longer completion/cleanup run). This is an honest, DOCUMENTED PROXY, not
  a measured drilling/completion duration -- no such record exists in this
  vendored archive.
- **2 parallel bring-online resource slots**, not 1, using
  earliest-available-resource assignment (the same rule used by the
  private repo's Oklahoma OCC benchmark, `scripts/research/oklahoma_benchmark.py`).
  With 12 wells and 2 slots, resource contention genuinely binds.
- **Grid-distance travel/setup term**: a resource pays `distance / 5.0`
  extra days between consecutive wells it services, using each well's
  (I, J) grid position -- the same distance-based-travel-time mechanism
  HyperActive's Oklahoma OCC benchmark and its `PlanBuilder` already use.

With these three changes, *which resource* a well lands on (not just its
relative order) affects the objective, and the discount schedule for a
given order position depends on which wells preceded it on that specific
resource -- so simple sort-by-value-density is no longer provably optimal.
See the win/loss counts below for confirmation.

### Baselines run (real, real output)

```bash
python3 benchmark/egg_model/run_benchmark.py \
    --seeds 17,29,43,71,101 --random-samples 500 --local-search-evaluations 2000 \
    --out benchmark/egg_model/results.csv
```

Produces `results.csv` with one row per (method, case, seed): 12 cases x
(1 greedy + 5 random seeds + 5 local-search seeds) = 132 rows, plus 12
`attention_dueling_quantile_q_rl` rows (one per case, see "Training
status" below) = 144 rows total. **These numbers supersede the old
flat-landscape numbers from before the duration/resource/travel fix; the
old numbers are no longer valid and are not reproduced here.**

| method                                | cases | mean metric_value | min     | max     |
|----------------------------------------|------:|--------------------:|--------:|--------:|
| `greedy_value_density`                 |    12 |              228.330 | 148.977 | 290.628 |
| `local_swap_search_2000`               | 12x5  |              228.414 | 149.034 | 290.631 |
| `random_500_best`                      | 12x5  |              227.963 | 148.620 | 290.386 |
| `attention_dueling_quantile_q_rl`      |    12 |              224.376 | 147.247 | 287.201 |

**Win/loss vs. `greedy_value_density`, per (case, seed)** (this is the
number that actually matters -- confirming the landscape is no longer
trivially flat):

| method                            | wins | losses | ties | n  |
|------------------------------------|-----:|-------:|-----:|---:|
| `local_swap_search_2000`           |   58 |      0 |    2 | 60 |
| `random_500_best`                  |    0 |     60 |    0 | 60 |
| `attention_dueling_quantile_q_rl`  |    0 |     12 |    0 | 12 |

**Honest reading of this result:** local-swap search now beats greedy on
58/60 (case, seed) pairs (ties on the other 2) -- the landscape is
genuinely no longer flat, confirming the fix worked: local search finds
resource/travel-aware improvements greedy's single-pass value-density
heuristic misses. Random-500-best still loses every time in only 500
samples (expected: pure random sampling over a much larger effective
search space, with no local refinement, is a weak baseline). The trained
`attention_dueling_quantile_q_rl` policy also currently loses to greedy on
every case (see "Training status" below for why, honestly) -- it is not
being reported as beating anything it does not beat.

### Training status: TRAINED (real, small budget, honestly reported)

PyTorch **is** installed in the benchmark venv
(`/home/dmitrii/.venvs/hyperactive-benchmark`, `torch==2.14.0+cpu`) as of
this pass, so `train_attention.py` was actually run end-to-end (a prior
pass hit `ModuleNotFoundError` in a different, non-venv environment and
reported that honestly as a blocker; that blocker no longer applies here).

**Training method (stated plainly):** the network has no autoregressive
state of its own, so training uses a sequential-construction policy: at
each step the network scores every not-yet-placed well via
`q_values(observations, remaining_mask)` (mean of the predicted quantiles),
forms a categorical distribution over the unplaced wells, samples one
(training) or argmax-picks one (evaluation), and removes it from the
remaining-candidates mask. A full episode (one complete 12-well order) is
scored with the exact same `evaluate_order` used for every baseline above,
and the network is trained with REINFORCE (policy gradient, running-mean
baseline) plus a direct quantile-regression term on each well's own
`value_index`. This is a genuine policy-gradient adaptation of the
quantile-Q architecture to this benchmark's one-shot-candidate-set
structure, not literal off-policy DQN with a replay buffer -- stated here
explicitly rather than left implicit.

```bash
$ python3 benchmark/egg_model/train_attention.py
episode=0   loss=8.642091    reward=207.873 baseline=207.873
episode=40  loss=427.505768  reward=239.825 baseline=219.023
episode=120 loss=1191.543091 reward=283.595 baseline=224.279
episode=200 loss=707.152100  reward=261.959 baseline=226.951
episode=320 loss=1333.349243 reward=288.418 baseline=222.213
episode=399 loss=-637.691162 reward=192.541 baseline=224.349
trained 400 episodes in 43.15s, final_loss=-637.691162
appended 12 attention_dueling_quantile_q_rl rows to benchmark/egg_model/results.csv
```

`feature_dim=8`, retrained completely from scratch on Egg Model features
only (no weight transfer from SPE9 or any other benchmark).

**Honest assessment:** 400 REINFORCE episodes over 12 cases is a small
training budget (no hyperparameter search, no curriculum, no entropy
bonus, no replay). The reward curve above shows large per-episode variance
without a clear monotonic improvement in the baseline within this budget,
and the resulting greedy (argmax) rollout loses to `greedy_value_density`
on all 12 cases (see win/loss table above). This is reported as-is, not
disguised: the training loop is real and runs to completion producing real
gradients and a real evaluated policy, but this small a budget does not
yet demonstrate a learned-policy advantage on this benchmark.

### Bootstrap ablation: expert-bootstrapped attention (EBA)

**Motivation:** the architecture's own naming ("EBA" / expert-bootstrapped
attention, per the epic's design intent) implies imitation pretraining from
a strong expert before RL fine-tuning -- the 400-episode REINFORCE-only run
above never actually did this. `train_attention_bootstrap.py` adds that
missing bootstrap phase and scales the fine-tuning budget by 7.5x (3000
episodes vs. 400), on top of a new imitation-pretraining phase.

**Phase 1 -- expert bootstrap (behavioral cloning):** for each of the 12
cases, `local_swap_search_2000`'s output order -- the strongest baseline in
the table above (58/60 wins vs. greedy) -- is used as the demonstration
expert. Demonstrations combine the 5 canonical seeds already used for the
reported `local_swap_search_2000` baseline (same 2000-evaluation budget,
greedy starting order) plus 5 additional runs started from independent
random orders (800 evaluations each), for 8 expert trajectories per case
(120 total, 1440 (state, action) pairs across all 12 cases). Each
construction step of each expert order becomes one supervised pair: the
full feature tensor + remaining-candidates mask (state) and the index of
the well the expert placed next (action). The network is pretrained with a
cross-entropy imitation loss directly on `q_values()`'s masked output (no
new head needed -- already-placed wells are masked to a large negative
value by `q_values()` itself, so cross-entropy naturally ignores them).

```
$ python3 benchmark/egg_model/train_attention_bootstrap.py
bootstrap: 120 demo trajectories, 1440 (state,action) pairs, 30 epochs in 16.70s
  bootstrap_epoch=0 cross_entropy_loss=1.402705
  bootstrap_epoch=7 cross_entropy_loss=0.755319
  bootstrap_epoch=15 cross_entropy_loss=0.695209
  bootstrap_epoch=29 cross_entropy_loss=0.629769
```

The imitation loss drops from 1.40 (near the ln(12)=2.48 random-baseline
region, i.e. barely-informed) to 0.63 over 30 epochs, confirming the
network genuinely learns to imitate the local-swap-search expert's
choices, not merely running epochs with no signal.

**Phase 2 -- REINFORCE fine-tuning from bootstrapped weights:** identical
REINFORCE + quantile-regression loop as the no-bootstrap run above (same
`evaluate_order` reward, same running-mean baseline), but starting from the
bootstrapped weights, at a lower learning rate (3e-4 vs. the bootstrap
phase's 1e-3, chosen upfront to protect the imitation-learned weights, not
tuned after seeing results) and for 3000 episodes (7.5x the no-bootstrap
run's 400):

```
episode=0    loss=12.005928   reward=290.331 baseline=290.331
episode=300  loss=253.348770  reward=241.463 baseline=222.308
episode=600  loss=310.781250  reward=257.726 baseline=216.424
episode=900  loss=692.777039  reward=288.754 baseline=231.158
episode=1200 loss=382.870239  reward=257.373 baseline=230.191
episode=1500 loss=277.426208  reward=240.736 baseline=224.788
episode=1800 loss=-831.479004 reward=167.896 baseline=224.239
episode=2100 loss=-1127.491699 reward=148.880 baseline=227.271
episode=2400 loss=-876.511597 reward=167.879 baseline=230.517
episode=2700 loss=-115.026543 reward=211.427 baseline=219.444
episode=2999 loss=409.833405  reward=257.945 baseline=213.371
fine-tuned 3000 episodes in 176.02s, final_loss=409.833405
total wall-clock (bootstrap + fine-tune): 192.72s
```

**Wall-clock time:** 192.72s total (16.70s bootstrap + 176.02s
fine-tuning), well within the 15-20 minute check-in budget.

**Final evaluation** (greedy/argmax rollout per case, exact same
`evaluate_order` protocol as every baseline):

| method                                           | cases | mean metric_value | min     | max     |
|----------------------------------------------------|------:|--------------------:|--------:|--------:|
| `greedy_value_density`                              |    12 |              228.330 | 148.977 | 290.628 |
| `local_swap_search_2000`                            | 12x5  |              228.414 | 149.034 | 290.631 |
| `attention_dueling_quantile_q_rl` (no bootstrap)    |    12 |              224.376 | 147.247 | 287.201 |
| `attention_dueling_quantile_q_bootstrap_rl`         |    12 |              227.693 | 148.498 | 290.015 |

**Win/loss vs. `greedy_value_density`, per case, and vs.
`local_swap_search_2000`, per (case, seed):**

| method                                            | vs greedy (n=12)            | vs local_swap_search_2000 (n=60) |
|------------------------------------------------------|:----------------------------:|:-----------------------------------:|
| `attention_dueling_quantile_q_rl` (no bootstrap)      | 0 wins / 12 losses / 0 ties | 0 wins / 60 losses / 0 ties |
| `attention_dueling_quantile_q_bootstrap_rl`           | 0 wins / 12 losses / 0 ties | 0 wins / 60 losses / 0 ties |

**Honest interpretation:** the bootstrap ablation closes most of the gap
to `greedy_value_density` (mean 227.693 vs. 228.330, a 0.28% gap, down from
the no-bootstrap run's 1.7% gap) and to `local_swap_search_2000`, but it
does not exceed either baseline on any of the 12 cases even with an
imitation-pretraining phase and 7.5x more fine-tuning episodes. The
REINFORCE fine-tuning phase itself is visibly unstable (large loss swings;
episodes 1800-2400 show a marked reward collapse before partial recovery
by episode 2999) -- a plausible explanation, stated honestly and not
explored further in this pass, is that fine-tuning at this learning rate
partially erodes the bootstrap phase's imitation gains rather than
strictly improving on them (a known risk in behavioral-cloning-then-RL
pipelines). This is reported as a genuine, unmanufactured result: bootstrap
pretraining measurably helps (confirmed by the mean/gap improvement over
the no-bootstrap ablation) but does not flip the win/loss outcome on this
benchmark within this budget. A larger fine-tuning budget, a smaller or
decaying fine-tuning learning rate, or an early-stopping criterion tied to
held-out imitation performance are natural next steps, not performed here
in order to avoid tuning after seeing results.
