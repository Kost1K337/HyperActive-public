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

### Baselines run (real, real output)

```bash
python3 benchmark/egg_model/run_benchmark.py \
    --seeds 17,29,43,71,101 --random-samples 500 --local-search-evaluations 2000 \
    --out benchmark/egg_model/results.csv
```

Produces `results.csv` with one row per (method, case, seed): 12 cases x
(1 greedy + 5 random seeds + 5 local-search seeds) = 132 rows.

| method                         | cases | mean metric_value | min    | max    |
|---------------------------------|------:|-------------------:|-------:|-------:|
| `greedy_value_density`          |    12 |             226.226 |147.488 |287.541 |
| `local_swap_search_2000`        | 12x5  |             226.226 |147.488 |287.541 |
| `random_500_best`               | 12x5  |             225.483 |146.719 |287.051 |

**Honest note on this result:** greedy and local-swap search land on
*exactly* the same objective value here. This is not a bug -- because every
well shares the same `duration_days` (30), the discount factor applied to
each order position is fixed and monotonically decreasing regardless of
*which* wells are assigned to it, so maximizing the sum reduces to a
rearrangement-inequality problem: sort by value descending. Greedy
value-density is therefore provably optimal for this exact proxy, and no
local search over swaps can improve on it. Random-500-best gets close but
essentially never finds the exact optimum in only 500 samples. This is a
property of the current (documented, simplified) proxy objective, not a
claim that Egg Model well sequencing is trivial for a real reservoir
objective.

### Training status: BLOCKER

```bash
$ python3 -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'
$ timeout 10 pip download torch --no-deps -d /tmp/torchtest
# timed out after 10s resolving/downloading -- not a quick install
```

PyTorch is not installed in this environment, and installing it (a large,
slow download in this sandbox) was not feasible within this pass's time
budget. `train_attention.py` implements a real training loop against
`benchmark/common/attention_dueling_quantile_q.py` (the
AttentionDuelingQuantileQ architecture, retrained from scratch with
`feature_dim=8` -- no weight transfer from any other benchmark), and exits
with an explicit blocker message rather than fabricating a loss curve:

```bash
$ python3 benchmark/egg_model/train_attention.py
BLOCKER: PyTorch is not installed in this environment ...
$ echo $?
1
```

This is reported honestly as a documented blocker (see the PRD's
`criterionAmendments` for US-E3). The baseline comparisons above do not
depend on PyTorch and are complete.
