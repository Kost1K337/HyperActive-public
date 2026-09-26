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
- The same single-shared-resource, 10%/year-discounted evaluator as the Egg
  Model is used (`benchmark/common/sequencing_env.py`), with each well
  occupying the resource for 10 days (the deck's initial `TSTEP`
  granularity) before the next well comes online.

This is a transparent proxy for value, not simulated cumulative
production; the numbers below are not physical NPV.

### Baselines run (real, real output)

```bash
python3 benchmark/spe9/run_benchmark.py \
    --seeds 17,29,43,71,101 --random-samples 500 --local-search-evaluations 2000 \
    --out benchmark/spe9/results.csv
```

Produces `results.csv` with one row per (method, seed): 1 greedy + 5 random
seeds + 5 local-search seeds = 11 rows.

| method                         | n | mean metric_value | min     | max     |
|---------------------------------|--:|--------------------:|--------:|--------:|
| `greedy_value_density`          | 1 |             174.311 | 174.311 | 174.311 |
| `local_swap_search_2000`        | 5 |             174.311 | 174.311 | 174.311 |
| `random_500_best`               | 5 |             173.623 | 173.296 | 173.839 |

As with the Egg Model, greedy and local-swap search land on the same value:
since every well shares the same `duration_days` (10), the discount factor
per order position is fixed, so the optimal order is exactly "sort by
`value_index` descending" (a rearrangement-inequality result) -- greedy
value-density already produces that order, so local search cannot improve
on it. Random-500-best comes close but does not reliably find the exact
optimum in 500 samples.

### Training status: BLOCKER

Identical situation to the Egg Model: PyTorch is not installed in this
environment and installing it was not feasible within this pass's time
budget.

```bash
$ python3 -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'
$ python3 benchmark/spe9/train_attention.py
BLOCKER: PyTorch is not installed in this environment ...
$ echo $?
1
```

`train_attention.py` implements a real training loop against
`benchmark/common/attention_dueling_quantile_q.py` with `feature_dim=10`
(retrained from scratch, no weight transfer from the Egg Model or any other
benchmark) but cannot execute without PyTorch. This is reported honestly as
a documented blocker (see the PRD's `criterionAmendments` for US-E3). The
baseline comparisons above do not depend on PyTorch and are complete.
