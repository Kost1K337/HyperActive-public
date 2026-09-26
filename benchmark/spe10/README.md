# SPE10 benchmark (10th SPE Comparative Solution Project, Model 2)

This is a real-pipeline-only benchmark (US-F6): unlike Egg Model and SPE9,
there is no earlier proxy-objective adapter for SPE10 in this repo. This
adapter was written directly against HyperActive-public's real, deployed
pipeline (`hyperactive-plan`, the real greedy `PlanBuilder`, the real
MaskedCDQN `models/bc39` checkpoint, and the real NPV/Arps-decline
economics), with real spatial-proximity clustering from the start (no
one-cluster-per-well mapping mistake to fix here, per US-F6).

## Source

- Official SPE benchmark: "Tenth SPE Comparative Solution Project: A
  Comparison of Upscaling Techniques" by Mike Christie (Heriot-Watt
  University) and Martin Blunt (Imperial College), SPE, August 2001.
  Model 2: a 60 x 220 x 85 (1,122,000-cell) heterogeneous permeability
  field, the standard fine-grid upscaling-comparison case.
- Open input files: `github.com/OPM/opm-data`, path `spe10model2/` (also
  mirrored/used by `github.com/OPM/opm-tests`; there is also a smaller,
  unrelated `spe10model1/` in the same repo which was not used here).
  Repository: https://github.com/OPM/opm-data
  Commit acquired from: `eaa2261683a97027e057c2bc49612ad1c86390b3` (opm-data
  `master`, authored 2023-05-05 by Atgeirr Flø Rasmussen -- same commit as
  the SPE9 acquisition, since both live in the same upstream repo).

## License

**Open Database License (ODbL) v1.0.** Confirmed both from the opm-data
repository root `README.md` (same license statement as documented in
`benchmark/spe9/README.md`) and directly from the deck header itself
(`SPE10_MODEL2.DATA`):

> This reservoir simulation is made available under the open Database
> License: http://opendatacommons.org/licenses/odbl/1.0/
> Individual contents of the database are licensed under the Database
> contents License: http://opendatacommons.org/licenses/dbcl/1.0/
> Copyright(C) 2015 Statoil

Full license text: https://github.com/OPM/opm-data/blob/master/odbl-10.txt
(database content license: https://github.com/OPM/opm-data/blob/master/dbcl-10.txt)

## Exact commands used

```bash
cd /tmp && git init spe10-fetch && cd spe10-fetch
git remote add origin https://github.com/OPM/opm-data.git
git config core.sparseCheckout true
echo "spe10/*" > .git/info/sparse-checkout
git pull --depth 1 origin master
```

**Correction found during acquisition:** the sparse-checkout pattern
`spe10/*` (by analogy with SPE9's `spe9/*`) matched nothing -- the upstream
repository actually stores this benchmark under `spe10model1/` and
`spe10model2/`, not `spe10/`. Confirmed with
`git ls-tree -r origin/master --name-only | grep -i spe10`, which listed
`spe10model1/*` (4 files) and `spe10model2/*` (4 files). The
sparse-checkout pattern was corrected to:

```bash
printf "spe10model1/*\nspe10model2/*\n" > .git/info/sparse-checkout
git read-tree -mu HEAD
```

Only `spe10model2/` (Model 2, the standard fine-grid case used by this
benchmark) was then copied into `benchmark/spe10/raw/` in this repository;
`spe10model1/` (a smaller, less commonly used variant) was left in the
scratch checkout and not vendored here, keeping the same avoid-vendoring-
unrelated-benchmarks discipline as the SPE9 acquisition.

## File listing

```
$ find benchmark/spe10/raw -type f
benchmark/spe10/raw/SPE10_MODEL2.DATA
benchmark/spe10/raw/SPE10MODEL2_PERM.INC
benchmark/spe10/raw/SPE10MODEL2_PHI.INC
benchmark/spe10/raw/SPE10MODEL2_TOPS.INC
```

```
$ ls -la benchmark/spe10/raw
     4725  SPE10_MODEL2.DATA
 57946924  SPE10MODEL2_PERM.INC
 19287206  SPE10MODEL2_PHI.INC
     1028  SPE10MODEL2_TOPS.INC

$ du -sh benchmark/spe10/raw
74M
```

4 files, 74 MB total -- much larger than Egg Model/SPE9's ~1 MB decks,
because SPE10's defining feature is a very fine, per-cell heterogeneous
permeability (`SPE10MODEL2_PERM.INC`) and porosity
(`SPE10MODEL2_PHI.INC`) field over 1,122,000 cells; both files are still
well under GitHub's 100 MB single-file limit.

## Well count: checked from the real parsed WELSPECS, not assumed

Per the task's own note that SPE10 Model 2's well count is much smaller
than its 1.1M grid cells, the real `WELSPECS`/`COMPDAT` sections of
`SPE10_MODEL2.DATA` were parsed with the same shared parser used for Egg
Model/SPE9 (`benchmark/common/eclipse_parse.py`). Result: **5 wells** -- 1
injector (`INJ`) and 4 producers (`P1`..`P4`). This is confirmed by
`WELLDIMS 5 340 1 5 /` in `RUNSPEC` (a hard cap of 5 wells) and matches
the well-known SPE10 pattern: one injector at the field centre, (I, J) =
(30, 110), and one producer at each of the four grid corners, (1,1),
(60,1), (60,220), (1,220) -- a coarse waterflood pattern sampling a very
fine, very heterogeneous permeability field. SPE10 Model 2 is a
permeability-heterogeneity/upscaling benchmark, not a well-count
benchmark; 5 wells is the real, correct answer for this deck, not a
parsing shortcut.

COMPDAT confirms every well opens the full vertical column, K=1..85 (all
85 layers), consistent with a benchmark whose complexity is in the
permeability field, not the well completions.

## Adapter (`build_hyperactive_input.py`) and real spatial-proximity clustering

Full mapping detail (grid-to-metre conversion, permeability lookup, rate
modelling, and the clustering method/threshold with real nearest-neighbour
distances measured *before* running the pipeline) is documented in
`build_hyperactive_input.py`'s module docstring. Summary:

* **Positions**: real (I, J) from `WELSPECS`, converted to metres with the
  deck's real cell size (`DX = 20 ft`, `DY = 10 ft`, FIELD units).
* **Clustering (real spatial proximity from the start, US-F6)**: the same
  fixed-size spatial-binning method as the Egg Model/SPE9 adapters
  (`_bin_cluster`), with `CLUSTER_BIN_M = 150` m. The real pairwise
  distances between these 5 wells range from 359.66 m (the two closest
  pairs, P1-P2 and P3-P4) up to 758.24 m -- i.e. the *closest* two wells in
  the whole deck are still 360 m apart. 150 m was chosen specifically
  because it clears (is comfortably below) that real 359.66 m minimum
  nearest-neighbour gap, guaranteeing no genuinely distant wells are ever
  merged, while being a plausible cluster radius at this field's own scale
  (under half the field's shorter side, 366 m). Applying this threshold
  correctly produces **5 singleton clusters** here -- this reflects the
  real, physical isolation of these 5 wells (a corners-plus-centre
  five-spot layout across a 366 m x 671 m field), not a repeat of the
  earlier US-F2 one-cluster-per-well mistake (which mapped wells that
  really were close together, e.g. Egg Model's 13.6-18.4-cell spacing or
  SPE9's 1.4-4.1-cell diagonal chain, into artificial singleton clusters).
  The 150 m threshold was fixed from the real distance table above before
  `hyperactive-plan` was run or any NPV outcome was seen, and was not
  revisited afterwards.
* **Rates**: the injector's `liq_rate` is the deck's real `WCONINJE` RESV
  target (5000 rb/day, converted with the deck's real water density);
  producers are BHP-controlled with no rate target in the deck at all (as
  in the Egg Model), so their `oil_rate` uses the same
  local-permeability-derived proxy formula and ceiling as the Egg Model
  adapter (`55 * clamp(local_PERMX / 300, 0, 1)` t/day), with the 300 mD
  normaliser chosen from this deck's real 122.35-271.18 mD producer-local-
  permeability range.
* **Length**: real, deck-derived completion interval (`85 * DZ = 85 * 2 ft
  = 51.816 m` for every well, since COMPDAT opens the full K=1..85 column
  for all 5 wells).

**Same disclosure as Egg Model/SPE9:** HyperActive's economics have no
injection-support concept, so `INJ` always shows negative well NPV in the
results below -- this reflects the producer-only economic model, not a
judgement on injection.

## Real pipeline run

```bash
python benchmark/spe10/build_hyperactive_input.py

.venv_real/bin/hyperactive-plan \
  --wells benchmark/spe10/hyperactive_input/wells.xlsx \
  --coordinates benchmark/spe10/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking \
  --out benchmark/spe10/plan_bc39.csv
```

**Real output (verbatim console output of the command above):**

```
wells: 5 valid, 0 rejected
RL plan (bc39): 5 wells, NPV 1,764,158,272
greedy plan: 5 wells, NPV 1,764,158,272 (RL vs greedy +0.00%)
```

**Wells scheduled: greedy 5/5, bc39 5/5** -- with only 5 wells, 2 drilling
crews, and a 12-month drilling window, there is no resource contention at
all (2 crews clear 5 wells in 3 drilling waves), so both methods trivially
schedule every well. RL and greedy also produced an **identical NPV and
identical well order** (visible in `plan_bc39.csv`: P3, P1, P4, P2, INJ) --
with a well pool this small (4 producers with different but non-competing
permeability-proxy rates, 1 injector) there is little scheduling-order
degree of freedom left for a policy to differentiate on once every well
fits within the horizon regardless of order; both methods converge on
scheduling the higher-value producers as early as crew availability
allows, then the loss-making injector last (as expected: greedy's
value-first ordering and bc39's learned policy plainly agree that INJ,
with real NPV -698,245,880, should be scheduled last).

Full RL schedule: [`plan_bc39.csv`](plan_bc39.csv).

## Results table

See [`benchmark/results_real_pipeline.csv`](../results_real_pipeline.csv)
for the combined results table across all benchmarks (Egg Model, SPE9,
SPE10, ...).


## Baseline fairness correction (bug-fix pass)

The earlier numbers in this README (US-F6: bc39 0.00% vs greedy) are kept as recorded; this
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
  --wells benchmark/spe10/hyperactive_input/wells.xlsx \
  --coordinates benchmark/spe10/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking --greedy-penalty both \
  --out benchmark/spe10/plan_bc39.csv
```

```
wells: 5 valid, 0 rejected
RL plan (bc39): 5 wells, NPV 1,764,158,272
greedy plan (NPV only): 5 wells, NPV 1,764,158,272 (RL vs greedy +0.00%)
greedy plan (NPV - drill_team_penalty): 5 wells, NPV 1,764,158,272 (RL vs greedy +0.00%)
```

| | wells | NPV |
|---|---:|---:|
| bc39 | 5 | 1,764,158,272 |
| greedy, NPV only (PRIMARY) | 5 | 1,764,158,272 |
| greedy, NPV - drill_team_penalty (secondary, previously the only baseline) | 5 | 1,764,158,272 |

bc39 vs primary greedy: **+0.0000%**; vs penalised greedy:
+0.0000% (previously reported: 0.00%).

No change: 5 wells, 2 crews, no contention; all three plans are identical.

Feature-scale check (disclosed limitation of bc39 on out-of-range inputs,
see the overview): 0.0% of candidate rows clipped (15 rows) -- not affected.
