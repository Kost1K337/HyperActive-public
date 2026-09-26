# Norne field benchmark (real field data, real production/injection history)

Real-pipeline-only benchmark (US-F8), same discipline as SPE10/Volve: no
proxy-objective sibling exists for Norne in this repo. This adapter is
written directly against HyperActive-public's real, deployed pipeline
(`hyperactive-plan`, the real greedy `PlanBuilder`, the real MaskedCDQN
`models/bc39` checkpoint, and the real NPV/Arps-decline economics), with
real spatial-proximity clustering from the start.

## Source and acquisition

[OPM's `opm-data` GitHub repository](https://github.com/OPM/opm-data),
`norne/` subfolder -- the same public, real-field, community-maintained
Eclipse-deck archive used for SPE9/SPE10 in this repo.

```bash
git clone --filter=blob:none --sparse https://github.com/OPM/opm-data.git /tmp/opm-data-norne-checkout
cd /tmp/opm-data-norne-checkout
git sparse-checkout set norne
```

License: **ODbL 1.0** (Open Database License) / **DbCL 1.0** (Database
Contents License) -- `odbl-10.txt` / `dbcl-10.txt` at the `opm-data` repo
root (copied into `raw/`), and restated as a file-header comment inside the
grid/schedule files themselves. Same license as the SPE9/SPE10 benchmarks.

Files copied into `raw/` (only what this adapter needs, not the full
~170 MB `norne/` tree):

* `NORNE_ATW2013.DATA` -- the main deck (confirms `METRIC` units, lists
  every `INCLUDE`d file, `SPECGRID 46 112 22`).
* `README` -- upstream notes (history-match citation, contributor credit).
* `INCLUDE/BC0407_HIST01122006.SCH` -- the real, history-matched well
  schedule (`WELSPECS`, `COMPDAT`, `WCONHIST`, `WCONINJE` records).
* `INCLUDE/GRID/IRAP_1005.GRDECL` -- the real corner-point grid geometry
  (`SPECGRID`/`COORD`/`ZCORN`), used below for real well X/Y and
  completion length.
* `INCLUDE/PVT/PVT-WET-GAS.INC` -- the real `DENSITY` keyword, used to
  convert historical rates from sm3/day to t/day.
* `odbl-10.txt`, `dbcl-10.txt` -- the license texts.

Multi-megabyte static property arrays (PERM/PORO/NTG/FIPNUM/...), fault
multipliers, VFP lift-curve tables, and relperm/SCAL tables were left out
of `raw/` -- not needed to build `wells.xlsx`/`coordinates.xlsx`.

## Well count (checked from the real parsed WELSPECS, not assumed)

Parsing every `WELSPECS` block in `BC0407_HIST01122006.SCH` with
`benchmark/common/eclipse_parse.parse_well_records` returns **36 unique
well names** (`B-1AH` .. `K-3H`):

```
B-1AH B-1BH B-1H B-2H B-3H B-4AH B-4BH B-4DH B-4H
C-1H C-2H C-3H C-4AH C-4H
D-1CH D-1H D-2H D-3AH D-3BH D-3H D-4AH D-4H
E-1H E-2AH E-2H E-3AH E-3BH E-3CH E-3H E-4AH E-4H
F-1H F-2H F-3H F-4H
K-3H
```

This resolves the ambiguity flagged in the task before acquisition
("~7 wells in a commonly-used reduced version" vs "~36 over full field
life"): the reduced ~7/~22-well subsets used by some RL papers are
hand-picked training-friendly slices of this same field; the real,
unfiltered `opm-data` schedule deck used here genuinely defines all 36
wells drilled over the field's full history, and that is the number used
here (no filtering applied).

Role split (from real `WCONHIST`/`WCONINJE` presence):

* **27 pure producers** -- appear only in `WCONHIST`.
* **1 pure injector** (`C-2H`) -- appears only in `WCONINJE`.
* **8 producer-to-injector conversions** (`C-1H`, `C-3H`, `C-4H`, `C-4AH`,
  `F-1H`, `F-2H`, `F-3H`, `F-4H`) -- appear in BOTH: a real, documented
  Norne operating practice (a well drilled/completed as an oil producer,
  later converted to water/gas injection for pressure support). These are
  classified as **injectors** in `wells.xlsx` (terminal/dominant real role
  in this schedule).

## Real well geometry (corner-point grid, not a proxy)

Norne uses a **corner-point grid** (`SPECGRID` 46 x 112 x 22), not a
Cartesian `DX`/`DY`/`DZ` grid like SPE10, so there is no single constant
cell size to multiply (I, J) by. `build_hyperactive_input.py` reads the
deck's own `COORD`/`ZCORN` arrays directly:

* **X/Y (UTM metres)**: each well's (I, J) cell-centre X/Y is the mean of
  its 4 bounding pillars' top-face X/Y from `COORD`.
* **Completion length (metres)**: for each well's real `COMPDAT` K-range
  (K1..K2), the mean top-face Z at layer K1 and mean bottom-face Z at
  layer K2 are read from `ZCORN`, and `length = bottom_z - top_z` -- the
  real local gross vertical interval the well's completions span (not an
  assumed per-layer thickness). This slightly overstates true
  wellbore-completion length for wells with non-contiguous COMPDAT layers
  (spans min-to-max K rather than summing only completed layers), the same
  conservative convention the SPE10 adapter uses for its own K1..K2 span.

## Spatial clustering (mandatory real-distance-first discipline)

Real pairwise UTM distances between all 36 wells' computed X/Y were
computed **before** running any pipeline or looking at any NPV result.
Several wells share the *exact same* surface location (0.0 m apart) --
real multilateral sidetracks from one physical pad/slot:

| pad group | wells |
|---|---|
| 1 | B-1AH, B-1BH |
| 2 | B-4AH, B-4BH |
| 3 | D-3AH, D-3BH, D-3H |
| 4 | E-2AH, E-2H |
| 5 | E-3BH, E-3CH |

Beyond those exact-pad groups, the closest two **distinct** pads are
**136.3 m** apart (`B-4DH` - `K-3H`). Full nearest-neighbour distance
distribution (36 wells): `0.0` (x7 co-located pairs/triples), then
`136.3, 203.7, 248.7, 255.4, 280.9, 294.1, 307.2, 307.8, 470.7, 569.2,
573.3, 582.9, 660.8, 662.6, 748.4, 910.5, 1635.8` m.

`CLUSTER_BIN_M = 100` m was chosen from this real distribution:
comfortably below the 136.3 m minimum distinct-pad gap (so no two
genuinely separate pads are ever merged), while still merging every
0 m-apart same-pad sidetrack group into one cluster (binning is exact at
0 m regardless of bin size, so this choice only affects the distinct-pad
boundary, which is what was checked). This threshold was fixed **before**
`hyperactive-plan` was run or any NPV outcome was seen, and was not
revisited afterwards.

Applying fixed-size spatial binning directly to the real X/Y (metres, same
`_bin_cluster` method as the Egg Model/SPE9/SPE10 adapters, adapted to bin
directly in metres since real UTM X/Y are already available here) yields
**30 clusters from 36 wells** (25 singletons + 5 multi-well pads matching
the table above).

## Rates: real history-matched data (not a proxy)

Unlike every prior adapter in this repo, Norne's `WCONHIST` records
genuinely carry per-period, per-well measured `ORAT`/`WRAT` (oil/water
rate, sm3/day, METRIC units) from the field's real production history:

* **Producers**: `oil_rate` = mean of every real `ORAT` value across the
  well's full `WCONHIST` history in this schedule, converted sm3/day ->
  t/day with the deck's real oil density (`DENSITY` keyword,
  `INCLUDE/PVT/PVT-WET-GAS.INC`: 859.5 kg/m3 = 0.8595 t/m3, first PVT
  region). `liq_rate = oil_rate + mean(WRAT) converted with the deck's
  real water density` (1033.0 kg/m3 = 1.033 t/m3) -- a **real measured
  water cut**, not the 5% assumption used in the Egg Model/SPE9/SPE10
  adapters (no historical water-rate data existed there). 5 of the 27
  producers (`B-1AH`, `B-4AH`, `D-3H`, `E-3BH`, `E-4H`) have only a
  single, zero-valued `WCONHIST` record in this schedule window -- their
  real mean rate is honestly `0.0` t/day, reported as-is, not backfilled.
* **Injectors**: `oil_rate = 0` (no oil production; same
  injector/producer economic-model mismatch disclosure as every prior
  adapter). `liq_rate` = mean of the well's real `WCONINJE` **WATER**-type
  rate records, converted with the real water density. Gas-injection
  volumes (present alongside water injection for `C-1H`, `C-3H`, `C-4H`,
  `C-4AH`) are **not** included -- HyperActive's schema/economics have no
  gas-injection concept, same producer/injector-only disclosure as prior
  adapters.

## Proxy / schema-fit disclosure

Norne's oil AND water rates are both real, measured, per-well historical
values -- there is no permeability-derived or assumed-water-cut proxy for
the producers here, unlike every earlier benchmark in this repo. The only
proxy-like choices are (a) dropping gas-injection volumes (a schema/
economics limitation, not a data-availability one) and (b) 5 producers
whose real history in this schedule window is genuinely near-zero. As with
every prior adapter, injector liquid volume is a pure lift cost with no
oil revenue in HyperActive's economics, so injectors show negative well
NPV in the results below -- this reflects HyperActive's producer-only
economic model, not a judgement on injection.

`well_type = "ГС"` for every well (same reasoning as prior adapters: no
fracturing/stimulation keyword in the deck, so the closest honest fit
among HyperActive's task-chain vocabulary is drilling-only; Norne's wells
are in fact mostly horizontal/multilateral, consistent with this choice).

## Real pipeline run

```bash
python benchmark/norne/build_hyperactive_input.py

.venv_real/bin/hyperactive-plan \
  --wells benchmark/norne/hyperactive_input/wells.xlsx \
  --coordinates benchmark/norne/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking \
  --out benchmark/norne/plan_bc39.csv
```

**Real output (verbatim console output of the command above):**

```
wells: 36 valid, 0 rejected
RL plan (bc39): 8 wells, NPV 401,689,185,673
greedy plan: 8 wells, NPV 440,744,130,579 (RL vs greedy -8.86%)
```

**Wells scheduled: greedy 8/36, bc39 8/36** -- with a 36-well pool but only
2 drilling crews and a 12-month drilling window per crew slot, only 8
wells clear the 20-year horizon before running out of scheduling capacity
for both methods; both schedule the same 8 wells (`B-2H`, `E-1H`, `D-1H`,
`D-4H`, `D-3BH`, `B-1H`, `D-3AH`, `D-3H` -- see `plan_bc39.csv`) but in a
different order/pairing across the 2 crews, which changes the launch-date
sequencing enough to produce a real NPV gap.

**Result: bc39 loses to greedy by -8.86% on this benchmark.** This is
reported honestly, per the task's own instruction not to tune parameters
after seeing the outcome -- no adapter or pipeline parameter was changed
after this run.

Full RL schedule: [`plan_bc39.csv`](plan_bc39.csv).

## Results table

See [`benchmark/results_real_pipeline.csv`](../results_real_pipeline.csv)
for the combined results table across all benchmarks (Egg Model, SPE9,
SPE10, case_10, Volve, NLOG, Drogon, Norne, ...).


## Baseline fairness correction (bug-fix pass)

The earlier numbers in this README (US-F8: bc39 -8.86% vs greedy) are kept as recorded; this
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
  --wells benchmark/norne/hyperactive_input/wells.xlsx \
  --coordinates benchmark/norne/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking --greedy-penalty both \
  --out benchmark/norne/plan_bc39.csv
```

```
wells: 36 valid, 0 rejected
RL plan (bc39): 8 wells, NPV 401,689,185,673
greedy plan (NPV only): 8 wells, NPV 440,744,130,579 (RL vs greedy -8.86%)
greedy plan (NPV - drill_team_penalty): 8 wells, NPV 440,744,130,579 (RL vs greedy -8.86%)
```

| | wells | NPV |
|---|---:|---:|
| bc39 | 8 | 401,689,185,673 |
| greedy, NPV only (PRIMARY) | 8 | 440,744,130,579 |
| greedy, NPV - drill_team_penalty (secondary, previously the only baseline) | 8 | 440,744,130,579 |

bc39 vs primary greedy: **-8.86%**; vs penalised greedy:
-8.86% (previously reported: -8.86%).

No change: both greedy rules produce the same plan here; bc39 still loses by
8.86%.

**Disclosed limitation: feature-scale saturation in the bc39 checkpoint.**
`src/hyperactive/env/features.py` feeds `oil_rate`, `liq_rate`, `cost_npv`,
`cash_flow`, `capex` and `travel_cost` to the network as absolute values;
bc39's VecNormalize statistics were fit on its training distribution and
clip normalised features at +/-10 (`clip_obs=10.0`). On this benchmark
**96.1% of candidate rows** have at least one feature at the clip bound
(row-0 rollout, 51 valid candidate rows; per feature: oil_rate 96%, liq_rate 96%, cost_npv 96%, cash_flow 96%, job_window_share 4%). On the
clipped features the network cannot tell those candidates apart, so bc39 is
here being evaluated outside the input range it was trained on. This is
disclosed, not corrected: no retraining and no inference-time rescaling was
applied, because either would change the model after the fact without
proper training and validation.
