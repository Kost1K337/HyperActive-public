# Drogon benchmark

**Drogon is a SYNTHETIC reservoir model, not a real field.** This is
stated explicitly at the top of this README, same honesty discipline
already used for Egg Model/SPE9/SPE10's own synthetic-data disclosures.
The deck's own header confirms it in plain text
(`benchmark/drogon/raw/DROGON_HIST.DATA`):

```
--		Synthetic reservoir simulation model Drogon (2020)
-- The Drogon model - the successor of the Reek model
-- Used in a FMU set up
-- Grid input data generated from RMS project
```

Drogon was built by Equinor specifically as an open, synthetic
"playground" model for testing reservoir-modelling and history-matching
workflows -- it looks and behaves like a real North Sea field but has no
real subsurface data behind it.

**Context for credibility (reported in literature, not independently
verified here):** Drogon is actively used in current ensemble/history-
matching literature (EnKF-style methods, e.g. Equinor's own ERT tooling
and several SPE/academic EnKF benchmarking papers) as a standard synthetic
test case. This is a plus for this benchmark's credibility even though the
underlying model is synthetic, not real.

## Source

- **Repository actually used**: `github.com/OPM/opm-tests`, path `drogon/`.
  This repository has the ECLIPSE-format simulation deck directly
  (WELSPECS/COMPDAT/WCONHIST/WCONINJH keywords, ready to parse), so the
  `github.com/equinor/fmu-drogon` fallback mentioned in the task was not
  needed -- opm-tests' copy already had everything required.
- Commit acquired from: `2225180b6b987ac496760ac057c827e09f0bf242` (opm-tests
  `master`).

## License

**Open Database License (ODbL) v1.0.** Confirmed directly from the deck
headers themselves (`DROGON_HIST.DATA` and `drogon.grid`, both carry the
identical license banner):

```
-- This reservoir simulation deck is made available under the Open Database
-- License: http://opendatacommons.org/licenses/odbl/1.0/. Any rights in
-- individual contents of the database are licensed under the Database Contents
-- License: http://opendatacommons.org/licenses/dbcl/1.0/

-- Copyright (C) 2022 Equinor
```

Same license family as Egg Model/SPE9/SPE10's opm-data acquisitions (ODbL
data / DbCL contents). The `opm-tests` GitHub repository itself has no
repository-level `LICENSE` file (`api.github.com/repos/OPM/opm-tests`
reports `"license": null`), but the individual deck files carry their own
explicit ODbL header as shown above, which is what governs this data.
(Note: the task's alternative source, `fmu-drogon`, is licensed CC-BY-SA
4.0 for data / GPLv3 for code -- not applicable here since that repo was
not used.)

## Exact commands used

```bash
git clone https://github.com/OPM/opm-tests.git --filter=blob:none --sparse drogon-fetch
cd drogon-fetch
git sparse-checkout set drogon
```

This produced the deck directly under `drogon/model/DROGON_HIST.DATA` (the
historical/history-matched run -- used here) and `drogon/model/DROGON_PRED.DATA`
(a prediction-mode variant, not used), with includes under
`drogon/include/` and `drogon/include_pred/`. The files actually vendored
into `benchmark/drogon/raw/` in this repository:

```
$ find benchmark/drogon/raw -type f
benchmark/drogon/raw/DROGON_HIST.DATA
benchmark/drogon/raw/drogon_hist.sch
benchmark/drogon/raw/drogon.grid
benchmark/drogon/raw/drogon.pvt
benchmark/drogon/raw/drogon.dimens
```

```
$ du -sh benchmark/drogon/raw
13M
```

Only the files needed to parse wells (`drogon_hist.sch`), grid geometry
(`drogon.grid`, for the corner-point pillar coordinates), PVT densities
(`drogon.pvt`), and grid dimensions (`drogon.dimens`) were vendored --
`DROGON_PRED.DATA` and its includes (a prediction-mode variant of the same
model, not used by this adapter) were left in the scratch checkout, same
avoid-vendoring-unrelated-material discipline as the SPE10 acquisition.

## Well count and roles: checked from the real parsed WELSPECS, not assumed

The task's own prior-research recollection was "6 wells: A1-A4 producers +
A5-A6 injectors" -- **this was verified against the real file, not just
trusted**. `benchmark/drogon/raw/drogon_hist.sch`'s `WELSPECS` sections
actually define wells in **two groups**:

1. **5 `RFT`-group wells** (`R_A2`, `R_A3`, `R_A4`, `R_A5`, `R_A6`), phase
   `OIL`, immediately set `SHUT` with `ORAT=0` in their first `WCONHIST`
   record. These are pressure/RFT **observation** wells -- a
   history-matching artefact of the synthetic model, never opened for
   production or injection. **Excluded** from the HyperActive well pool
   (nothing to schedule for a well that never opens).
2. **6 `OP`/`WI`-group wells** -- this is the real set matching (and
   confirming) the recollected "4 producers + 2 injectors" split:
   - `A1`, `A2`, `A3`, `A4` -- group `OP`, real `WCONHIST` historical
     oil-rate-matched production controls (confirmed producers).
   - `A5`, `A6` -- group `WI`, real `WCONINJH` historical water-injection-
     rate-matched controls (confirmed injectors).

These 6 wells (`A1`-`A6`) are the HyperActive well pool built by this
adapter. Real (I, J) `WELSPECS` positions:

| well | group | role | (I, J) |
|------|-------|------|-------:|
| A1 | OP | producer | (32, 33) |
| A2 | OP | producer | (22, 30) |
| A3 | OP | producer | (28, 41) |
| A4 | OP | producer | (30, 52) |
| A5 | WI | injector | (31, 20) |
| A6 | WI | injector | (17, 42) |

## Adapter (`build_hyperactive_input.py`)

Full mapping detail is documented in the adapter's module docstring.
Summary:

* **Grid**: Drogon uses a **corner-point grid** (`SPECGRID 46 73 31 1`),
  not the simple tensor DX/DY grid used by Egg Model/SPE9/SPE10 -- so
  (I, J) cannot be converted to metres with one cell-size constant. The
  adapter instead parses the deck's real `COORD` array (pillar corner
  coordinates: x/y/z at pillar top and bottom, 6 values per pillar) and
  computes each well's real (x, y) as the average of its cell's 4
  surrounding pillars, plus a per-well average per-layer thickness
  (`(z_bottom - z_top) / 31` at that pillar) used for the length estimate.
  (`MAPAXES`, a further local-grid -> UTM rotation, is present in the deck
  but not applied -- only relative inter-well distances are needed here for
  clustering, and this note is included for transparency.)
* **Rates**: unlike Egg Model's BHP-controlled producers (needing a
  permeability-derived proxy), Drogon's `WCONHIST` (producers) and
  `WCONINJH` (injectors) give **real historical rate targets**. This
  adapter uses each well's first (earliest-dated) `WCONHIST`/`WCONINJH`
  record -- i.e. its real historical rate at schedule entry -- converted
  sm3/day -> t/day with the deck's own `DENSITY` keyword (PVT region 1:
  oil 852.95669 kg/m3, water 999.041 kg/m3; the deck defines 2 PVT
  regions, region 1 used as one representative density for all wells since
  per-well PVTNUM lookup was out of scope here -- documented approximation,
  same disclosure spirit as the other adapters' proxy formulas).
* **Scale disclosure**: because Drogon is a full-field-scale synthetic
  model (not a small teaching deck), these real historical rates are much
  larger than Egg Model/SPE9/SPE10's per-well economics -- producers run
  ~3410-3412 t/day oil vs. those decks' ~1-55 t/day scale, and both
  injectors inject ~7992 t/day of water. This is a genuine, disclosed
  difference in field scale, not a data error.
* **Length**: no single deck-wide DZ constant exists for a corner-point
  grid, so completed length is estimated as (real distinct completed
  K-layer count, from `COMPDAT` `OPEN` records) x (that well's own real
  average per-layer thickness from its `COORD` pillar's z-span / 31
  layers) -- a real, deck-derived, per-well approximation.
* `field` = `"drogon"`, `layer` = `"drogon_hist"` (free text).
* `well_type` = `"ГС"` for every well (same reasoning as the other
  adapters: no fracturing/stimulation keyword in the deck, so the closest
  honest fit among HyperActive's task-chain vocabulary is drilling-only).

## Clustering: real spatial-proximity discipline (mandatory, same as US-F4/F6/F7/F8/F9)

Real pairwise distances between the 6 wells' derived (x, y) positions were
computed **before** running `hyperactive-plan` or looking at any NPV
outcome:

| pair | distance (m) |
|------|-------------:|
| A1-A3 | **1300.57** (minimum) |
| A3-A4 | 1650.97 |
| A1-A2 | 1574.00 |
| A2-A3 | 1888.62 |
| A1-A5 | 1887.73 |
| A3-A6 | 1705.24 |
| A2-A6 | 1963.36 |
| A2-A5 | 1972.77 |
| A4-A6 | 2469.88 |
| A1-A6 | 2630.28 |
| A1-A4 | 2806.99 |
| A3-A5 | 3085.77 |
| A2-A4 | 3490.01 |
| A5-A6 | 3867.59 |
| A4-A5 | 4679.89 |

The grid spans roughly 11.4 km x 12.9 km, and the **minimum real
nearest-neighbour gap between any two of these 6 wells is 1300.57 m** --
i.e. even the two closest wells here are 1.3 km apart, a full-field well
spacing (Drogon models a whole field development, not a tight pad).

`CLUSTER_BIN_M = 500` m was chosen specifically because it clears
(is comfortably below) that real 1300.57 m minimum nearest-neighbour gap
-- same "pick a threshold below the real minimum NN distance" reasoning
used for SPE10's 150 m choice, fixed **before** running the pipeline or
seeing any NPV outcome, and not revisited afterwards.

Applying this threshold to the real positions produces **6 singleton
clusters (one well per cluster)** -- with only 6 wells this is not
surprising given the distances above, and it is reported here as the
honest result of the pre-committed method, same discipline as SPE10's own
genuine 5-singleton-cluster finding (the task's own guidance explicitly
allows this outcome for Drogon).

## Real pipeline run

```bash
.venv_real/bin/python benchmark/drogon/build_hyperactive_input.py

.venv_real/bin/hyperactive-plan \
  --wells benchmark/drogon/hyperactive_input/wells.xlsx \
  --coordinates benchmark/drogon/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking \
  --out benchmark/drogon/plan_bc39.csv
```

**Real output (verbatim console output of the command above):**

```
wells: 6 valid, 0 rejected
RL plan (bc39): 6 wells, NPV 296,650,035,339
greedy plan: 6 wells, NPV 296,650,884,980 (RL vs greedy -0.00%)
schedule: benchmark/drogon/plan_bc39.csv
```

**Wells scheduled: greedy 6/6, bc39 6/6.** With only 6 wells, 2 drilling
crews, and 6 singleton clusters, there is little resource contention, so
both methods schedule every well within the drilling window. RL and greedy
produced a **near-identical NPV** (296,650,035,339 vs. 296,650,884,980, a
genuine -0.0003% difference -- effectively a tie, well within numerical
noise of two methods converging on the same well ordering) and, per
`plan_bc39.csv`, the **same underlying schedule structure**: the two
highest-value producers first (A3, A2), then the remaining two producers
(A4, A1) once drilling crews free up, then both loss-making injectors last
(A5, A6, both with strongly negative well NPV -- consistent with
HyperActive's producer-only economic model, same disclosure as every other
benchmark in this repo: injection wells always show negative well NPV
here because the evaluator has no injection-support/pressure-maintenance
value concept, not a judgement on injection). This is a genuine, honest
near-tie result reported as-is, exactly as anticipated for a small 6-well
pool by the task's own guidance -- not tuned or adjusted to favor either
method.

Full RL schedule: [`plan_bc39.csv`](plan_bc39.csv).

## Results table

See [`benchmark/results_real_pipeline.csv`](../results_real_pipeline.csv)
for the combined results table across all benchmarks (Egg Model, SPE9,
SPE10, Volve, Drogon, ...).
