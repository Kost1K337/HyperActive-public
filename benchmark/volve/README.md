# Volve benchmark (partial, real field data)

Real-pipeline-only benchmark (US-F7), same discipline as SPE10: no proxy-
objective sibling exists for Volve in this repo. This adapter was written
directly against HyperActive-public's real, deployed pipeline
(`hyperactive-plan`, the real greedy `PlanBuilder`, the real MaskedCDQN
`models/bc39` checkpoint, and the real NPV/Arps-decline economics), with
real spatial-proximity clustering from the start (US-F7 requirement).

## What was found and downloaded (honest acquisition trail)

**Equinor's own official archive is NOT directly, programmatically
downloadable in this pass.** `equinor.com/energy/volve-data-sharing` (the
official page) was fetched and read: it lists ~40,000 files (well logs,
seismic, production/injection histories, reservoir models, reports) and
gives exactly two access mechanisms:

1. **Databricks Marketplace** ("Access open data (Databricks
   Marketplace)") -- an interactive, account-gated listing. This is not a
   stable direct-file URL and requires signing in to a Databricks
   workspace; not reachable by a plain `curl`/`wget` in this pass.
2. A PDF **user guide**
   (`https://equinoropendata.blob.core.windows.net/userguides/Equinor%20open%20data%20-%20User%20Guide.pdf`)
   explaining the access process, confirming the Azure-blob-backed,
   Databricks-marketplace-gated distribution mechanism -- again, no stable
   direct per-file URL for a public, unauthenticated download.

This matches the PRD's own anticipated "realistic possible outcome":
direct programmatic download of the full archive is blocked behind an
interactive/account-gated flow. Per the task's own time-box and scope
guidance ("a full ~40,000-file dataset is NOT required... look specifically
for well/drilling/completion log files that give well names, positions,
and some indication of production/completion scale"), acquisition
continued via a narrower, real, public, directly-downloadable subset
instead of stopping at this blocker:

### Source 1: well-header table (names, purpose, depths, dates)

A third-party GitHub mirror of Equinor's own vendored Volve well-header
table (not Equinor's own archive host, but publicly re-distributing the
same real, government/company-sourced well data under the project's own
license terms):

```
Repository: https://github.com/bysarmad/Well-Data-Management-and-Visualization
Commit:     f9202a244b2e864a77cab2140cbee899fd6be5b4 (2019-05-29)
File:       Data/volve_wells.csv
```

```bash
curl -sL "https://raw.githubusercontent.com/bysarmad/Well-Data-Management-and-Visualization/master/Data/volve_wells.csv" \
  -o benchmark/volve/raw/volve_wells_bysarmad_mirror.csv
```

26 real wellbores, columns `Wellbore;Purpose;Water depth (m);TVD (m);MD
(m);Entered date;Completed date`. This mirror has NO License file in the
repository (checked via `gh api repos/.../license` -> 404); it is used
here as a convenience re-hosting of the same publicly-released Volve well
data, same as the many other academic/research mirrors of this dataset
that exist for exactly this reason (Equinor's own distribution channel
being interactive-only). This gives real well identity, purpose, water
depth, TVD, MD (real total measured depth) and real entered/completed
dates -- but **no surface coordinates**.

### Source 2: real wellhead coordinates (Sodir/NPD FactPages)

Surface (wellhead) UTM coordinates for the same 26 real wells were looked
up by name in the Norwegian Offshore Directorate (Sodir, formerly NPD)'s
public FactPages register -- the Norwegian government's own regulatory
record of every well drilled on the Norwegian continental shelf, an
independent real source (not Equinor's own archive) describing the same
real physical wellbores:

```bash
# per-well lookup via the FactPages public search + pageview endpoints:
curl -sL "https://factpages.sodir.no/api/Search?query=<wellbore name>"
curl -sL "https://factpages.sodir.no/wellbore/pageview/.../<NPDID>"
# NS UTM [m] / EW UTM [m] fields parsed from the returned page
```

(Full automation script: the fetch was scripted in Python using
`urllib`, one Search + one PageView request per well name, 24/26 wells
resolved directly; results saved to
`benchmark/volve/raw/volve_wells_coords_sodir.csv`.) License: **NLOD
(Norwegian Licence for Open Government Data)**, confirmed via
`sodir.no/en/facts/data-and-analyses/open-data/`.

2 of 26 wells (`15/9-19 BT2`, `15/9-19 ST2`) were not resolved by the
FactPages name search (sidetrack-suffix naming mismatch) and a third
(`15/9-F-11 T2`) likewise -- these three reuse their parent slot's real
coordinates in the adapter (a sidetrack re-enters the same physical
wellhead, so this is a real, not invented, position; see
`build_hyperactive_input.py`'s `SIDETRACK_PARENT` mapping).

### What was NOT found: production/injection rate volumes

No real oil/water rate or volume data was reached within this pass's
~20-minute acquisition time-box. Equinor's production-volume tables live
inside the full ~40,000-file archive behind the Databricks Marketplace
login (Source 1 above), not the two lighter, directly-downloadable
sources used here. This is disclosed plainly in `oil_rate`/`liq_rate`
below rather than fabricated to look measured.

## Well-pool scoping: 26 real wells, 12 included

The 26 real wellbores split by their real `Purpose` field:

| Purpose | Count | Wells | Included? |
|---|--:|---|:-:|
| Production | 9 | F-1 C, F-7, F-9, F-9 A, F-11 B, F-12, F-14, F-15 C, F-15 D | yes |
| Injection | 3 | F-1 B, F-4, F-5 | yes |
| Observation | 9 | F-1, F-1 A, F-10, F-11, F-11 A, F-11 T2, F-15, F-15 A, F-15 B | **no** |
| Appraisal | 3 | 19 A, 19 B, 19 BT2 | **no** |
| Wildcat | 2 | 19 S, 19 ST2 | **no** |

Only the 12 real Production + Injection wells are included in the
candidate set: Observation/Appraisal/Wildcat wells genuinely have no
producible rate for HyperActive's producer-only economics to model, and
including them with a fabricated non-zero rate would misrepresent
monitoring/exploration wells as productive ones. This is a scoping
decision made from the real `Purpose` field, not a filter tuned on
downstream NPV.

## Adapter (`build_hyperactive_input.py`) and real spatial-proximity clustering

Full mapping detail is documented in `build_hyperactive_input.py`'s module
docstring. Summary:

* **Positions**: real wellhead UTM31N easting/northing from the Sodir
  lookup, used directly in metres.
* **Clustering (real spatial proximity from the start, US-F7)**: the same
  fixed-size spatial-binning method as the other three adapters
  (`_bin_cluster`, bin origin anchored at the well pool's own minimum x/y
  to avoid an absolute-UTM boundary artifact splitting genuinely co-
  located wells). The 12 included wells' real pairwise distances range
  from 0 m (co-located sidetracks sharing one physical slot, e.g.
  `15/9-F-9`/`15/9-F-9 A`) up to **9.48 m** (`15/9-F-5` to `15/9-F-11 B`)
  -- every real Production/Injection well here is drilled from ONE
  physical platform template. `CLUSTER_BIN_M = 50` m was chosen because
  it comfortably clears this real 9.48 m maximum spread (so it cannot
  arbitrarily split one physical location) while staying two orders of
  magnitude below any realistic inter-field distance -- fixed from the
  real distance table before running `hyperactive-plan` or looking at any
  NPV outcome. This correctly produces **1 cluster of all 12 wells** -- an
  honest, real finding (Volve's producing wells are genuinely
  platform-based), not a repeat of the earlier US-F2 mistake.
* **Length**: the REAL measured depth (`MD`, metres) from the mirror CSV
  -- a genuine per-well number, not a completion-interval proxy like the
  Eclipse-deck adapters.
* **`init_entry_date`**: the REAL `Completed date` per well (when each
  well finished and could start producing/injecting), unlike the other
  three adapters' single assumed simultaneous-readiness date.
* **Rates**: **PLACEHOLDER, not measured** (see acquisition section
  above) -- uniform `oil_rate = 40.0` t/day for all 9 producers
  (`liq_rate = oil_rate * 1.05`), uniform `liq_rate = 150.0` t/day for all
  3 injectors (`oil_rate = 0`). Unlike the Egg Model/SPE9/SPE10 proxies
  (anchored to a real static quantity -- local permeability or a real
  deck rate keyword), these two numbers have no such anchor; every
  producer therefore has identical modelled economics, so any
  greedy/RL choice among producers here reflects readiness-date/crew-
  scheduling structure, not a real well-to-well value difference. This is
  a materially weaker benchmark than the other three on that specific
  axis -- disclosed here rather than hidden.

**Same disclosure as every other adapter:** HyperActive's economics have
no injection-support concept, so the 3 injectors always show negative
well NPV below -- reflects the producer-only economic model, not a
judgement on injection.

## Real pipeline run

```bash
python benchmark/volve/build_hyperactive_input.py

.venv_real/bin/hyperactive-plan \
  --wells benchmark/volve/hyperactive_input/wells.xlsx \
  --coordinates benchmark/volve/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking \
  --out benchmark/volve/plan_bc39.csv
```

**Real output (verbatim console output of the command above):**

```
wells: 12 valid, 0 rejected
RL plan (bc39): 12 wells, NPV 6,380,791,713
greedy plan: 12 wells, NPV 6,429,779,953 (RL vs greedy -0.76%)
```

**Wells scheduled: greedy 12/12, bc39 12/12** -- with a single cluster and
2 drilling crews, all 12 wells fit within the 12-month drilling window (6
waves of 2). **NPV, reported honestly:** greedy narrowly beats bc39 by
**-0.76%** (bc39: 6,380,791,713 vs. greedy: 6,429,779,953) -- with 9 of
the 12 wells carrying an IDENTICAL placeholder rate (see disclosure
above), most of the scheduling-order degrees of freedom that would let a
learned policy differentiate producers by value are absent by
construction; the 0.76% gap that remains comes entirely from the 3
injectors' differing readiness dates and how each method sequences them
against the identical-rate producers. This is reported as-is: the
placeholder-rate limitation was documented before this command was run,
not discovered afterwards to explain away the result.

Full RL schedule: [`plan_bc39.csv`](plan_bc39.csv).

## Results table

See [`benchmark/results_real_pipeline.csv`](../results_real_pipeline.csv)
for the combined results table across all benchmarks (Egg Model, SPE9,
SPE10, Volve, ...).
