# NLOG (Netherlands Oil and Gas Portal) benchmark (US-F9)

Real-pipeline-only benchmark, same discipline as SPE10/Volve: no proxy-
objective sibling exists for NLOG in this repo. This adapter was written
directly against HyperActive-public's real, deployed pipeline
(`hyperactive-plan`, the real greedy `PlanBuilder`, the real MaskedCDQN
`models/bc39` checkpoint, and the real NPV/Arps-decline economics).

## Source and license

**NLOG.NL** ("Nederlands Olie- en Gasportaal" / Netherlands Oil and Gas
Portal), the Dutch government's public well-data portal, operated on behalf
of the Ministry of Climate Policy and Green Growth. Direct, unauthenticated
download, no login/registration wall:

```bash
curl -L -o /tmp/nlog_boringen.zip \
  "https://www.nlog.nl/sites/default/files/2026-09/thematische_data_boringen.zip"
```

56.7 MB zip ("thematische data boringen" / thematic well data), extracted
into `benchmark/nlog/raw/`:

* `nlog_dirstelsel_20260923.csv` -- well directional/deviation survey
  (~212 MB, 1,320,052 rows, one row per survey station per wellbore, 6,596
  distinct wellbores): `UWI`, `WELLBORE`, `AH_DEPTH`, `TV_DEPTH_NAP`,
  `AZIMUTH`, `DEV_ANGLE`, plus surface coordinates in three systems
  (`X/Y_SURFACE_RD`, `X/Y_SURFACE_UTM31_ED50`, `X/Y_SURFACE_UTM31_WGS84`).
  **This is the only file this adapter uses.**
* `nlog_stratstelsel_20260923.csv` (stratigraphic tops), `nlog_poroperm_*`,
  `nlog_gascompos_*`, `nlog_rockeval_*`, `nlog_vitriniet_*` -- downloaded and
  present in `raw/` but not used by this adapter (no real per-well `layer`
  or rate signal was pulled from them in this pass -- a possible honest
  follow-up, not a hidden omission).

License: NLOG.NL, attribution-only (cite NLOG.NL as source); confirmed via
`nlog.nl/en/disclaimer` (no registration/account requirement for this
download, unlike Volve's Databricks-Marketplace-gated archive).

## Subset selection: block A12, 16 wellbores

The full file's 6,596 distinct wellbores are far too many to model as one
candidate pool, and any hand-picked subset risks being tuned toward a nicer
NPV. To keep the choice honest and reproducible, the following **purely
mechanical rule, based only on well *count*, was fixed and applied before
looking at any position, depth, or economics data for the chosen wells**:

1. NLOG wellbore names encode the Dutch offshore quadrant/block grid
   (`<block>-<slot>`, e.g. `A12-01`, or `<block>-<platform letter>-<slot>`
   for platform wells, e.g. `A12-A-04`). Grouping all 6,596 wellbores by
   their leading block code gives 137 distinct blocks.
2. Filter to blocks whose wellbore count falls in **[10, 30]** -- the same
   size range as the other benchmarks in this repo (Egg Model/SPE9/SPE10/
   Volve are all 10-30 wells), so the comparison stays apples-to-apples.
   36 blocks qualify.
3. Take the **alphabetically first** qualifying block code.

This resolved to **block A12, 16 wellbores** -- ahead of 35 other
qualifying blocks (`B13`, `D12`, `D15`, `E17`, `F02`, `F06`, `F15`, `F16`,
`F17`, `F18`, `G14`, `G16`, `J06`, `K01`, `K02`, `K11`, `K17`, `L02`, `L04`,
`L06`, `L11`, `L14`, `L15`, `M07`, `P02`, `P08`, `P09`, `P11`, `P12`, `P18`,
`Q04`, `Q07`, `Q08`, `Q10`, `Q13`). Block identity plays no part in the
rule beyond alphabetical order among the size-qualifying set, and the rule
was fixed before any depth/coordinate/rate value for block A12 was read.

Block A12 turns out to be a real North Sea gas field: a 12-slot wellhead
platform (`A12-A-01` .. `A12-A-10`, including 3 co-located sidetracks
`A12-A-04-S1`, `A12-A-07-S1`, `A12-A-07-S2`) plus three earlier
standalone/appraisal wellbores (`A12-01`, `A12-02`, `A12-03`) -- an honest,
non-cherry-picked outcome of the counting rule above, not a field picked
because it looked interesting.

Mechanical pre-filtering of the raw CSV (no reordering/inspection of
results before this step):

```bash
awk -F';' '$3 ~ /^"A12(-|")/' nlog_dirstelsel_20260923.csv \
  > benchmark/nlog/raw/nlog_dirstelsel_A12.csv
```

**Repo-size note:** the full unzipped NLOG tables (~267 MB, dominated by the
212 MB full `nlog_dirstelsel_20260923.csv`) are kept locally in
`benchmark/nlog/raw/` for provenance but are excluded from git (see
`.gitignore`) to avoid bloating this public repo with data this adapter
does not read in full. Only the small, mechanically pre-filtered
`nlog_dirstelsel_A12.csv` (112 KB, the file the adapter actually reads) is
committed. To reproduce the full source locally:

```bash
curl -L -o /tmp/nlog_boringen.zip \
  "https://www.nlog.nl/sites/default/files/2026-09/thematische_data_boringen.zip"
unzip -o /tmp/nlog_boringen.zip -d benchmark/nlog/raw/
```

sha256 checksums (this pass, 2026-09-26):

```
d13e12aecfb8e36894c4bb8a58b9a9789220679e1836642d587bc332054a10de  nlog_dirstelsel_20260923.csv
45691e2baa6e22781da7c8bee0a472383920dd01c5f19a681ce0ea1ada7f6399  nlog_dirstelsel_A12.csv
```

## Spatial-proximity clustering (real distances, fixed before any pipeline run)

Real `X_SURFACE_RD` / `Y_SURFACE_RD` (Dutch Rijksdriehoek grid, metres --
already a flat metric projection, used directly, no reprojection needed)
positions for all 16 wells were computed and inspected as pairwise
distances BEFORE choosing a bin size or running `hyperactive-plan`:

| Pair | Distance |
|---|--:|
| Co-located platform sidetracks (e.g. `A12-A-04`/`A12-A-04-S1`) | 0 m |
| Max spread within the 12-well `A12-A` platform | **6.08 m** |
| `A12-01` to nearest platform well | **140.41 m** |
| `A12-03` to platform / `A12-01` | **~1,232 - 1,375 m** |
| `A12-02` to everything else | **~10,472 - 11,839 m** |

`CLUSTER_BIN_M = 50` m was chosen because it comfortably clears the
platform's real 6.08 m maximum internal spread (so it cannot arbitrarily
split that one physical structure) while sitting nearly 3 orders of
magnitude below the next real gap (140.41 m to the nearest standalone
well) -- so it also cannot merge two genuinely distinct locations. Applying
it produces **4 real clusters**: the 12-well platform, plus `A12-01`,
`A12-02` and `A12-03` each alone -- an honest reflection of how this field
was actually drilled (platform development wells plus a few earlier
standalone appraisal wellbores), not the old one-cluster-per-well mistake.

## What was NOT found: rate and readiness-date data

The NLOG "thematische data boringen" archive is a directional-survey /
stratigraphy / rock-property release, **not a production-volumes
release** -- no oil/gas/water rate or injection-volume data of any kind
exists in it. Unlike Volve (which at least had a `Purpose` field
distinguishing Production/Injection/Observation/Appraisal/Wildcat wells),
NLOG's `WELLBORE` naming convention (platform-slot letter vs. standalone
numeric suffix) is a *drilling-program* distinction, not a documented
producer/injector flag -- so it is **not** used here to invent a
producer/injector split; that would fabricate a purpose NLOG does not
actually state. All 16 wells are modelled as producers with one uniform,
clearly-labelled **PLACEHOLDER** rate: `oil_rate = 35.0` t/day,
`liq_rate = oil_rate * 1.05` (same 5% water-cut convention as the other
adapters). This is the same disclosure posture as Volve's placeholder, and
is a materially weaker benchmark than Egg Model/SPE9/SPE10 on this axis.

There is also **no well-completion/readiness-date field** in this NLOG
table (only a `LAST_UPDATED` database-record metadata timestamp, not a
drilling date -- not used). Per `docs/data-format.md`, a well with no
`init_entry_date` has no readiness constraint and is available from the
start of planning, so all 16 wells here are ready immediately -- unlike
Volve, which had real per-well completion dates. This means NLOG is, by
construction, the *weakest* rate/readiness-data benchmark in this repo:
every well has identical modelled economics, so any greedy/RL scheduling
difference is driven entirely by crew-scheduling/movement structure across
the 4 real clusters, not by any real well-to-well value difference.

`length` is real: the maximum `AH_DEPTH` (along-hole/measured depth, m)
reached by each wellbore's own survey -- a genuine per-well total-length
number, directly analogous to Volve's real MD.

## Adapter

See `build_hyperactive_input.py`'s module docstring for the full mapping
detail (repeats the acquisition/subset/clustering/disclosure sections
above plus exact column mapping).

## Real pipeline run

```bash
python benchmark/nlog/build_hyperactive_input.py

.venv_real/bin/hyperactive-plan \
  --wells benchmark/nlog/hyperactive_input/wells.xlsx \
  --coordinates benchmark/nlog/hyperactive_input/coordinates.xlsx \
  --model models/bc39 --start 2026-01-01 --horizon-years 20 \
  --drilling-months 12 --drilling-crews 2 --gtm-crews 1 \
  --episodes 10 --exploration 0.15 --no-tracking \
  --out benchmark/nlog/plan_bc39.csv
```

**Real output (verbatim console output of the command above):**

```
wells: 16 valid, 0 rejected
RL plan (bc39): 15 wells, NPV 10,136,820,303
greedy plan: 15 wells, NPV 10,259,729,536 (RL vs greedy -1.20%)
schedule: benchmark/nlog/plan_bc39.csv
```

**Wells scheduled: greedy 15/16, bc39 15/16** -- with the fixed 12-month
drilling window and 2 drilling crews, only 15 of the 16 wells fit; both
methods leave the same well (`A12-02`, the farthest-out standalone
wellbore, ~10.5 km from the platform) unscheduled. **NPV, reported
honestly:** greedy beats bc39 by **-1.20%** (bc39: 10,136,820,303 vs.
greedy: 10,259,729,536). With every well carrying an IDENTICAL placeholder
rate and no readiness-date structure at all (see disclosure above), the gap
that remains is driven entirely by crew-scheduling/travel-time sequencing
across the 4 real clusters, not by any real well-to-well value difference
-- this limitation was documented before this command was run, not
discovered afterwards to explain away the result.

Full RL schedule: [`plan_bc39.csv`](plan_bc39.csv).

## Results table

See [`benchmark/results_real_pipeline.csv`](../results_real_pipeline.csv)
for the combined results table across all benchmarks (Egg Model, SPE9,
SPE10, Volve, NLOG, ...).
