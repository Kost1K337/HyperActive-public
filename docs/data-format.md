# Input data format

Two tables describe a well pool: **wells** (required) and **cluster
coordinates** (optional). Both accept CSV or Excel (`.xlsx`/`.xls`/`.xlsm`);
the format is picked from the file extension
(`hyperactive.data.loader.read_table`).

## Wells table

One row per well. Columns are matched by header name (case-insensitive,
whitespace-normalised), in any order; unrecognised columns are ignored.
Accepted header spellings per field:

| Field | Accepted headers | Required | Notes |
|---|---|:-:|---|
| `name` | `well`, `well_name`, `name` | yes | Unique well identifier. |
| `cluster` | `cluster`, `pad` | yes | Matched against the coordinates table by exact string; `25` and `25.0` are treated as the same cluster. |
| `field` | `field` | yes | Free-text grouping, not used by the planner logic. |
| `layer` | `layer`, `reservoir` | yes | Free-text, not used by the planner logic. |
| `well_type` | `well_type` | yes | A `+`-separated task chain, e.g. `ГС+ГРП`. See [Task codes](#task-codes) below. |
| `oil_rate` | `oil_rate`, `oil_rate_tpd` | yes | Initial oil rate, t/day. |
| `liq_rate` | `liq_rate`, `liq_rate_tpd`, `liquid_rate_tpd` | yes | Initial liquid rate, t/day (≥ `oil_rate`). |
| `length` | `length`, `length_m`, `lateral_length_m` | yes | Total well length, m. Missing values are filled with the column mean. |
| `purpose` | `purpose` | no | Free text, not used by the planner logic. |
| `init_entry_date` | `init_entry_date`, `readiness_date` | no | Planned/actual readiness date (any format `pandas.to_datetime` parses). Determines when the well becomes available for scheduling — see [Readiness](#readiness) below. |

A row missing any required field, or whose `name`/`cluster` is empty, is
dropped and counted in `load_wells`'s `rejected` return value rather than
raising an error.

### Task codes

`well_type` is a `+`-joined chain of codes (`hyperactive.core.Task`):

| Code | Task | Crew type | Duration |
|---|---|---|---|
| `ГС` (horizontal well), `ННС` (directional well), `МЗС` (multilateral well), `БУРЕНИЕ` (drilling) | `DRILLING` | drilling crew | 30 days |
| `ГРП` (hydraulic fracturing) | `GTM` | GTM (well-intervention) crew | 20 days |

E.g. `ГС+ГРП` is a horizontal well drilled and then fractured; `ГС` alone is
drilling only. Task durations are fixed constants, not derived from the
input table.

### Readiness

`load_wells(path, readiness_hour=8)` turns `init_entry_date` into
`readiness_date` — the earliest moment a well may be scheduled
(`SimpleInfrastructure.get_ready_date`) — at `readiness_hour:00` of that
calendar day. Pass `readiness_hour=None` to use the exact timestamp from the
table instead (this is what `benchmark/run_benchmark.py` does, matching how the
reference benchmark cells were produced). A well without `init_entry_date`
has no readiness constraint and is available from the start of planning.

## Cluster coordinates table

Optional. Columns `cluster, x, y[, z]` in metres, matched by header name
(`z` defaults to 0 when absent). Used to compute crew move time between
clusters (`hyperactive.planning.team_manager.DistanceTeamMovement`):

```
move_days = min_days_between_clusters + distance / (team_speed_kmh · 1000) / 24
```

Without a coordinates table (`hyperactive.data.load_coordinates(None, wells)`
or a missing file), clusters are placed evenly on a 2.5 km circle purely so a
distance-based movement model still has *some* geometry
(`ring_coordinates`); for production use, always supply real coordinates —
`hyperactive.cli` falls back to a fixed move time (`SimpleTeamMovement`,
1 day within a cluster, 14 days between clusters) when no coordinates file is
given at all.

## Synthetic data

`hyperactive.data.synthetic` generates a well pool with this schema for
experimentation without proprietary field data — see
[`reproducing.md`](reproducing.md#synthetic-data).
