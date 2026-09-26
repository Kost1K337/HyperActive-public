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

This benchmark is data acquisition only (US-E2). Adapting these files into a
HyperActive-style candidate-set / evaluator representation (US-E3) is tracked
separately and has not been implemented yet.
