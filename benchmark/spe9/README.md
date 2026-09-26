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

This benchmark is data acquisition only (US-E2). Adapting `SPE9.DATA` into a
HyperActive-style candidate-set / evaluator representation (US-E3) is tracked
separately and has not been implemented yet.
