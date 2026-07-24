# SysGF maintenance and integration

This repository is the Germany-specific research application. Shared model
changes should originate in the SysGF PyPSA-Eur fork; German data, scenario
construction, reporting and policy assumptions stay here.

## Branch policy

- `main`: last promoted, reproducible research baseline.
- `integration/sysgf-research-YYYY-MM`: tested candidate for the next baseline.
- `feat/<topic>`: one bounded change, merged into the integration branch.
- `sync/pypsa-eur-<commit>`: temporary transfer branch for a pinned Eur commit.

Record the exact PyPSA-Eur commit with a merge commit. A tree-preserving merge
is appropriate when DE already contains newer ports of the shared code; apply
the missing feature patches afterward and test them in the DE architecture.

## Consolidated 2026-07 line

The integration line combines the validated DE layered-PTES feature branch with
the consolidated SysGF PyPSA-Eur history. It includes:

- the existing DE scenario/data/reporting stack and DH-subnode implementation;
- energy-only and layered PTES, potential limits and feature-matrix evaluation;
- native-EPSG:4326 river/lake/seawater processing while retaining DE's robust
  empty/inland seawater handling;
- both ATES research paths described below;
- executable SysGF district-heating reporting helpers and lint-clean plot scripts;
- synchronized data-version ordering and focused regression tests.

## ATES data sources

The public legacy model remains available and is the default:

```yaml
sector:
  district_heating:
    ates:
      enable: true
      data_source: bgr
```

The colleague's HI-grid model is selected explicitly:

```yaml
sector:
  district_heating:
    ates:
      enable: true
      data_source: hi_grid
      data_file: data/ates/HI_ATES_grid.gpkg
  heat_sources:
    urban central:
    - ptes
    - ates
```

See `config/examples/config.ates_hi_grid.yaml`. The licensed file is not stored
in Git. It must contain `CAPEX_EUR_MW`, `VERLUSTRATE`, and geometry. The
provider confirmed inverse-CAPEX regional weighting and the conversion
`standing_loss = 1 - VERLUSTRATE ** (1 / 8760)`. The supplied dummy GeoPackage
has 353,152 valid EPSG:25832 cells and is suitable for schema/performance smoke
tests; its constant values do not test spatial variation.

## District-heating subnode selection

`sector.district_heating.subnodes.countries` limits the countries in which
explicit subnodes are attached. If `demand_share` is set, the code first applies
that country filter and then selects the largest systems until the configured
share of model-aware DH demand *inside that subset* is covered. A regression
test with a much larger excluded-country demand protects this denominator
ordering. The older `dh-subnodes` branch can therefore be archived once this
integration line is published.

## Test ladder

Cheap checks:

```bash
pixi run ruff check scripts test
pixi run pytest test \
  --ignore=test/test_base_network.py \
  --ignore=test/test_build_shapes.py
pixi run snakemake --list-rules --cores 1
pixi run snakemake --list-rules --cores 1 \
  --configfile config/examples/config.ates_hi_grid.yaml
pixi run snakemake -n --cores 1 \
  --configfile config/test/config.integration_dag.yaml
```

`test_base_network.py` and `test_build_shapes.py` download Natural Earth data;
run the complete `pixi run pytest test` in a connected environment.

The committed integration DAG config deliberately disables Ariadne scenario
management. Whenever `run.name`, the configured manual scenario YAML or its
scenario definitions change, regenerate the DE production file before running
the normal DAG:

```bash
pixi run snakemake --cores 1 --force build_scenarios
```

`build_scenarios` is the single DE production entry point. It reads
`run.scenarios.manual_file`, retrieves/uses the configured Ariadne/IIASA data,
and writes `run.scenarios.file` (by default
`config/scenarios.automated.yaml`). The upstream generic `create_scenarios`
rule is intentionally not exposed in DE because it does not add those data.

For workflow promotion, run the existing small DE build and the PTES feature
matrix, then add one BGR-ATES and one licensed HI-grid-ATES scenario. Record
config, commit, solver, objective and output-network path.

## Data and workflow gates

- Production approval for the licensed HI dataset; the dummy file is not
  research input.
- Local census-area input when census-based DH geometry refinement is enabled.
- Copernicus Marine credentials for primary seawater retrieval, supplied as
  `COPERNICUSMARINE_USERNAME` and `COPERNICUSMARINE_PASSWORD` via the shell or
  an untracked repository-root `.env` file. The default archive source does
  not require them.
- External downloads and a solver for full end-to-end builds.
- Review of the temperature-blindness study's stage-1/stage-2 production runs;
  the committed scenario/DAG checks do not replace those solves.

## Promotion checklist

1. Clean integration worktree.
2. Lint, focused unit tests and workflow parsing pass.
3. Full network-enabled tests pass.
4. Representative DE builds and the feature matrix solve successfully.
5. Output reports are reviewed for energy balance and objective regressions.
6. Merge the candidate into `main`, tag it, and record the paired Eur commit.
