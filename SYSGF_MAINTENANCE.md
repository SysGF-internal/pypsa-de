# SysGF maintenance and integration

This fork is the reusable European model layer for SysGF research. Keep
PyPSA-DE-specific data, rules and scenario reporting in the PyPSA-DE repository.

## Branch policy

- `master`: last promoted, reproducible research baseline.
- `integration/sysgf-research-YYYY-MM`: tested candidate for the next baseline.
- `feat/<topic>`: one bounded change, merged into the integration branch.
- `sync/upstream-<release>`: temporary upstream-release integration branch.

Pin upstream work to a PyPSA-Eur release. Do not merge a moving upstream
`master` directly into the research baseline. Preserve merge commits for
external PRs so their provenance remains visible.

## Consolidated 2026-07 line

The integration line is based on the SysGF `dev` history and contains:

- the layered and energy-only PTES models, their validation and scenario files;
- SysGF-TUB PR #1, with river/lake processing retained in native EPSG:4326;
- SysGF-TUB PR #2's PTES simplification, inherited through PR #3;
- SysGF-TUB PR #3's HI-grid ATES aggregation and network model, alongside the
  retained public BGR potential model;
- schema-safe config generation and focused regression tests.

ATES is disabled by default. The public BGR pathway remains the default source:

```yaml
sector:
  district_heating:
    ates:
      enable: true
      data_source: bgr
```

The separately licensed HI-grid pathway requires:

```yaml
sector:
  district_heating:
    ates:
      enable: true
      data_source: hi_grid
      data_file: /path/to/licensed/HI_ATES_grid.gpkg
  heat_sources:
    urban central:
    - ptes
    - ates
```

The GeoPackage must provide `CAPEX_EUR_MW`, `VERLUSTRATE`, and geometry. It is
not stored in Git. HI confirmed the regional inverse-CAPEX weighting and the
conversion `standing_loss = 1 - VERLUSTRATE ** (1 / 8760)`.

## Test ladder

Run cheap checks first:

```bash
pixi run generate-config
pixi run ruff check scripts test
pixi run pytest test/test_config_schema.py \
  test/test_data_versions_layer.py \
  test/test_ates_config.py \
  test/test_ates_heat_source.py \
  test/test_build_ates_potentials.py \
  test/test_solve_network_tes_guards.py \
  test/test_surface_water_heat_approximator.py \
  test/test_build_cop_profiles.py
pixi run snakemake --list-rules --cores 1
```

Then run `pixi run pytest test`. Some upstream tests download Natural Earth
fixtures and therefore require network access.

Before promotion, run at least one small end-to-end sector scenario for:

1. the default PTES setup;
2. energy-only PTES;
3. layered PTES;
4. river and lake heat;
5. HI-grid ATES with an approved data file.

Record the config, commit, solver, objective, and output-network path for every
promotion test.

## Known integration gates

- Obtain production approval for the licensed HI GeoPackage; the supplied dummy
  file is suitable only for schema and pipeline checks.
- Keep the newer DH-subnode implementation in PyPSA-DE. It filters the selected
  countries before resolving the explicitly represented cumulative demand
  share, so the older divergent `dh-subnodes` branch can be archived.
- Treat the older `layered-ptes-energy-exact`, `add-river-*`, `add-seawater-*`,
  and `create-example-data` branches as archival unless a missing behavior is
  demonstrated by a test.
- Upgrade to a newer upstream release only on a dedicated sync branch, with
  the test ladder above and a PyPSA-DE compatibility run.

## Promotion checklist

1. Clean worktree and generated config/schema in sync.
2. Focused tests and lint pass.
3. Full offline-capable unit suite passes.
4. Network-dependent tests pass in a connected environment.
5. Representative workflow outputs are archived and reviewed.
6. Fast-forward or merge the integration candidate into `master`.
7. Tag the research baseline and update PyPSA-DE to record that exact commit.
