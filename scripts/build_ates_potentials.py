# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calculate per-region Aquifer Thermal Energy Storage (ATES) parameters.

Two explicitly selectable sources are supported:

- ``bgr`` retains the public BGR aquifer-area potential model. Suitable aquifer
  area inside buffered district-heating areas is converted to MWh using the
  region's mean district-heating forward/return temperature difference.
- ``hi_grid`` aggregates the separately licensed HI grid onto the model's
  onshore regions using inverse-CAPEX weights.

The HI provider confirmed that ``CAPEX_EUR_MW`` and ``VERLUSTRATE`` are the
relevant columns, that lower-CAPEX cells should receive higher weight, and that
``VERLUSTRATE`` is converted to an hourly retained fraction with the 8760th
root. Its output contains annualised ``capital_cost``,
``hourly_standing_losses`` and ``feasible``. The BGR output retains the legacy
``ates_potential`` energy-capacity limit.
"""

import logging

import geopandas as gpd
import pandas as pd
import xarray as xr

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 8760


def annual_retention_to_hourly_standing_loss(
    annual_retention: pd.Series,
) -> pd.Series:
    """
    Convert an annual retained-energy fraction to hourly self-discharge.

    The HI data provider confirmed that ``VERLUSTRATE`` is to be converted by
    taking its 8760th root. It is therefore used here as the annually retained
    energy fraction; PyPSA's ``standing_loss`` is one minus the hourly retained
    fraction.
    """
    return 1 - annual_retention ** (1 / HOURS_PER_YEAR)


def aggregate_ates_to_regions(
    ates: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    capex_col: str = "CAPEX_EUR_MW",
    loss_col: str = "VERLUSTRATE",
) -> pd.DataFrame:
    """
    Inverse-CAPEX-weighted mean of CAPEX and loss rate per onshore region.

    Each ATES grid cell is assigned to the region that contains it
    (point-in-polygon on the cell's representative point). Cells with lower CAPEX
    receive a higher weight (``weight = 1 / capex``). Regions without any cell
    become NaN rows.

    Parameters
    ----------
    ates : geopandas.GeoDataFrame
        Gridded ATES data with ``capex_col`` and ``loss_col`` columns.
    regions : geopandas.GeoDataFrame
        Onshore (sub)region shapes with a ``name`` column.
    capex_col, loss_col : str
        Column names of the per-cell CAPEX and annual loss factor.

    Returns
    -------
    pandas.DataFrame
        Indexed by region ``name`` with columns ``capex_col``, ``loss_col`` and
        ``n_cells``; NaN for regions without any cell.
    """
    if (ates[capex_col] <= 0).any():
        raise ValueError(f"{capex_col} must be positive for inverse-CAPEX weighting")
    if not ates[loss_col].between(0, 1).all():
        raise ValueError(f"{loss_col} must contain annual retention factors in [0, 1]")

    cells = ates.to_crs(regions.crs).copy()
    cells["geometry"] = cells.geometry.representative_point()
    joined = gpd.sjoin(
        cells,
        regions[["name", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Inverse-CAPEX weighting: lower capex -> higher weight.
    joined["weight"] = 1.0 / joined[capex_col]
    df = pd.DataFrame(joined.drop(columns="geometry"))

    def _weighted_average(g):
        w = g["weight"]
        return pd.Series(
            {
                capex_col: (g[capex_col] * w).sum() / w.sum(),
                loss_col: (g[loss_col] * w).sum() / w.sum(),
                "n_cells": len(g),
            }
        )

    agg = df.groupby("name", sort=False).apply(_weighted_average, include_groups=False)
    return agg.reindex(regions["name"])


def aggregate_bgr_ates_potential(
    aquifers: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    dh_areas: gpd.GeoDataFrame,
    temperature_difference: pd.Series,
    *,
    suitable_aquifer_types: list[str],
    aquifer_volumetric_heat_capacity: float,
    fraction_of_aquifer_area_available: float,
    effective_screen_length: float,
    dh_area_buffer: float,
) -> pd.Series:
    """Convert suitable BGR aquifer area in DH regions to storage MWh."""
    if "AQUIF_NAME" not in aquifers:
        raise KeyError("BGR aquifer data must contain an 'AQUIF_NAME' column")

    regions = regions.to_crs("EPSG:3035")
    aquifers = aquifers.to_crs(regions.crs).copy()
    dh_areas = dh_areas.to_crs(regions.crs).copy()
    aquifers.geometry = aquifers.geometry.make_valid()
    suitable = aquifers[aquifers["AQUIF_NAME"].isin(suitable_aquifer_types)]

    result = pd.Series(0.0, index=regions["name"], name="ates_potential")
    if suitable.empty or dh_areas.empty:
        return result

    # Union first so overlapping DH buffers cannot count aquifer area twice.
    buffered_dh = gpd.GeoDataFrame(
        geometry=[dh_areas.geometry.buffer(dh_area_buffer).union_all()],
        crs=regions.crs,
    )
    aquifers_by_region = gpd.overlay(
        suitable[["geometry"]],
        regions[["name", "geometry"]],
        how="intersection",
    )
    eligible = gpd.overlay(aquifers_by_region, buffered_dh, how="intersection")
    if eligible.empty:
        return result

    area_by_region = eligible.groupby("name").geometry.apply(lambda s: s.area.sum())
    delta_t = temperature_difference.reindex(result.index).clip(lower=0).fillna(0.0)
    # kJ/(m³ K) * m * K = kJ/m²; 3.6e6 kJ = 1 MWh.
    mwh_per_m2 = (
        aquifer_volumetric_heat_capacity
        * fraction_of_aquifer_area_available
        * effective_screen_length
        * delta_t
        / 3.6e6
    )
    result.loc[area_by_region.index] = (
        area_by_region * mwh_per_m2.reindex(area_by_region.index)
    )
    return result


def _mean_temperature_difference(forward_file: str, return_file: str) -> pd.Series:
    """Return the temporal-mean forward-minus-return temperature per region."""
    forward = xr.open_dataarray(forward_file)
    return_ = xr.open_dataarray(return_file)
    try:
        time_dims = [d for d in ("time", "snapshot") if d in forward.dims]
        forward_mean = forward.mean(dim=time_dims).to_series()
        time_dims = [d for d in ("time", "snapshot") if d in return_.dims]
        return_mean = return_.mean(dim=time_dims).to_series()
        return (forward_mean - return_mean).clip(lower=0)
    finally:
        forward.close()
        return_.close()


def build_bgr_potentials(snakemake) -> None:
    """Build the public BGR energy-capacity potential table."""
    regions = gpd.read_file(snakemake.input.regions_onshore)
    aquifers = gpd.read_file(snakemake.input.aquifer_shapes_shp)
    aquifers.geometry = aquifers.geometry.make_valid()
    dh_areas = gpd.read_file(snakemake.input.dh_areas)

    all_aquifers = gpd.overlay(
        aquifers.to_crs("EPSG:3035")[["geometry"]],
        regions.to_crs("EPSG:3035")[["name", "geometry"]],
        how="intersection",
    )
    covered = set(all_aquifers["name"])
    missing = set(regions["name"]) - covered
    if missing:
        message = f"Regions without BGR aquifer coverage: {sorted(missing)}"
        if not snakemake.params.ignore_missing_regions:
            raise ValueError(
                message
                + ". Set sector.district_heating.ates.ignore_missing_regions: true "
                "to assign zero potential."
            )
        logger.warning(message)

    delta_t = _mean_temperature_difference(
        snakemake.input.central_heating_forward_temperature_profiles,
        snakemake.input.central_heating_return_temperature_profiles,
    )
    potential = aggregate_bgr_ates_potential(
        aquifers,
        regions,
        dh_areas,
        delta_t,
        suitable_aquifer_types=snakemake.params.suitable_aquifer_types,
        aquifer_volumetric_heat_capacity=snakemake.params.aquifer_volumetric_heat_capacity,
        fraction_of_aquifer_area_available=snakemake.params.fraction_of_aquifer_area_available,
        effective_screen_length=snakemake.params.effective_screen_length,
        dh_area_buffer=snakemake.params.dh_area_buffer,
    )
    potential.to_frame().to_csv(snakemake.output.ates_potentials, index_label="name")


def build_hi_grid_potentials(snakemake) -> None:
    """Build annualised regional cost/loss inputs from the licensed HI grid."""
    from scripts.add_electricity import calculate_annuity

    regions = gpd.read_file(snakemake.input.regions_onshore)
    ates_grid = gpd.read_file(snakemake.input.ates_grid)[
        ["CAPEX_EUR_MW", "VERLUSTRATE", "geometry"]
    ]
    ates_by_region = aggregate_ates_to_regions(ates_grid, regions)
    feasible = ates_by_region["n_cells"].notna() & (ates_by_region["n_cells"] > 0)
    logger.info("ATES feasible in %s of %s regions", int(feasible.sum()), len(feasible))
    annuity = calculate_annuity(
        snakemake.params.lifetime, snakemake.params.discount_rate
    )
    out = pd.DataFrame(
        {
            "capital_cost": (ates_by_region["CAPEX_EUR_MW"] * annuity).where(
                feasible, 0.0
            ),
            "hourly_standing_losses": annual_retention_to_hourly_standing_loss(
                ates_by_region["VERLUSTRATE"]
            ).where(feasible, 0.0),
            "feasible": feasible,
        }
    )
    out.to_csv(snakemake.output.ates_potentials, index_label="name")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_ates_potentials", clusters="10", planning_horizons=2040
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    if snakemake.params.data_source == "hi_grid":
        build_hi_grid_potentials(snakemake)
    else:
        build_bgr_potentials(snakemake)
