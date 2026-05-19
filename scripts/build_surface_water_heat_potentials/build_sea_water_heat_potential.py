# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calculate sea water heat potential for district heating systems.

This script computes the thermal potential of sea water as a heat source for district
heating applications. It uses sea water temperature data to estimate average water
temperatures across regions intersected with district heating areas.

The approximation accounts for spatial variations in sea water temperature,
providing both spatial and temporal aggregates. Temporal aggregates are only used for plotting.

Relevant Settings
-----------------

.. code:: yaml

    sector:
        district_heating:
            dh_area_buffer: # Buffer around DH areas in meters to include nearby coastal areas
            heat_source_cooling: # Exploitable temperature delta
    snapshots:
        start:
        end:
    enable:
        drop_leap_day:

Inputs
------
- `data/seawater_temperature.nc`: Sea water temperature data
- `resources/<run_name>/regions_onshore_base_s_{clusters}.geojson`: Onshore regions
- `resources/<run_name>/dh_areas_base_s_{clusters}.geojson`: District heating areas

Outputs
-------
- `resources/<run_name>/temp_sea_water_base_s_{clusters}.nc`: Sea water temperature profiles by region
- `resources/<run_name>/temp_sea_water_base_s_{clusters}_temporal_aggregate.nc`: Temporal aggregated temperature data
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scripts._helpers import (
    configure_logging,
    get_snapshots,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.build_surface_water_heat_potentials.approximators.sea_water_heat_approximator import (
    SeaWaterHeatApproximator,
)
from rioxarray.exceptions import NoDataInBounds, OneDimensionalRaster

logger = logging.getLogger(__name__)


def _create_empty_datasets(
    snapshots: pd.DatetimeIndex, center_lon: float, center_lat: float
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Create empty spatial and temporal aggregate datasets for regions without DH areas.

    When a region has no intersection with district heating areas, we still need
    to provide valid datasets with zero values to maintain consistent data structure
    across all regions. This prevents errors in downstream processing.

    Parameters
    ----------
    snapshots : pd.DatetimeIndex
        Time snapshots for the spatial aggregate
    center_lon : float
        Longitude of region center (for fallback coordinate)
    center_lat : float
        Latitude of region center (for fallback coordinate)

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Tuple of (spatial_aggregate, temporal_aggregate) datasets with zero values
    """
    # Create spatial aggregate with time-series of zeros
    # This represents no available sea water temperature over time
    spatial_aggregate = xr.Dataset(
        data_vars={
            "average_temperature": xr.DataArray(
                np.zeros(len(snapshots)),  # Zero temperature for all timestamps
                dims=["time"],
                coords={"time": snapshots},
            ),
        }
    )

    # Create temporal aggregate with single spatial point of zeros
    # This represents no available sea water temperature spatially
    temporal_aggregate = xr.Dataset(
        data_vars={
            "average_temperature": xr.DataArray(
                [[0.0]],  # Zero temperature at region center
                dims=["longitude", "latitude"],
                coords={"longitude": [center_lon], "latitude": [center_lat]},
            ),
        }
    )

    return spatial_aggregate, temporal_aggregate


def _empty_result_for_region(
    region: gpd.GeoDataFrame,
    snapshots: pd.DatetimeIndex,
) -> dict[str, xr.Dataset]:
    region_center = region.to_crs("EPSG:3035").centroid.to_crs("EPSG:4326").iloc[0]
    spatial_aggregate, temporal_aggregate = _create_empty_datasets(
        snapshots,
        region_center.x,
        region_center.y,
    )
    return {
        "spatial aggregate": spatial_aggregate,
        "temporal aggregate": temporal_aggregate,
    }


def _load_regional_seawater_temperature(
    seawater_temperature_fn: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> xr.DataArray | None:
    water_temperature = (
        xr.open_mfdataset(
            seawater_temperature_fn,
            chunks={
                "time": "auto",
                "latitude": "auto",
                "longitude": "auto",
                "depth": 1,
            },
        )["thetao"]
        .mean(dim="depth")
        .rio.write_crs("EPSG:4326")
    )

    try:
        water_temperature = water_temperature.rio.clip_box(
            minx,
            miny,
            maxx,
            maxy,
            auto_expand=True,
        ).rio.reproject("EPSG:3035")
    except (NoDataInBounds, OneDimensionalRaster, ValueError) as exc:
        logger.warning(
            "Sea-water temperature clip failed for bounds (%.5f, %.5f, %.5f, %.5f): %s",
            minx,
            miny,
            maxx,
            maxy,
            exc,
        )
        return None

    if any(water_temperature.sizes.get(dim, 0) < 2 for dim in ["x", "y"]):
        logger.warning(
            "Sea-water temperature clip for bounds (%.5f, %.5f, %.5f, %.5f) collapsed to x=%s, y=%s; using empty result",
            minx,
            miny,
            maxx,
            maxy,
            water_temperature.sizes.get("x", 0),
            water_temperature.sizes.get("y", 0),
        )
        return None

    return water_temperature


def get_regional_result(
    seawater_temperature_fn: str,
    region: gpd.GeoDataFrame,
    dh_areas: gpd.GeoDataFrame,
    snapshots: pd.DatetimeIndex,
) -> dict[str, xr.Dataset]:
    """
    Calculate sea water heat potential for a given region and district heating areas.

    Parameters
    ----------
    seawater_temperature_fn : str
        Path to NetCDF file containing sea water temperature data.
    region : geopandas.GeoDataFrame
        Geographical region for which to compute the heat potential.
    dh_areas : geopandas.GeoDataFrame
        District heating areas to intersect with the region.
    snapshots : pd.DatetimeIndex
        Time snapshots, used only for regions without dh_areas

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary with keys 'spatial aggregate' and 'temporal aggregate'.
        'spatial aggregate' contains average temperature.
        'temporal aggregate' contains temperature for analysis/plotting.
    """
    original_region = region.copy()

    intersected_geometry = gpd.overlay(
        region,
        dh_areas,
        how="intersection",
    ).union_all()

    region = region.copy()
    region.geometry = gpd.GeoSeries(
        [intersected_geometry],
        index=region.index,
        crs=region.crs,
    )

    if region.geometry.is_empty.any():
        return _empty_result_for_region(original_region, snapshots)

    minx, miny, maxx, maxy = region.total_bounds

    water_temperature = _load_regional_seawater_temperature(
        seawater_temperature_fn,
        minx,
        miny,
        maxx,
        maxy,
    )
    if water_temperature is None:
        return _empty_result_for_region(original_region, snapshots)

    region = region.to_crs("EPSG:3035")

    seawater_heat_approximator = SeaWaterHeatApproximator(
        water_temperature=water_temperature,
        region=region,
    )

    spatial_aggregate = seawater_heat_approximator.get_spatial_aggregate()

    temporal_aggregate = (
        seawater_heat_approximator.get_temporal_aggregate()
        .rio.reproject("EPSG:4326")
        .rename({"x": "longitude", "y": "latitude"})
    )

    spatial_aggregate = spatial_aggregate.compute()
    temporal_aggregate = temporal_aggregate.compute()

    return {
        "spatial aggregate": spatial_aggregate,
        "temporal aggregate": temporal_aggregate,
    }


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_sea_heat_potential",
            clusters="39",
            opts="",
            ll="vopt",
            sector_opts="",
            planning_horizons=2050,
        )

    # Configure logging and scenario
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    snapshots: pd.DatetimeIndex = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day
    )

    regions_onshore = gpd.read_file(snakemake.input["regions_onshore"])
    regions_onshore.set_index("name", inplace=True)
    regions_onshore.set_crs("EPSG:4326", inplace=True)

    dh_areas = gpd.read_file(snakemake.input["dh_areas"]).to_crs("EPSG:3035")
    dh_areas["geometry"] = dh_areas.geometry.buffer(snakemake.params.dh_area_buffer)
    dh_areas = dh_areas.to_crs("EPSG:4326")

    results = []
    for region_name in regions_onshore.index:
        logging.info(f"Processing region {region_name}")

        region = regions_onshore.loc[[region_name]].copy(deep=True)

        results.append(
            get_regional_result(
                seawater_temperature_fn=[
                    val
                    for key, val in dict(snakemake.input).items()
                    if key.startswith("seawater_temperature")
                ],
                region=region,
                dh_areas=dh_areas,
                snapshots=snapshots,
            )
        )

    temperature = (
        xr.concat(
            [res["spatial aggregate"]["average_temperature"] for res in results],
            dim="name",
        )
        .assign_coords(name=regions_onshore.index)
        .dropna(dim="time")
    )

    temperature = temperature.sel(time=snapshots, method="nearest").assign_coords(
        time=snapshots
    )

    temperature.to_netcdf(snakemake.output.heat_source_temperature)

    # Temperature temporal aggregate: spatial distribution of sea water temperatures
    # Units: °C (degrees Celsius) - average water temperature per coastal region
    # Used for analysis and plotting
    xr.concat(
        [res["temporal aggregate"]["average_temperature"] for res in results],
        dim=regions_onshore.index,
    ).to_netcdf(snakemake.output.heat_source_temperature_temporal_aggregate)
