# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build per-region PTES energy capacity potentials.

This script computes an upper bound on pit thermal energy storage (PTES)
energy capacity for each modeled onshore region (subnodes and base remnants
alike, when subnode modelling is enabled) by:

1. clipping district-heating areas to the region geometry,
2. excluding non-eligible land within the clipped area using a LUISA
   land-cover raster (and optionally a Natura 2000 raster),
3. optionally filtering out sub-areas with shallow groundwater (water-table
   depth above ``max_groundwater_depth``), and
4. converting the remaining eligible area to an energy capacity using a
   configured per-storage default capacity and minimum footprint.

Per region the result is in MWh; regions for which no limit applies
(feature disabled, or region not selected as a target) carry ``np.inf``.

Mirrors the way geothermal/river-water potentials are constrained per
district-heating area, so PTES is treated consistently with the other
limited heat sources.

Relevant settings
-----------------
.. code:: yaml

    sector:
        district_heating:
            ptes:
                potential_limit:
                    enable: true
                    explicit_subnodes: true
                    remaining_regions: true
                    land_cover_codes: [21, 23, 32, 33]
                    excluder_resolution: 10
                    min_area: 10000
                    default_capacity_mwh: 4500
                    natura: true
                    groundwater_depth: true
                    max_groundwater_depth: -10
"""

import logging
import os
import tempfile
import weakref

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from atlite.gis import ExclusionContainer, shape_availability
from rasterio.windows import Window

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _cleanup_temp_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def get_chunked_raster(
    dataset_path: str,
    bounds: tuple[float, float, float, float],
    buffer_distance: float = 1000,
) -> rasterio.io.DatasetReader:
    """
    Return a windowed copy of a raster around ``bounds`` (CRS-native units).

    A small temporary GeoTIFF is written to disk and opened for reading;
    the file is auto-deleted when the returned dataset is garbage-collected.
    """
    with rasterio.open(dataset_path) as src:
        buffered_bounds = (
            bounds[0] - buffer_distance,
            bounds[1] - buffer_distance,
            bounds[2] + buffer_distance,
            bounds[3] + buffer_distance,
        )
        window = rasterio.windows.from_bounds(*buffered_bounds, transform=src.transform)
        window = Window(
            int(max(0, window.col_off)),
            int(max(0, window.row_off)),
            int(min(src.width - int(window.col_off), window.width)),
            int(min(src.height - int(window.row_off), window.height)),
        )

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as temp:
            temp_path = temp.name

        with rasterio.open(
            temp_path,
            "w",
            driver="GTiff",
            height=window.height,
            width=window.width,
            count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=src.window_transform(window),
            nodata=src.nodata,
        ) as dst:
            dst.write(src.read(window=window))

    dataset = rasterio.open(temp_path)
    weakref.finalize(dataset, _cleanup_temp_file, temp_path)
    return dataset


def _valid_band_mask(band: np.ndarray, nodata: float | None) -> np.ndarray:
    if nodata is None:
        return np.isfinite(band)
    return band != nodata


def _filter_by_groundwater_depth(
    eligible: gpd.GeoDataFrame,
    groundwater: xr.Dataset | None,
    max_groundwater_depth: float | None,
) -> gpd.GeoDataFrame:
    """
    Drop eligible polygons whose nearest water-table depth is shallower
    than ``max_groundwater_depth`` (a non-positive metre value, e.g. ``-10``
    means "water table must lie at least 10 m below surface").
    """
    if groundwater is None or max_groundwater_depth is None:
        return eligible
    if eligible.empty:
        return eligible

    centroids_4326 = eligible.geometry.to_crs("EPSG:4326").centroid
    eligible = eligible.copy()
    eligible["water_table_depth"] = [
        float(
            groundwater["WTD"]
            .sel(lon=p.x, lat=p.y, method="nearest")
            .values
        )
        for p in centroids_4326
    ]
    return eligible.loc[eligible["water_table_depth"] < max_groundwater_depth].drop(
        columns=["water_table_depth"]
    )


def eligible_area_in_region(
    geometry,
    land_cover_path: str,
    natura_path: str | None,
    excluder_resolution: int,
    land_cover_codes: list[int],
    groundwater: xr.Dataset | None = None,
    max_groundwater_depth: float | None = None,
) -> float:
    """
    Compute the eligible PTES land area inside ``geometry`` in m².

    Eligibility is defined as land-cover pixels matching ``land_cover_codes``
    (LUISA), optionally outside any Natura 2000 protected area, and
    optionally with a sufficiently deep groundwater table.
    """
    bounds = geometry.bounds
    land_cover = get_chunked_raster(land_cover_path, bounds)
    natura = get_chunked_raster(natura_path, bounds) if natura_path else None

    try:
        excluder = ExclusionContainer(crs=3035, res=excluder_resolution)
        excluder.add_raster(land_cover, codes=land_cover_codes, invert=True, crs=3035)
        if natura is not None:
            excluder.add_raster(natura, codes=[1], invert=False, crs=3035)

        region = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:3035")
        band, transform = shape_availability(region.geometry, excluder)
        row_indices, col_indices = np.where(_valid_band_mask(band, land_cover.nodata))

        if len(row_indices) == 0:
            return 0.0

        x_coords, y_coords = rasterio.transform.xy(transform, row_indices, col_indices)
        eligible = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x_coords, y_coords),
            crs=land_cover.crs,
        )
        eligible["geometry"] = eligible.geometry.buffer(
            excluder_resolution / 2,
            cap_style="square",
        )
        eligible = gpd.GeoDataFrame(
            geometry=[eligible.union_all()],
            crs=eligible.crs,
        ).explode(index_parts=False)
        eligible = eligible.reset_index(drop=True)
        eligible = gpd.overlay(eligible, region, how="intersection")

        if eligible.empty:
            return 0.0

        eligible = _filter_by_groundwater_depth(
            eligible, groundwater, max_groundwater_depth
        )
        if eligible.empty:
            return 0.0

        return float(eligible.geometry.area.sum())
    finally:
        land_cover.close()
        if natura is not None:
            natura.close()


def build_ptes_potentials(
    regions_onshore: gpd.GeoDataFrame,
    dh_areas: gpd.GeoDataFrame,
    subnode_names: set[str],
    params: dict,
    land_cover_path: str,
    natura_path: str | None,
    groundwater_path: str | None = None,
) -> pd.DataFrame:
    """
    Build a per-region PTES energy capacity potential table.

    Parameters
    ----------
    regions_onshore : geopandas.GeoDataFrame
        Onshore region geometries (extended with subnodes when subnode
        modelling is enabled). Must contain a ``name`` column or a
        ``name``-named index.
    dh_areas : geopandas.GeoDataFrame
        District-heating area geometries (one row per DH area).
    subnode_names : set[str]
        Names of regions corresponding to explicit DH subnodes. Used to
        decide whether to apply the limit per region (vs. the aggregated
        base-region remnants).
    params : dict
        Contents of ``sector.district_heating.ptes.potential_limit``.
        Switches ``explicit_subnodes`` and ``remaining_regions`` decide
        which set of regions is limited.
    land_cover_path : str
        Path to the LUISA land-cover raster (EPSG:3035).
    natura_path : str | None
        Path to the Natura 2000 raster, or ``None`` when not used.
    groundwater_path : str | None
        Path to a NetCDF dataset providing water-table depth (variable
        ``WTD``, dims ``lon``/``lat``, in metres, sign convention with
        ``WTD < 0`` for "below surface"), or ``None`` to skip the
        groundwater filter.

    Returns
    -------
    pandas.DataFrame
        Single column ``ptes_potential`` (MWh), indexed by region name.
        Regions outside the targeted set carry ``np.inf`` (no limit).
    """
    regions = regions_onshore.to_crs("EPSG:3035").copy()
    if "name" not in regions.columns:
        regions = regions.reset_index()
    regions = regions[["name", "geometry"]]

    potentials = pd.Series(np.inf, index=regions["name"], name="ptes_potential")

    if not params.get("enable", False):
        return potentials.to_frame()

    limit_explicit_subnodes = params.get("explicit_subnodes", True)
    limit_remaining_regions = params.get("remaining_regions", True)
    if not limit_explicit_subnodes and not limit_remaining_regions:
        return potentials.to_frame()

    if not land_cover_path:
        raise ValueError(
            "PTES potential limiting is enabled but no land-cover raster path "
            "was provided. Check that retrieve_luisa_land_cover is wired in."
        )

    target_names = []
    if limit_explicit_subnodes:
        target_names.extend(name for name in regions["name"] if name in subnode_names)
    if limit_remaining_regions:
        target_names.extend(
            name for name in regions["name"] if name not in subnode_names
        )
    target_names = pd.Index(dict.fromkeys(target_names))

    if target_names.empty:
        return potentials.to_frame()

    # Targets get 0 by default; eligible area found below lifts that.
    potentials.loc[target_names] = 0.0

    limited_regions = regions.loc[regions["name"].isin(target_names)]
    dh_regions = gpd.overlay(
        dh_areas.to_crs(limited_regions.crs),
        limited_regions,
        how="intersection",
    )

    if dh_regions.empty:
        return potentials.to_frame()

    max_groundwater_depth = (
        params.get("max_groundwater_depth")
        if groundwater_path
        else None
    )
    groundwater = (
        xr.open_dataset(groundwater_path)
        if groundwater_path and max_groundwater_depth is not None
        else None
    )
    try:
        dh_regions = dh_regions.dissolve("name").reset_index()
        for row in dh_regions.itertuples(index=False):
            area_m2 = eligible_area_in_region(
                row.geometry,
                land_cover_path,
                natura_path,
                params["excluder_resolution"],
                params["land_cover_codes"],
                groundwater=groundwater,
                max_groundwater_depth=max_groundwater_depth,
            )
            potentials.loc[row.name] = (
                area_m2 / params["min_area"] * params["default_capacity_mwh"]
            )
    finally:
        if groundwater is not None:
            groundwater.close()

    return potentials.to_frame()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_ptes_potentials",
            clusters=48,
            planning_horizons=2045,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params.potential_limit
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    dh_areas = gpd.read_file(snakemake.input.dh_areas)

    dh_subnodes_path = snakemake.input.get("dh_subnodes")
    if dh_subnodes_path:
        dh_subnodes = gpd.read_file(dh_subnodes_path)
        subnode_names = set(dh_subnodes["name"])
    else:
        subnode_names = set()

    natura_path = snakemake.input.get("natura") or None
    groundwater_path = snakemake.input.get("groundwater_depth") or None
    ptes_potentials = build_ptes_potentials(
        regions_onshore,
        dh_areas,
        subnode_names,
        params,
        snakemake.input.get("land_cover", ""),
        natura_path,
        groundwater_path,
    )

    logger.info(f"Writing PTES potentials to {snakemake.output.ptes_potentials}")
    ptes_potentials.to_csv(snakemake.output.ptes_potentials, index_label="name")
