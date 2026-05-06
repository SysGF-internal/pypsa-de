# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Selects n largest district heating systems from dh_areas, creates extended
onshore regions, and outputs subnode metadata for downstream rules.
"""

import logging
import math
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _sanitize_label(label):
    # Convert float-like numeric labels to int first (170.0 → 170)
    # to avoid the decimal point being stripped by regex (170.0 → "1700")

    if isinstance(label, float) and not math.isnan(label) and label == int(label):
        label = int(label)
    return (
        pd.Series([label])
        .fillna("DH")
        .astype(str)
        .str.replace(r"[^\w\s-]", "", regex=True)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    ).iloc[0]


def _load_census_data(path):
    """
    Load the German Zensus 100m heating raster (CSV or ZIP-of-CSV).

    Returns a GeoDataFrame in EPSG:3035 with point geometries at the cell
    centroids and the columns ``Fernheizung`` and ``Insgesamt_Heizungsart``
    coerced to numeric.
    """
    csv_kwargs = {"encoding": "latin1", "sep": ";"}
    if path.endswith(".zip"):
        with ZipFile(path) as zip_file:
            candidates = [
                name
                for name in zip_file.namelist()
                if name.endswith(".csv") and "Heizungsart" in name
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No census heating CSV found inside archive {path}"
                )
            with zip_file.open(candidates[0]) as source:
                census = pd.read_csv(source, **csv_kwargs)
    else:
        census = pd.read_csv(path, **csv_kwargs)

    census = census.replace("\x96", 0)
    census = gpd.GeoDataFrame(
        census,
        geometry=gpd.points_from_xy(census.x_mp_100m, census.y_mp_100m),
        crs="EPSG:3035",
    )

    for column in ["Fernheizung", "Insgesamt_Heizungsart"]:
        census[column] = pd.to_numeric(census[column], errors="coerce").fillna(0)

    return census


def _refine_census_shapes(census_cells, processing):
    """
    Iteratively clean and dissolve census-derived DH cell geometries per subnode.

    Each iteration explodes multipolygons, drops parts smaller than
    ``min_area[i]``, applies a buffer of ``buffer_factor[i] * sqrt(area)``
    plus an absolute ``buffer_absolute[i]`` (both default to 0), and
    re-dissolves by subnode ``name``.  Lists are zipped by index; any
    short list is treated as 0 from its end.
    """
    refined = census_cells.copy()
    min_areas = processing.get("min_area", [0])
    buffer_factors = processing.get("buffer_factor", [0])
    buffer_absolute = processing.get("buffer_absolute", 0)
    buffer_absolutes = (
        buffer_absolute if isinstance(buffer_absolute, list) else [buffer_absolute]
    )

    iterations = max(len(min_areas), len(buffer_factors), len(buffer_absolutes))
    for idx in range(iterations):
        refined = refined.explode(index_parts=False).reset_index(drop=True)
        min_area = min_areas[idx] if idx < len(min_areas) else 0
        buffer_factor = buffer_factors[idx] if idx < len(buffer_factors) else 0
        buffer_abs = buffer_absolutes[idx] if idx < len(buffer_absolutes) else 0

        refined = refined.loc[refined.geometry.area > min_area].copy()
        if refined.empty:
            break

        if buffer_factor or buffer_abs:
            refined["geometry"] = refined.geometry.buffer(
                refined.geometry.area.pow(0.5) * buffer_factor + buffer_abs
            )

        refined = refined.dissolve("name").reset_index()

    return refined


def _refine_subnodes_with_census(
    subnodes,
    census,
    min_district_heating_share,
    processing,
):
    """
    Replace ISI subnode geometries with census-derived shapes where available.

    For each subnode, intersects the ISI DH polygon with the union of all
    100m census cells whose district-heating share meets
    ``min_district_heating_share``, then refines via ``_refine_census_shapes``.
    Subnodes for which no census signal survives keep their ISI geometry
    (with a warning).
    """
    if subnodes.empty:
        return subnodes

    eligible = census.loc[
        census["Insgesamt_Heizungsart"] > 0,
        ["Fernheizung", "Insgesamt_Heizungsart", "geometry"],
    ].copy()
    eligible["district_heating_share"] = (
        eligible["Fernheizung"] / eligible["Insgesamt_Heizungsart"]
    )
    eligible = eligible.loc[
        eligible["district_heating_share"] >= min_district_heating_share
    ].copy()

    if eligible.empty:
        logger.warning(
            "No census cells matched the district-heating-share threshold; keeping ISI subnode geometries"
        )
        return subnodes

    eligible["geometry"] = eligible.geometry.buffer(50, cap_style="square")
    eligible = eligible[["geometry"]]

    subnodes_3035 = subnodes.to_crs("EPSG:3035")
    original_shapes = subnodes_3035[["name", "geometry"]].copy()
    census_in_subnodes = gpd.overlay(original_shapes, eligible, how="intersection")

    if census_in_subnodes.empty:
        logger.warning(
            "Census refinement produced no overlaps with ISI DH areas; keeping ISI subnode geometries"
        )
        return subnodes

    refined = _refine_census_shapes(
        census_in_subnodes[["name", "geometry"]],
        processing,
    )
    if refined.empty:
        logger.warning(
            "Census refinement removed all candidate subnode areas; keeping ISI subnode geometries"
        )
        return subnodes

    refined = gpd.overlay(refined, original_shapes, how="intersection")
    if refined.empty:
        logger.warning(
            "Processed census shapes no longer intersect ISI DH areas; keeping ISI subnode geometries"
        )
        return subnodes

    refined = refined.dissolve("name").reset_index().set_index("name")
    updated = subnodes_3035.copy()
    replacement_names = updated["name"].isin(refined.index)
    updated.loc[replacement_names, "geometry"] = updated.loc[
        replacement_names, "name"
    ].map(refined["geometry"])

    logger.info(
        f"Replaced ISI geometries with census-refined shapes for {replacement_names.sum()} subnodes"
    )
    return updated.to_crs(subnodes.crs)


def _merge_same_city_areas(dh_areas, demand_column):
    """
    Merge DH areas mapped to the same city within the same country.

    Unions geometries into multipolygons and sums demand so that each
    city is represented by a single DH area entry.  Areas without a
    city mapping are kept unchanged.
    """
    if "city" not in dh_areas.columns:
        return dh_areas

    has_city = dh_areas["city"].notna() & (dh_areas["city"] != "")
    no_city = dh_areas[~has_city].copy()
    with_city = dh_areas[has_city].copy()

    if with_city.empty:
        return dh_areas

    n_before = len(with_city)

    # Build per-column aggregation: sum demand, keep first for others
    agg = {}
    for col in with_city.columns:
        if col in ("geometry", "country", "city"):
            continue
        agg[col] = "sum" if col == demand_column else "first"

    merged = with_city.dissolve(by=["country", "city"], aggfunc=agg).reset_index()

    n_merged = n_before - len(merged)
    if n_merged > 0:
        dup_cities = with_city.groupby(["country", "city"]).size()
        dup_cities = dup_cities[dup_cities > 1]
        for (ct, city), count in dup_cities.items():
            demand_sum = with_city.loc[
                (with_city["country"] == ct) & (with_city["city"] == city),
                demand_column,
            ].sum()
            logger.info(
                f"  Merged {count} DH areas → {city} ({ct}): "
                f"{demand_sum:.1f} GWh combined"
            )

    result = pd.concat([merged, no_city], ignore_index=True)
    return gpd.GeoDataFrame(result, crs=dh_areas.crs)


def _find_containing_region(point, regions):
    """
    Find the region containing a point, or the nearest region if not contained.

    Parameters
    ----------
    point : shapely.geometry.Point
        The point to locate
    regions : geopandas.GeoDataFrame
        GeoDataFrame of regions with geometry column, indexed by region name

    Returns
    -------
    str
        Name (index) of the containing or nearest region
    """
    containing = regions[regions.contains(point)]
    if len(containing) > 0:
        return containing.index[0]
    distances = regions.geometry.distance(point)
    return distances.idxmin()


def identify_largest_district_heating_systems(
    dh_areas: gpd.GeoDataFrame,
    regions_onshore: gpd.GeoDataFrame,
    n_subnodes: int,
    countries: list[str],
    demand_column: str = "Dem_GWh",
    label_column: str = "Label",
) -> gpd.GeoDataFrame:
    """
    Select the n largest DH systems by demand, filter by country, and map
    each to its containing onshore region.

    Parameters
    ----------
    dh_areas : geopandas.GeoDataFrame
        District heating areas with geometry and demand data.
    regions_onshore : geopandas.GeoDataFrame
        Onshore regions indexed by cluster name.
    n_subnodes : int
        Number of largest DH systems to select.
    countries : list[str]
        Country codes to filter DH areas.
    demand_column : str, optional
        Column with demand values in GWh/a.
    label_column : str, optional
        Column with DH system labels.

    Returns
    -------
    geopandas.GeoDataFrame
        Subnodes with columns: name, cluster, subnode_label,
        yearly_heat_demand_MWh, country, geometry (and city if present).
    """
    if "country" in dh_areas.columns:
        available = set(dh_areas["country"].unique())
        missing = set(countries) - available
        if missing:
            logger.warning(f"DH areas missing for countries {sorted(missing)}")
        dh_areas = dh_areas[dh_areas["country"].isin(available & set(countries))].copy()
        logger.info(
            f"Filtered to {len(dh_areas)} DH areas in {available & set(countries)}"
        )
    else:
        logger.warning("No 'country' column in dh_areas")

    dh_areas_valid = dh_areas.dropna(subset=[demand_column])
    if len(dh_areas_valid) == 0:
        logger.warning("No valid DH areas found")
        return gpd.GeoDataFrame()

    subnodes = dh_areas_valid.nlargest(n_subnodes, demand_column).copy()
    logger.info(
        f"Selected {len(subnodes)} largest DH systems ({subnodes[demand_column].sum():.1f} GWh/a)"
    )

    subnodes["yearly_heat_demand_MWh"] = subnodes[demand_column] * 1e3

    if subnodes.crs != regions_onshore.crs:
        subnodes = subnodes.to_crs(regions_onshore.crs)

    subnodes["centroid"] = subnodes.geometry.centroid
    subnodes["cluster"] = subnodes["centroid"].apply(
        lambda p: _find_containing_region(p, regions_onshore)
    )

    subnodes["subnode_label"] = subnodes[label_column].apply(_sanitize_label)
    subnodes["name"] = subnodes["cluster"] + " " + subnodes["subnode_label"]

    keep_cols = [
        "name",
        "cluster",
        "subnode_label",
        "yearly_heat_demand_MWh",
        "country",
        "geometry",
    ]
    if "city" in subnodes.columns:
        keep_cols.insert(3, "city")

    return subnodes[keep_cols].copy()


def extend_regions_onshore(
    regions_onshore: gpd.GeoDataFrame,
    subnodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Extend onshore regions with subnode geometries cut from parent clusters.

    For each subnode, cuts its geometry from the parent cluster and adds
    the subnode as a new region entry.

    Parameters
    ----------
    regions_onshore : geopandas.GeoDataFrame
        Original onshore regions indexed by cluster name
    subnodes : geopandas.GeoDataFrame
        Subnodes with name, cluster, and geometry columns

    Returns
    -------
    geopandas.GeoDataFrame
        Extended regions with subnodes added and parent geometries updated
    """
    if len(subnodes) == 0:
        return regions_onshore.copy()

    subnodes_crs = subnodes.to_crs(regions_onshore.crs)
    regions_extended = regions_onshore.copy()

    for cluster in subnodes_crs["cluster"].unique():
        cluster_subnodes = subnodes_crs[subnodes_crs["cluster"] == cluster]
        subnode_union = cluster_subnodes.union_all()

        if cluster in regions_extended.index:
            original_geom = regions_extended.loc[cluster, "geometry"]
            regions_extended.loc[cluster, "geometry"] = original_geom.difference(
                subnode_union
            )

    subnode_entries = subnodes_crs[["name", "geometry"]].set_index("name")
    return pd.concat([regions_extended, subnode_entries])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_district_heating_subnodes",
            clusters=48,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    countries = snakemake.params.get("countries", [])
    subnode_countries = snakemake.params.get("subnode_countries", None)
    n_subnodes = snakemake.params.get("n_subnodes", 40)
    demand_column = snakemake.params.get("demand_column", "Dem_GWh")
    label_column = snakemake.params.get("label_column", "Label")
    census_areas = snakemake.params.get("census_areas", {})

    if subnode_countries:
        invalid = set(subnode_countries) - set(countries)
        if invalid:
            logger.warning(f"Invalid subnode_countries {invalid} ignored")
        effective_subnode_countries = [c for c in subnode_countries if c in countries]
    else:
        effective_subnode_countries = countries

    logger.info(f"Identifying {n_subnodes} subnodes from {effective_subnode_countries}")

    dh_areas = gpd.read_file(snakemake.input.dh_areas)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    city_lookup_path = snakemake.input.get("dh_city_lookup") or snakemake.params.get(
        "dh_city_lookup", ""
    )
    if not city_lookup_path or not Path(city_lookup_path).exists():
        logger.warning(
            "dh_city_lookup missing; proceeding without city names. "
            "Run map_dh_systems_to_cities to enable city-labelled subnodes."
        )
        city_lookup = None
    else:
        city_lookup = pd.read_csv(city_lookup_path)

    # Attach city names to ALL dh_areas before selection so that areas
    # sharing the same city can be merged into one entry.  This avoids
    # duplicate subnodes (e.g. Duesseldorf_0 / Duesseldorf_1) and frees
    # slots for additional cities to fill up to n_subnodes.
    if city_lookup is not None:
        city_lookup_clean = city_lookup.copy()
        city_lookup_clean["subnode_label"] = city_lookup_clean["subnode_label"].apply(
            _sanitize_label
        )
        dh_areas["_tmp_label"] = dh_areas[label_column].apply(_sanitize_label)
        city_mapping = city_lookup_clean[
            ["subnode_label", "country", "city"]
        ].drop_duplicates(subset=["subnode_label", "country"])
        dh_areas = dh_areas.merge(
            city_mapping,
            how="left",
            left_on=["_tmp_label", "country"],
            right_on=["subnode_label", "country"],
        ).drop(columns=["_tmp_label", "subnode_label"])

        n_before = len(dh_areas)
        dh_areas = _merge_same_city_areas(dh_areas, demand_column)
        n_after = len(dh_areas)
        if n_before != n_after:
            logger.info(
                f"Pre-merged {n_before - n_after} same-city DH areas "
                f"({n_before} → {n_after} areas)"
            )

    subnodes = identify_largest_district_heating_systems(
        dh_areas,
        regions_onshore,
        n_subnodes,
        effective_subnode_countries,
        demand_column,
        label_column,
    )

    # Use city name for readable node names when available
    if "city" in subnodes.columns:
        city_clean = (
            subnodes["city"]
            .fillna("")
            .astype(str)
            .str.replace(r"[^\w\s-]", "", regex=True)
            .str.strip()
        )
        use_city = city_clean != ""
        subnodes.loc[use_city, "name"] = (
            subnodes.loc[use_city, "cluster"] + " " + city_clean[use_city]
        )

    # Handle duplicate names (e.g. same city spans two clusters)
    duplicates = subnodes["name"].duplicated(keep=False)
    if duplicates.any():
        for name in subnodes.loc[duplicates, "name"].unique():
            mask = subnodes["name"] == name
            subnodes.loc[mask, "name"] = [f"{name}_{i}" for i in range(mask.sum())]

    if len(subnodes) == 0:
        logger.warning("No subnodes identified")
        subnodes = gpd.GeoDataFrame(
            columns=[
                "name",
                "cluster",
                "subnode_label",
                "city",
                "yearly_heat_demand_MWh",
                "country",
                "geometry",
            ],
            crs=regions_onshore.crs,
        )
    elif census_areas.get("enable", False):
        census_path = snakemake.input.get("census", "")
        if not census_path or not Path(census_path).exists():
            raise FileNotFoundError(
                "Subnode census refinement is enabled but no census data file was found. "
                "Set sector.district_heating.subnodes.census_areas.data_file to a local CSV or ZIP."
            )

        subnodes = _refine_subnodes_with_census(
            subnodes,
            _load_census_data(census_path),
            census_areas.get("min_district_heating_share", 0.01),
            census_areas.get("processing", {}),
        )

    regions_extended = extend_regions_onshore(regions_onshore, subnodes)

    subnodes.to_file(snakemake.output.dh_subnodes, driver="GeoJSON")
    regions_extended.to_file(
        snakemake.output.regions_onshore_extended, driver="GeoJSON"
    )

    if len(subnodes) > 0:
        logger.info(
            f"Saved {len(subnodes)} subnodes ({subnodes['yearly_heat_demand_MWh'].sum() / 1e6:.2f} TWh/a)"
        )
