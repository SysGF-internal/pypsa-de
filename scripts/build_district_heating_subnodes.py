# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Select district heating systems from dh_areas, create extended onshore
regions, and output subnode metadata for downstream rules.
"""

import logging
import math
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_district_heating_subnode_demands import (
    _compute_useful_heat_for_cluster,
    _to_scalar,
)

logger = logging.getLogger(__name__)


def _sanitize_label(label):
    """Normalize DH labels into stable identifier fragments."""
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


def _empty_subnodes(crs=None, include_city=False):
    """Return an empty subnode GeoDataFrame with the expected columns."""
    data = {
        "name": pd.Series(dtype="str"),
        "cluster": pd.Series(dtype="str"),
        "subnode_label": pd.Series(dtype="str"),
        "yearly_heat_demand_MWh": pd.Series(dtype="float64"),
        "country": pd.Series(dtype="str"),
        "geometry": gpd.GeoSeries([], crs=crs),
    }
    if include_city:
        data = {
            "name": data["name"],
            "cluster": data["cluster"],
            "subnode_label": data["subnode_label"],
            "city": pd.Series(dtype="str"),
            "yearly_heat_demand_MWh": data["yearly_heat_demand_MWh"],
            "country": data["country"],
            "geometry": data["geometry"],
        }
    return gpd.GeoDataFrame(data, geometry="geometry", crs=crs)


def _normalize_demand_share(demand_share):
    """Normalize demand-share selection inputs to a fraction in [0, 1]."""
    if demand_share in (None, ""):
        return None
    if pd.isna(demand_share):
        return None

    share = float(demand_share)
    if share < 0:
        raise ValueError("district heating subnode demand_share must be non-negative")
    if share > 1:
        if share <= 100:
            share /= 100
        else:
            raise ValueError(
                "district heating subnode demand_share must be between 0 and 1, or between 0 and 100 when given in percent"
            )
    return share


def _resolve_space_heat_reduction(
    reduce_space_heat,
    reduction_factors,
    investment_year,
):
    """Resolve the configured space-heat reduction for one planning year."""
    if not reduce_space_heat:
        return 0.0
    if isinstance(reduction_factors, dict):
        normalized = {int(year): float(value) for year, value in reduction_factors.items()}
        if investment_year in normalized:
            return normalized[investment_year]
        years = sorted(normalized)
        closest_year = min(years, key=lambda year: abs(year - investment_year))
        return normalized[closest_year]
    return float(reduction_factors)


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


def _attach_clusters(dh_areas, regions_onshore):
    """Map each DH area to the containing or nearest onshore region."""
    if dh_areas.empty:
        result = dh_areas.copy()
        result["cluster"] = pd.Series(dtype="str")
        return result

    mapped = dh_areas.to_crs(regions_onshore.crs) if dh_areas.crs != regions_onshore.crs else dh_areas.copy()
    centroids = mapped.geometry.centroid

    result = dh_areas.copy()
    result["cluster"] = centroids.apply(
        lambda point: _find_containing_region(point, regions_onshore)
    ).to_numpy()
    return result


def _model_urban_central_load_by_cluster(
    district_heat_share,
    pop_weighted_energy_totals,
    pop_weighted_heat_totals,
    heating_efficiencies,
    industrial_demand,
    district_heating_loss,
    space_heat_reduction,
):
    """Estimate modeled urban-central DH load per cluster in GWh/a."""
    cluster_loads_gwh = {}

    for cluster in pop_weighted_energy_totals.index:
        if cluster not in district_heat_share.index:
            continue

        useful_heat_mwh = _compute_useful_heat_for_cluster(
            cluster,
            pop_weighted_energy_totals,
            pop_weighted_heat_totals,
            heating_efficiencies,
            space_heat_reduction,
        )
        if useful_heat_mwh <= 0:
            continue

        district_fraction = _to_scalar(
            district_heat_share.loc[cluster, "district fraction of node"]
        )
        cluster_load_mwh = district_fraction * useful_heat_mwh * (1 + district_heating_loss)

        if (
            district_fraction > 0
            and cluster in industrial_demand.index
            and "low-temperature heat" in industrial_demand.columns
        ):
            cluster_load_mwh += (
                _to_scalar(industrial_demand.loc[cluster, "low-temperature heat"]) * 1e6
            )

        cluster_loads_gwh[cluster] = cluster_load_mwh / 1e3

    return pd.Series(cluster_loads_gwh, dtype="float64")


def _apply_model_urban_central_selection_demand(
    dh_areas,
    regions_onshore,
    demand_column,
    district_heat_share,
    pop_weighted_energy_totals,
    pop_weighted_heat_totals,
    heating_efficiencies,
    industrial_demand,
    district_heating_loss,
    space_heat_reduction,
):
    """Scale GIS DH areas to the modeled urban-central load used for selection."""
    dh_areas = _attach_clusters(dh_areas, regions_onshore)
    if dh_areas.empty:
        dh_areas["_selection_demand_GWh"] = pd.Series(dtype="float64")
        return dh_areas

    cluster_loads_gwh = _model_urban_central_load_by_cluster(
        district_heat_share,
        pop_weighted_energy_totals,
        pop_weighted_heat_totals,
        heating_efficiencies,
        industrial_demand,
        district_heating_loss,
        space_heat_reduction,
    )
    raw_cluster_demand = dh_areas.groupby("cluster")[demand_column].sum()
    cluster_scale = cluster_loads_gwh.reindex(raw_cluster_demand.index).div(raw_cluster_demand)
    cluster_scale = cluster_scale.replace([float("inf"), -float("inf")], 0).fillna(0)

    dh_areas["_selection_demand_GWh"] = (
        dh_areas[demand_column] * dh_areas["cluster"].map(cluster_scale).fillna(0)
    )

    logger.info(
        "Computed model-aware DH selection demand for %s clusters (%.1f GWh/a total)",
        len(cluster_scale),
        dh_areas["_selection_demand_GWh"].sum(),
    )
    return dh_areas


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

    if eligible.crs is None:
        raise ValueError("Census geometries must have a defined CRS")
    if eligible.crs.to_epsg() != 3035:
        eligible = eligible.to_crs(epsg=3035)

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

    refined = gpd.overlay(
        refined,
        original_shapes[["geometry"]],
        how="intersection",
    )
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
    demand_share: float | None = None,
    selection_column: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Select explicit DH systems by count or cumulative demand share, filter by
    country, and map each to its containing onshore region.

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
    demand_share : float, optional
        Cumulative demand share to represent explicitly. Values in [0, 1]
        are interpreted as fractions; values in (1, 100] are interpreted as
        percentages.
    selection_column : str, optional
        Column used for ranking and cumulative demand-share selection. If not
        provided, ``demand_column`` is used.

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

    selection_column = selection_column or demand_column

    dh_areas_valid = dh_areas.dropna(subset=[demand_column]).copy()
    if len(dh_areas_valid) == 0:
        logger.warning("No valid DH areas found")
        return _empty_subnodes(
            crs=dh_areas.crs,
            include_city="city" in dh_areas.columns,
        )
    if selection_column not in dh_areas_valid.columns:
        logger.warning(
            "Selection column '%s' missing; falling back to '%s'",
            selection_column,
            demand_column,
        )
        selection_column = demand_column

    demand_share = _normalize_demand_share(demand_share)
    include_city = "city" in dh_areas_valid.columns

    if demand_share is not None:
        if demand_share == 0:
            logger.info("Selected 0 DH systems because demand_share is 0%%")
            return _empty_subnodes(crs=dh_areas.crs, include_city=include_city)

        total_demand = dh_areas_valid[selection_column].sum()
        if total_demand <= 0:
            logger.warning("Total DH area demand is non-positive")
            return _empty_subnodes(crs=dh_areas.crs, include_city=include_city)

        ranked = dh_areas_valid.sort_values(selection_column, ascending=False).copy()
        if demand_share >= 1:
            subnodes = ranked
        else:
            cumulative_share = ranked[selection_column].cumsum() / total_demand
            n_selected = min(len(ranked), cumulative_share.lt(demand_share).sum() + 1)
            subnodes = ranked.iloc[:n_selected].copy()

        logger.info(
            "Selected %s largest DH systems covering %.1f%% of %.1f GWh/a (%s)",
            len(subnodes),
            100 * subnodes[selection_column].sum() / total_demand,
            total_demand,
            selection_column,
        )
    else:
        if n_subnodes <= 0:
            logger.info("Selected 0 DH systems because n_subnodes is 0")
            return _empty_subnodes(crs=dh_areas.crs, include_city=include_city)

        subnodes = dh_areas_valid.nlargest(n_subnodes, demand_column).copy()
        logger.info(
            f"Selected {len(subnodes)} largest DH systems ({subnodes[demand_column].sum():.1f} GWh/a)"
        )

    subnodes["yearly_heat_demand_MWh"] = subnodes[demand_column] * 1e3

    if subnodes.crs != regions_onshore.crs:
        subnodes = subnodes.to_crs(regions_onshore.crs)

    if "cluster" not in subnodes.columns or subnodes["cluster"].isna().any():
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

        if cluster not in regions_extended.index:
            raise KeyError(
                f"Parent cluster '{cluster}' missing from regions_onshore index"
            )

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
    demand_share = _normalize_demand_share(snakemake.params.get("demand_share", None))
    district_heating_loss = snakemake.params.get("district_heating_loss", 0.15)
    reduce_space_heat = snakemake.params.get("reduce_space_heat_exogenously", False)
    reduce_space_heat_factor = snakemake.params.get(
        "reduce_space_heat_exogenously_factor", {}
    )
    energy_totals_year = int(snakemake.params.get("energy_totals_year", 2019))
    selection_planning_horizon = snakemake.params.get("selection_planning_horizon")
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

    if demand_share is None:
        logger.info(
            f"Identifying {n_subnodes} subnodes from {effective_subnode_countries}"
        )
    else:
        logger.info(
            "Identifying explicit DH subnodes covering %.1f%% of demand from %s",
            100 * demand_share,
            effective_subnode_countries,
        )

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

    selection_column = None
    if demand_share is not None:
        if selection_planning_horizon is None:
            raise ValueError(
                "Demand-share DH subnode selection requires a planning horizon for model-aware demand inputs"
            )

        space_heat_reduction = _resolve_space_heat_reduction(
            reduce_space_heat,
            reduce_space_heat_factor,
            int(selection_planning_horizon),
        )
        district_heat_share = pd.read_csv(
            snakemake.input.district_heat_share_selection,
            index_col=0,
        )
        pop_weighted_energy_totals = pd.read_csv(
            snakemake.input.pop_weighted_energy_totals_selection,
            index_col=0,
        )
        pop_weighted_heat_totals = pd.read_csv(
            snakemake.input.pop_weighted_heat_totals_selection,
            index_col=0,
        )
        heating_efficiencies = pd.read_csv(
            snakemake.input.heating_efficiencies_selection,
            index_col=[1, 0],
        ).loc[energy_totals_year]
        industrial_demand = pd.read_csv(
            snakemake.input.industrial_demand_selection,
            index_col=0,
        )

        dh_areas = _apply_model_urban_central_selection_demand(
            dh_areas,
            regions_onshore,
            demand_column,
            district_heat_share,
            pop_weighted_energy_totals,
            pop_weighted_heat_totals,
            heating_efficiencies,
            industrial_demand,
            district_heating_loss,
            space_heat_reduction,
        )
        selection_column = "_selection_demand_GWh"

    subnodes = identify_largest_district_heating_systems(
        dh_areas,
        regions_onshore,
        n_subnodes,
        effective_subnode_countries,
        demand_column,
        label_column,
        demand_share,
        selection_column,
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
        subnodes = _empty_subnodes(
            crs=regions_onshore.crs,
            include_city="city" in dh_areas.columns,
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
