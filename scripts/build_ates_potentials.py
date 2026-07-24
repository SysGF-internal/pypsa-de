# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calculate the Aquifer Thermal Energy Storage (ATES) potentials for each region based on suitable aquifers, district heating areas, and temperature profiles.

The script filters aquifers based on their geological suitability (currently only highly productive porous aquifers), intersects them with future district heating areas and onshore regions, and calculates the energy storage potential. The difference between average forward and return temperatures is used as temperature differentials to convert from volumetric to energetic storage potentials.

The methodology is loosely based on Jackson, Regnier, Staffell (2024).
Future district heating areas are sourced from Manz et al. (2024), based on Fallahnejad et al. (2024). Data on aquifers stems from the German Bundesanstalt für Geowissenschaften und Rohstoffe (BGR).


Relevant Settings
-----------------

.. code:: yaml
    sector:
        aquifer_thermal_energy_storage:
            aquifer_volumetric_heat_capacity:
            fraction_of_aquifer_area_available:
            effective_screen_length:
            suitable_aquifer_types:
            dh_area_buffer:

Inputs
------
- `resources/<run_name>/regions_onshore.geojson`: Shapes of onshore regions
- `resources/<run_name>/aquifer_shapes.shp`: Shapes of aquifers
- `resources/<run_name>/dh_areas.geojson`: Shapes of district heating areas
- `resources/<run_name>/central_heating_forward_temperature_profiles.nc`: Forward temperature profiles
- `resources/<run_name>/central_heating_return_temperature_profiles.nc`: Return temperature profiles

Outputs
-------
- `resources/<run_name>/ates_potentials.csv`: ATES potentials per region in MWh

References
----------
- Jackson, Regnier, Staffell 2024: "Aquifer Thermal Energy Storage for low carbon heating and cooling in the United Kingdom: Current status and future prospects", Applied Energy, vol. 376, no. 124096, https://doi.org/10.1016/j.apenergy.2024.124096
- Manz et al. 2024: "Spatial analysis of renewable and excess heat potentials for climate-neutral district heating in Europe", Renewable Energy, vol. 224, no. 120111, https://doi.org/10.1016/j.renene.2024.120111
- Fallahnejad et al. 2024: "District heating potential in the EU-27: Evaluating the impacts of heat demand reduction and market share growth", Applied Energy, vol. 353, no. 122154, https://https://doi.org/10.1016/j.apenergy.2023.122154
- BGR: IHME1500 - Internationale Hydrogeologische Karte von Europa 1:1.500.000 (https://www.bgr.bund.de/DE/Themen/Wasser/Projekte/laufend/Beratung/Ihme1500/ihme1500_projektbeschr.html?nn=1546102)
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

    The interpretation of the HI source column ``VERLUSTRATE`` must be
    confirmed with the data provider before production use. The handover
    implementation treats it as a retention factor rather than a lost-energy
    fraction.
    """
    return 1 - annual_retention ** (1 / HOURS_PER_YEAR)


def aggregate_ates_to_regions(
    ates: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    capex_col: str = "CAPEX_EUR_MW",
    loss_col: str = "VERLUSTRATE",
) -> pd.DataFrame:
    """Aggregate HI grid cells using inverse-CAPEX weights per model region."""
    cells = ates.to_crs(regions.crs).copy()
    cells["geometry"] = cells.geometry.representative_point()
    joined = gpd.sjoin(
        cells,
        regions[["name", "geometry"]],
        how="inner",
        predicate="within",
    )
    joined["weight"] = 1.0 / joined[capex_col]
    frame = pd.DataFrame(joined.drop(columns="geometry"))

    def weighted_average(group):
        weights = group["weight"]
        return pd.Series(
            {
                capex_col: (group[capex_col] * weights).sum() / weights.sum(),
                loss_col: (group[loss_col] * weights).sum() / weights.sum(),
                "n_cells": len(group),
            }
        )

    aggregated = frame.groupby("name", sort=False).apply(
        weighted_average, include_groups=False
    )
    return aggregated.reindex(regions["name"])


def build_hi_grid_potentials(snakemake) -> None:
    """Build annualised regional cost/loss inputs from the licensed HI grid."""
    from scripts.add_electricity import calculate_annuity

    regions = gpd.read_file(snakemake.input.regions_onshore)
    ates_grid = gpd.read_file(snakemake.input.ates_grid)[
        ["CAPEX_EUR_MW", "VERLUSTRATE", "geometry"]
    ]
    ates_by_region = aggregate_ates_to_regions(ates_grid, regions)
    feasible = ates_by_region["n_cells"].notna() & (ates_by_region["n_cells"] > 0)
    annuity = calculate_annuity(
        snakemake.params.lifetime, snakemake.params.discount_rate
    )

    output = pd.DataFrame(
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
    output.to_csv(snakemake.output.ates_potentials, index_label="name")


def mwh_ates_per_m2(
    aquifer_volumetric_heat_capacity: float,
    fraction_of_aquifer_area_available: float,
    effective_screen_length: float,
    hot_well_temperature: float,
    cold_well_temperature: float,
    kwh_per_kj: float = 1 / 3600,
    mwh_per_kwh: float = 1 / 1e3,
) -> float:
    """
    Calculate the MWh potential per square meter for ATES systems.

    This is based on Jackson, Regnier, Staffell 2024 (https://doi.org/10.1016/j.apenergy.2024.124096).

    Parameters
    ----------
    aquifer_volumetric_heat_capacity : float
        Volumetric heat capacity of the aquifer in kJ/(m³·K)
    fraction_of_aquifer_area_available : float
        Fraction of the aquifer area that can be used for ATES (between 0 and 1)
    effective_screen_length : float
        Effective depth/thickness of the aquifer in meters
    hot_well_temperature : float
        Temperature of the hot well in degrees Celsius
    cold_well_temperature : float
        Temperature of the cold well in degrees Celsius
    kwh_per_kj : float, optional
        Conversion factor from kJ to kWh, by default 1/3600
    mwh_per_kwh : float, optional
        Conversion factor from kWh to MWh, by default 1/1000

    Returns
    -------
    float
        The ATES energy storage potential in MWh per square meter

    Raises
    ------
    Exception
        If calculation fails

    References
    ----------
    - Jackson, Regnier, Staffell 2024 (https://doi.org/10.1016/j.apenergy.2024.124096): Aquifer Thermal Energy Storages for low carbon heating and cooling in the United Kingdom: CUrrent status and future prospects
    """
    try:
        return (
            aquifer_volumetric_heat_capacity
            * fraction_of_aquifer_area_available
            * effective_screen_length
            * (hot_well_temperature - cold_well_temperature)
            * kwh_per_kj
            * mwh_per_kwh
        )
    except Exception as e:
        logger.error(f"Error calculating ATES potential per m2: {e}")
        raise


def suitable_aquifers(
    aquifer_shapes: gpd.GeoDataFrame,
    suitable_aquifer_types: list,
) -> gpd.GeoDataFrame:
    """
    Filter the aquifer shapes by the suitable aquifer types.

    Parameters
    ----------
    aquifer_shapes : geopandas.GeoDataFrame
        GeoDataFrame containing the shapes of all aquifers
    suitable_aquifer_types : list
        List of aquifer types considered suitable for ATES

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered GeoDataFrame containing only suitable aquifers

    Raises
    ------
    KeyError
        If required column 'AQUIF_NAME' is not found in the input data
    Exception
        If filtering process fails
    """
    try:
        if "AQUIF_NAME" not in aquifer_shapes.columns:
            raise KeyError("Column 'AQUIF_NAME' not found in aquifer shapes dataframe")

        filtered = aquifer_shapes[
            aquifer_shapes["AQUIF_NAME"].isin(suitable_aquifer_types)
        ]

        if filtered.empty:
            logger.warning("No suitable aquifers found with the specified types")

        return filtered
    except Exception as e:
        logger.error(f"Error filtering suitable aquifers: {e}")
        raise


def ates_potential_per_onshore_region(
    suitable_aquifers: gpd.GeoDataFrame,
    regions_onshore: gpd.GeoDataFrame,
    dh_areas: gpd.GeoDataFrame,
    dh_area_buffer: float,
    mwh_per_m2: float,
) -> gpd.GeoDataFrame:
    """
    Calculate the ATES potential for each onshore region.

    This function overlays suitable aquifers with onshore regions and district heating areas
    to calculate the potential ATES capacity for each region.

    Parameters
    ----------
    suitable_aquifers : geopandas.GeoDataFrame
        GeoDataFrame containing filtered suitable aquifers
    regions_onshore : geopandas.GeoDataFrame
        GeoDataFrame containing the shapes of onshore regions
    dh_areas : geopandas.GeoDataFrame
        GeoDataFrame containing the shapes of district heating areas
    dh_area_buffer : float
        Buffer distance in meters to apply around district heating areas
    mwh_per_m2 : float
        ATES potential in MWh per square meter

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with onshore regions and their ATES potential in MWh

    Raises
    ------
    KeyError
        If required column 'name' is not found in regions_onshore dataframe
    Exception
        If calculation process fails
    """
    try:
        if suitable_aquifers.empty or regions_onshore.empty or dh_areas.empty:
            logger.warning("One or more input GeoDataFrames are empty")

        ret_val = regions_onshore.copy()

        if "name" not in ret_val.columns:
            raise KeyError("Column 'name' not found in regions_onshore dataframe")

        ret_val.index = ret_val["name"]
        ret_val.drop(columns=["name"], inplace=True)

        suitable_aquifers_in_onshore_regions = gpd.overlay(
            suitable_aquifers, regions_onshore, how="intersection"
        )

        if suitable_aquifers_in_onshore_regions.empty:
            logger.warning("No suitable aquifers found in onshore regions")
            ret_val["ates_potential"] = 0
            return ret_val

        dh_areas_buffered = dh_areas.copy()
        dh_areas_buffered["geometry"] = dh_areas_buffered.geometry.buffer(
            dh_area_buffer
        )

        try:
            aquifers_in_dh_areas = (
                gpd.overlay(
                    dh_areas_buffered,
                    suitable_aquifers_in_onshore_regions,
                    how="intersection",
                )
                .groupby("name")["geometry"]
                .apply(lambda x: x.area.sum())
            )

            # Handle regions without any ATES potential
            missing_regions = set(ret_val.index) - set(aquifers_in_dh_areas.index)
            if missing_regions:
                logger.info(f"{len(missing_regions)} regions have no ATES potential")

            ret_val["ates_potential"] = 0  # Default value
            ret_val.loc[aquifers_in_dh_areas.index, "ates_potential"] = (
                aquifers_in_dh_areas * mwh_per_m2
            )

        except Exception as e:
            logger.error(f"Error in overlay calculation: {e}")
            ret_val["ates_potential"] = 0

        return ret_val

    except Exception as e:
        logger.error(f"Error calculating ATES potential per onshore region: {e}")
        raise


def check_dh_areas_coverage(dh_areas: gpd.GeoDataFrame, countries: list) -> None:
    """
    Check if district heating areas exist for all specified countries.

    Issues a warning if any country in the countries list doesn't have
    corresponding district heating areas in the provided GeoDataFrame.

    Parameters
    ----------
    dh_areas : geopandas.GeoDataFrame
        GeoDataFrame containing district heating areas
    countries : list
        List of country codes to check for coverage

    Returns
    -------
    None
        Function only logs warnings for missing countries

    Raises
    ------
    ValueError
        If countries list is empty or None, or if district heating areas dataframe is empty
    KeyError
        If the 'country' column is missing from the district heating areas dataframe
    """
    if countries is None or len(countries) == 0:
        raise ValueError("Countries list is empty or None.")

    if dh_areas.empty:
        raise ValueError(
            "District heating areas dataframe is empty. Cannot proceed with ATES calculation."
        )

    # Extract countries from DH areas - first try direct column
    if "country" in dh_areas.columns:
        dh_countries = set(dh_areas["country"].unique())
    else:
        raise KeyError(
            "Cannot find 'country' column in district heating areas dataframe. This column is required."
        )

    # Check for missing countries
    missing_countries = set(countries) - dh_countries

    if missing_countries:
        logger.info(
            f"No district heating areas found for the following countries: {', '.join(missing_countries)}"
        )
    else:
        logger.info("District heating areas available for all requested countries")


def check_aquifer_coverage(
    aquifer_shapes: gpd.GeoDataFrame,
    regions_onshore: gpd.GeoDataFrame,
    ignore_missing_regions: bool,
) -> None:
    """
    Check if aquifer shapes cover all onshore regions.

    Issues a warning if there are onshore regions without any aquifer coverage,
    which would result in zero ATES potential for those regions.

    Parameters
    ----------
    aquifer_shapes : geopandas.GeoDataFrame
        GeoDataFrame containing the shapes of all aquifers
    regions_onshore : geopandas.GeoDataFrame
        GeoDataFrame containing the shapes of onshore regions

    Returns
    -------
    None
        Function only logs warnings for uncovered regions

    Raises
    ------
    ValueError
        If input GeoDataFrames are empty
    """
    if aquifer_shapes.empty:
        raise ValueError(
            "Aquifer shapes dataframe is empty. Cannot proceed with ATES calculation."
        )

    if regions_onshore.empty:
        raise ValueError(
            "Onshore regions dataframe is empty. Cannot proceed with ATES calculation."
        )

    # Perform spatial overlay to check which regions have aquifer coverage
    aquifers_by_region = (
        gpd.overlay(aquifer_shapes, regions_onshore, how="intersection")
        .groupby("name")["geometry"]
        .count()
    )

    # Get regions without any aquifer coverage
    if "name" in regions_onshore.columns:
        all_region_names = set(regions_onshore["name"])
        covered_region_names = set(aquifers_by_region.index)
        uncovered_regions = all_region_names - covered_region_names

        if uncovered_regions:
            if ignore_missing_regions:
                logger.warning(
                    f"Regions without aquifer data coverage: {', '.join(uncovered_regions)}. "
                    "Ignoring these regions."
                )
            else:
                raise ValueError(
                    f"Regions without aquifer data coverage: {', '.join(uncovered_regions)}. Set `sector:district_heating:ates:ignore_missing_regions: true` to ignore these regions."
                )
        else:
            logger.info("All regions have aquifer coverage")
    else:
        raise KeyError(
            "Cannot find 'name' column in onshore regions dataframe. This column is required."
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_ates_potentials", clusters="48", planning_horizons=2030
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    if snakemake.params.data_source == "hi_grid":
        build_hi_grid_potentials(snakemake)
        raise SystemExit(0)

    # get onshore regions and index them by region name
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    regions_onshore = regions_onshore.to_crs(epsg=3035)

    countries: list = snakemake.params.countries

    aquifer_shapes = gpd.read_file(snakemake.input.aquifer_shapes_shp).to_crs(
        regions_onshore.crs
    )
    # fix any invalid geometries
    aquifer_shapes.geometry = aquifer_shapes.geometry.make_valid()

    dh_areas = gpd.read_file(snakemake.input.dh_areas).to_crs(regions_onshore.crs)

    # Check district heating areas coverage
    logger.info("Checking district heating areas coverage")
    check_dh_areas_coverage(
        dh_areas=dh_areas,
        countries=countries,
    )

    # Check aquifer coverage for onshore regions
    logger.info("Checking aquifer coverage for onshore regions")
    check_aquifer_coverage(
        aquifer_shapes=aquifer_shapes,
        regions_onshore=regions_onshore,
        ignore_missing_regions=snakemake.params.ignore_missing_regions,
    )

    forward_temperature_profiles = xr.open_dataarray(
        snakemake.input.central_heating_forward_temperature_profiles
    )
    return_temperature_profiles = xr.open_dataarray(
        snakemake.input.central_heating_return_temperature_profiles
    )

    logger.info("Calculating ATES potentials per m2")
    mwh_per_m2 = mwh_ates_per_m2(
        aquifer_volumetric_heat_capacity=snakemake.params.aquifer_volumetric_heat_capacity,
        fraction_of_aquifer_area_available=snakemake.params.fraction_of_aquifer_area_available,
        effective_screen_length=snakemake.params.effective_screen_length,
        hot_well_temperature=forward_temperature_profiles.mean(dim="time").to_pandas(),
        cold_well_temperature=return_temperature_profiles.to_pandas(),
    )

    logger.info("Filtering for suitable aquifers")
    suitable_aquifer_df = suitable_aquifers(
        aquifer_shapes=aquifer_shapes,
        suitable_aquifer_types=snakemake.params.suitable_aquifer_types,
    )

    logger.info("Calculating ATES potentials per region")
    ates_potentials = ates_potential_per_onshore_region(
        suitable_aquifers=suitable_aquifer_df,
        regions_onshore=regions_onshore,
        dh_areas=dh_areas,
        dh_area_buffer=snakemake.params.dh_area_buffer,
        mwh_per_m2=mwh_per_m2,
    )

    logger.info(f"Writing results to {snakemake.output.ates_potentials}")
    ates_potentials.to_csv(snakemake.output.ates_potentials, index_label="name")
