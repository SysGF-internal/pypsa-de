import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from scripts.build_ates_potentials import (
    aggregate_ates_to_regions,
    aggregate_bgr_ates_potential,
    annual_retention_to_hourly_standing_loss,
)


def test_aggregate_ates_to_regions_uses_inverse_capex_weights():
    regions = gpd.GeoDataFrame(
        {
            "name": ["region-a", "region-b"],
            "geometry": [box(0, 0, 2, 2), box(3, 0, 5, 2)],
        },
        crs="EPSG:4326",
    )
    cells = gpd.GeoDataFrame(
        {
            "CAPEX_EUR_MW": [100.0, 200.0],
            "VERLUSTRATE": [0.9, 0.8],
            "geometry": [box(0.1, 0.1, 0.9, 0.9), box(1.1, 0.1, 1.9, 0.9)],
        },
        crs="EPSG:4326",
    )

    result = aggregate_ates_to_regions(cells, regions)

    assert result.index.tolist() == ["region-a", "region-b"]
    assert result.loc["region-a", "CAPEX_EUR_MW"] == pytest.approx(400 / 3)
    assert result.loc["region-a", "VERLUSTRATE"] == pytest.approx(13 / 15)
    assert result.loc["region-a", "n_cells"] == 2
    assert result.loc["region-b"].isna().all()


def test_annual_retention_conversion():
    retention = pd.Series([1.0, 0.8])

    result = annual_retention_to_hourly_standing_loss(retention)

    assert result.iloc[0] == 0.0
    assert result.iloc[1] == pytest.approx(1 - 0.8 ** (1 / 8760))


def test_bgr_potential_uses_eligible_area_and_regional_temperature_difference():
    regions = gpd.GeoDataFrame(
        {"name": ["region-a"]},
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:3035",
    )
    aquifers = gpd.GeoDataFrame(
        {"AQUIF_NAME": ["suitable"]},
        geometry=[box(0, 0, 10, 10)],
        crs=regions.crs,
    )
    dh_areas = gpd.GeoDataFrame(
        geometry=[box(0, 0, 10, 10)],
        crs=regions.crs,
    )

    result = aggregate_bgr_ates_potential(
        aquifers,
        regions,
        dh_areas,
        pd.Series({"region-a": 50.0}),
        suitable_aquifer_types=["suitable"],
        aquifer_volumetric_heat_capacity=3600,
        fraction_of_aquifer_area_available=1.0,
        effective_screen_length=1.0,
        dh_area_buffer=0,
    )

    # 100 m² * 3600 kJ/(m³ K) * 1 m * 50 K / 3.6e6 kJ/MWh.
    assert result.loc["region-a"] == pytest.approx(5.0)
