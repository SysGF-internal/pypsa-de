# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import geopandas as gpd
import pytest
from shapely.geometry import box

from scripts.build_district_heating_subnodes import (
    _normalize_demand_share,
    identify_largest_district_heating_systems,
)
from scripts.lib.validation.config.sector import _SubnodesConfig


def test_normalize_demand_share_accepts_percent_values():
    assert _normalize_demand_share(50) == pytest.approx(0.5)


def test_subnodes_config_validates_demand_share_range():
    assert _SubnodesConfig(demand_share=50).demand_share == 50

    with pytest.raises(ValueError, match="demand_share"):
        _SubnodesConfig(demand_share=101)


def test_identify_largest_district_heating_systems_by_demand_share():
    dh_areas = gpd.GeoDataFrame(
        {
            "country": ["DE", "DE", "DE"],
            "Label": ["Alpha", "Beta", "Gamma"],
            "Dem_GWh": [60.0, 30.0, 10.0],
            "_selection_demand_GWh": [60.0, 30.0, 10.0],
        },
        geometry=[
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        crs="EPSG:3035",
    )
    regions_onshore = gpd.GeoDataFrame(
        {"name": ["DE1", "DE2", "DE3"]},
        geometry=[
            box(-0.5, -0.5, 1.5, 1.5),
            box(1.5, -0.5, 3.5, 1.5),
            box(3.5, -0.5, 5.5, 1.5),
        ],
        crs="EPSG:3035",
    ).set_index("name")

    subnodes = identify_largest_district_heating_systems(
        dh_areas,
        regions_onshore,
        n_subnodes=0,
        countries=["DE"],
        demand_share=0.8,
        selection_column="_selection_demand_GWh",
    )

    assert list(subnodes["name"]) == ["DE1 Alpha", "DE2 Beta"]
    assert list(subnodes["yearly_heat_demand_MWh"]) == [60000.0, 30000.0]


def test_demand_share_is_resolved_within_country_subset():
    """The share denominator must exclude countries without explicit subnodes."""
    dh_areas = gpd.GeoDataFrame(
        {
            "country": ["DE", "DE", "FR"],
            "Label": ["Alpha", "Beta", "Paris"],
            "Dem_GWh": [60.0, 40.0, 900.0],
            "_selection_demand_GWh": [60.0, 40.0, 900.0],
        },
        geometry=[
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        crs="EPSG:3035",
    )
    regions_onshore = gpd.GeoDataFrame(
        {"name": ["DE1", "DE2", "FR1"]},
        geometry=[
            box(-0.5, -0.5, 1.5, 1.5),
            box(1.5, -0.5, 3.5, 1.5),
            box(3.5, -0.5, 5.5, 1.5),
        ],
        crs="EPSG:3035",
    ).set_index("name")

    subnodes = identify_largest_district_heating_systems(
        dh_areas,
        regions_onshore,
        n_subnodes=0,
        countries=["DE"],
        demand_share=0.8,
        selection_column="_selection_demand_GWh",
    )

    # 80% of DE-only demand requires both systems (60 + 40 GWh/a).
    # If FR leaked into the denominator, Paris would be selected as well.
    assert list(subnodes["name"]) == ["DE1 Alpha", "DE2 Beta"]
    assert set(subnodes["country"]) == {"DE"}
