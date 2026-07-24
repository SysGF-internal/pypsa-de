import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from scripts.build_ates_potentials import (
    aggregate_ates_to_regions,
    annual_retention_to_hourly_standing_loss,
)


def test_aggregate_hi_ates_uses_inverse_capex_weights():
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
    result = annual_retention_to_hourly_standing_loss(pd.Series([1.0, 0.8]))

    assert result.iloc[0] == 0.0
    assert result.iloc[1] == pytest.approx(1 - 0.8 ** (1 / 8760))
