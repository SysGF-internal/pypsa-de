import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from scripts.build_surface_water_heat_potentials.approximators.surface_water_heat_approximator import (
    SurfaceWaterHeatApproximator,
)


def make_grid() -> xr.DataArray:
    return xr.DataArray(
        np.full((2, 3, 3), 10.0),
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            "latitude": [52.0, 51.0, 50.0],
            "longitude": [9.0, 10.0, 11.0],
        },
        dims=("time", "latitude", "longitude"),
    ).rio.write_crs("EPSG:4326")


def make_approximator() -> SurfaceWaterHeatApproximator:
    grid = make_grid()
    region = gpd.GeoSeries([box(8.5, 49.5, 11.5, 52.5)], crs="EPSG:4326")
    return SurfaceWaterHeatApproximator(
        volume_flow=grid,
        water_temperature=grid,
        region=region,
        min_distance_meters=2000,
    )


def test_native_wgs84_grid_resolution_and_aggregates():
    approximator = make_approximator()

    approximator._validate_and_reproject_input()

    assert approximator._data_resolution_degrees == pytest.approx(1.0)
    assert approximator._data_resolution_meters == pytest.approx(
        60 * 1852 * np.sqrt(np.cos(np.radians(51.0)))
    )
    assert set(approximator.get_spatial_aggregate().data_vars) == {
        "average_temperature",
        "total_power",
    }
    assert set(approximator.get_temporal_aggregate().data_vars) == {
        "average_temperature",
        "total_energy",
    }


def test_coordinate_mismatch_is_rejected():
    approximator = make_approximator()
    approximator.volume_flow = approximator.volume_flow.assign_coords(
        longitude=[9.0, 10.0, 12.0]
    )

    with pytest.raises(ValueError, match="longitude coordinates don't match"):
        approximator._validate_and_reproject_input()
