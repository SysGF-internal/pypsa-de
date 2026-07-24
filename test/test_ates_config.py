import pytest
from pydantic import ValidationError

from scripts.lib.validation.config import validate_config


def ates_config(
    *, data_source: str, data_file: str | None = None, heat_sources: list[str] | None = None
) -> dict:
    return {
        "sector": {
            "district_heating": {
                "ptes": {"enable": False},
                "ates": {
                    "enable": True,
                    "data_source": data_source,
                    "data_file": data_file,
                },
            },
            "heat_sources": {"urban central": heat_sources or ["air"]},
        }
    }


def test_legacy_bgr_ates_remains_available_without_private_data():
    config = validate_config(ates_config(data_source="bgr"))

    assert config.sector.district_heating.ates.enable


def test_hi_grid_ates_requires_input_data():
    with pytest.raises(ValidationError, match="no ATES input file"):
        validate_config(ates_config(data_source="hi_grid", heat_sources=["ates"]))


def test_hi_grid_ates_requires_heat_source():
    with pytest.raises(ValidationError, match="'ates' is not in"):
        validate_config(
            ates_config(
                data_source="hi_grid",
                data_file="hi.gpkg",
                heat_sources=["air"],
            )
        )


def test_hi_grid_ates_accepts_explicit_input_data():
    config = validate_config(
        ates_config(
            data_source="hi_grid",
            data_file="hi.gpkg",
            heat_sources=["ates"],
        )
    )

    assert config.sector.district_heating.ates.data_file == "hi.gpkg"
