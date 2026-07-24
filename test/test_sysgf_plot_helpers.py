from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import sysgf_plot_helpers as helpers


def test_apply_carrier_groups_claims_each_carrier_once_in_definition_order():
    frame = pd.DataFrame({"hp": [1.0], "ptes": [2.0], "other": [3.0]})

    result = helpers.apply_carrier_groups(
        frame,
        {
            "first": ["hp", "ptes"],
            "second": ["ptes", "other"],
        },
    )

    assert result.to_dict("records") == [{"first": 3.0, "second": 3.0}]
    assert frame.columns.tolist() == ["hp", "ptes", "other"]


def test_clean_label_applies_sysgf_storage_and_heat_source_names():
    assert helpers.clean_label("urban central river_water heat pump CC") == (
        "River water heat pump"
    )
    assert helpers.clean_label("urban central water pits charger") == "PTES charger"


def test_get_colors_uses_first_network_and_explicit_overrides():
    network = SimpleNamespace(
        carriers=pd.DataFrame(
            {"color": ["#111111", "", None]},
            index=["air", "empty", "missing"],
        )
    )

    result = helpers.get_colors({"scenario": network}, {"air": "#222222"})

    assert result == {"air": "#222222"}


def test_process_networks_requires_one_match_and_loads_it(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        helpers.pypsa,
        "Network",
        lambda path: loaded.append(path) or f"network:{path}",
    )

    result = helpers.process_networks(
        ["results/run/scenario/network.nc"],
        "run",
        ["scenario"],
    )

    assert result == {"scenario": "network:results/run/scenario/network.nc"}
    assert loaded == ["results/run/scenario/network.nc"]

    with pytest.raises(RuntimeError, match="found 0"):
        helpers.process_networks([], "run", ["scenario"])
