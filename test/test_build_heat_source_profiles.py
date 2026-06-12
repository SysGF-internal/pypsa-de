from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_heat_source_profiles.run import (
    expand_heat_sources_for_ptes_layers,
    resolve_heat_source_cooling,
)


def test_expand_heat_sources_for_multilayer_ptes_replaces_plain_ptes():
    heat_sources_by_system = {
        "urban central": ["air", "ptes", "geothermal"],
        "urban decentral": ["air"],
    }

    expanded_heat_sources = expand_heat_sources_for_ptes_layers(
        heat_sources_by_system=heat_sources_by_system,
        ptes_layer_temperatures=[90, 62.5, 35],
    )

    assert expanded_heat_sources["urban central"] == [
        "air",
        "ptes layer 0",
        "ptes layer 1",
        "ptes layer 2",
        "geothermal",
    ]
    assert "ptes" not in expanded_heat_sources["urban central"]
    assert heat_sources_by_system["urban central"] == ["air", "ptes", "geothermal"]


def test_expand_heat_sources_for_single_layer_keeps_plain_ptes():
    heat_sources_by_system = {
        "urban central": ["air", "ptes", "geothermal"],
    }

    expanded_heat_sources = expand_heat_sources_for_ptes_layers(
        heat_sources_by_system=heat_sources_by_system,
        ptes_layer_temperatures=[90],
    )

    assert expanded_heat_sources == heat_sources_by_system


def test_resolve_heat_source_cooling_scalar_applies_to_all_sources():
    assert resolve_heat_source_cooling(6, "geothermal") == 6.0
    assert resolve_heat_source_cooling(6.5, "air") == 6.5


def test_resolve_heat_source_cooling_mapping_with_default():
    cooling = {"default": 6, "geothermal": 15}
    assert resolve_heat_source_cooling(cooling, "geothermal") == 15.0
    assert resolve_heat_source_cooling(cooling, "electrolysis_waste") == 6.0
    assert resolve_heat_source_cooling(cooling, "air") == 6.0


def test_resolve_heat_source_cooling_ptes_layers_fall_back_to_ptes_entry():
    cooling = {"default": 6, "ptes": 4}
    assert resolve_heat_source_cooling(cooling, "ptes layer 2") == 4.0
    assert resolve_heat_source_cooling({"default": 6}, "ptes layer 2") == 6.0
    assert (
        resolve_heat_source_cooling(
            {"default": 6, "ptes layer 1": 3}, "ptes layer 1"
        )
        == 3.0
    )
