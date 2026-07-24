from scripts.build_cop_profiles.run import expand_heat_sources_for_ptes_layers


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
