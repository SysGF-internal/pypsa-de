from scripts.build_heat_source_utilisation_profiles import generic_heat_sources
from scripts.definitions.heat_source import HeatSource, HeatSourceType


def test_ates_is_not_suppressed_by_ptes_routing():
    assert generic_heat_sources(["ptes", "ates", "air"], ptes_enable=True) == [
        "ates",
        "air",
    ]


def test_ates_uses_storage_preheating_but_not_ptes_resistive_boosting():
    assert HeatSource.ATES.source_type == HeatSourceType.STORAGE
    assert HeatSource.ATES.supports_preheating
    assert HeatSource.ATES.requires_heat_pump(ptes_discharge_resistive_boosting=True)
    assert not HeatSource.PTES.requires_heat_pump(
        ptes_discharge_resistive_boosting=True
    )
