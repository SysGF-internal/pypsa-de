# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Regression tests for the shared TES ratio constraints.

With several thermal storage technologies enabled at once (TTES, PTES, ATES)
the filtered charger, discharger and store indices are neither equally long nor
equally ordered.  Pairing them positionally silently mixes technologies up or
aborts the solve, so both constraints must pair components by name.
"""

import pytest

pypsa = pytest.importorskip("pypsa")

from scripts.solve_network import (  # noqa: E402
    add_TES_charger_ratio_constraints,
    add_TES_energy_to_power_ratio_constraints,
)

# (carrier, energy-to-power ratio [h], discharger efficiency)
TECHS = [
    ("urban central water tanks", 3.0, 0.9),
    ("urban central water pits", 22.5, 0.8),
    ("urban central aquifer thermal energy storage", 100.0, 0.7),
]
NODES = ["DE0 0", "DE1 0"]


@pytest.fixture
def multi_tes_network():
    """Network whose store/charger/discharger index orders deliberately differ."""
    n = pypsa.Network()
    n.set_snapshots([0])
    for node in NODES:
        n.add("Bus", f"{node} heat")
        for tech, _, _ in TECHS:
            n.add("Bus", f"{node} {tech}")

    # Stores in technology-major order.
    for tech, _, _ in TECHS:
        for node in NODES:
            n.add(
                "Store",
                f"{node} {tech}",
                bus=f"{node} {tech}",
                e_nom_extendable=True,
                carrier=tech,
            )

    # Chargers in node-major order.
    for node in NODES:
        for tech, etpr, _ in TECHS:
            n.add(
                "Link",
                f"{node} {tech} charger",
                bus0=f"{node} heat",
                bus1=f"{node} {tech}",
                p_nom_extendable=True,
                carrier=f"{tech} charger",
            )
            n.links.loc[f"{node} {tech} charger", "energy to power ratio"] = etpr

    # Dischargers in reversed technology order.
    for node in NODES:
        for tech, _, eff in reversed(TECHS):
            n.add(
                "Link",
                f"{node} {tech} discharger",
                bus0=f"{node} {tech}",
                bus1=f"{node} heat",
                p_nom_extendable=True,
                efficiency=eff,
                carrier=f"{tech} discharger",
            )

    n.stores["capital_cost"] = 1.0
    n.links["capital_cost"] = 1.0
    n.optimize.create_model(include_objective_constant=False)
    return n


def test_energy_to_power_pairs_store_with_its_own_charger(multi_tes_network):
    n = multi_tes_network
    add_TES_energy_to_power_ratio_constraints(n)

    con = n.model.constraints["TES_energy_to_power_ratio"]
    # ATES is deliberately outside this constraint; TTES + PTES at two nodes.
    assert con.size == 4

    printed = str(con)
    for node in NODES:
        assert (
            f"+1 Store-e_nom[{node} urban central water tanks] "
            f"- 3 Link-p_nom[{node} urban central water tanks charger]" in printed
        )
        assert (
            f"+1 Store-e_nom[{node} urban central water pits] "
            f"- 22.5 Link-p_nom[{node} urban central water pits charger]" in printed
        )


def test_charger_ratio_pairs_charger_with_its_own_discharger(multi_tes_network):
    n = multi_tes_network
    add_TES_charger_ratio_constraints(n)

    con = n.model.constraints["TES_charger_ratio"]
    assert con.size == len(NODES) * len(TECHS)

    printed = str(con)
    for node in NODES:
        for tech, _, eff in TECHS:
            assert (
                f"+1 Link-p_nom[{node} {tech} charger] "
                f"- {eff} Link-p_nom[{node} {tech} discharger]" in printed
            )


def test_charger_without_matching_discharger_raises(multi_tes_network):
    n = multi_tes_network
    # Make one discharger non-extendable so its charger loses its partner.
    n.links.loc["DE0 0 urban central water pits discharger", "p_nom_extendable"] = False

    with pytest.raises(RuntimeError, match="no matching extendable discharger"):
        add_TES_charger_ratio_constraints(n)


def test_charger_without_matching_store_raises(multi_tes_network):
    n = multi_tes_network
    n.stores.loc["DE0 0 urban central water pits", "e_nom_extendable"] = False

    with pytest.raises(RuntimeError, match="no matching extendable TES store"):
        add_TES_energy_to_power_ratio_constraints(n)
