# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest

from scripts.add_existing_baseyear import align_existing_heating_capacities
from scripts.definitions.heat_system import HeatSystem


def test_missing_explicit_district_heating_subnodes_are_zero_filled():
    columns = pd.MultiIndex.from_tuples(
        [(HeatSystem.URBAN_CENTRAL.value, "air heat pump")]
    )
    existing_capacities = pd.DataFrame([[3.0]], index=["DE2"], columns=columns)
    nodes = pd.Index(["DE2", "DE2 3 Berlin"])

    aligned = align_existing_heating_capacities(
        existing_capacities, nodes, HeatSystem.URBAN_CENTRAL
    )

    assert aligned.index.equals(nodes)
    assert aligned.loc["DE2", columns[0]] == pytest.approx(3.0)
    assert aligned.loc["DE2 3 Berlin", columns[0]] == pytest.approx(0.0)


def test_missing_decentral_heating_nodes_remain_an_error():
    columns = pd.MultiIndex.from_tuples(
        [(HeatSystem.RESIDENTIAL_RURAL.value, "air heat pump")]
    )
    existing_capacities = pd.DataFrame([[3.0]], index=["DE2"], columns=columns)

    with pytest.raises(KeyError, match="DE3"):
        align_existing_heating_capacities(
            existing_capacities,
            pd.Index(["DE2", "DE3"]),
            HeatSystem.RESIDENTIAL_RURAL,
        )
