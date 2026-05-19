# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pypsa
import pytest


def _load_modify_prenetwork_module():
    module_path = Path("scripts/pypsa-de/modify_prenetwork.py")
    spec = spec_from_file_location("modify_prenetwork", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_capacity_limits_leaves_nom_max_unchanged_when_disabled():
    module = _load_modify_prenetwork_module()

    n = pypsa.Network()
    n.add("Bus", "FR0", country="FR")
    n.add(
        "Generator",
        "FR wind",
        bus="FR0",
        carrier="offwind-ac",
        p_nom=1.0,
        p_nom_extendable=True,
        p_nom_min=0.0,
        p_nom_max=10.0,
    )
    n.generators.loc["FR wind", "p_nom_opt"] = 1.0

    baseline_component = n.generators.copy()
    baseline_component.loc["FR wind", "p_nom_opt"] = 7.0
    baseline_component.loc["FR wind", "p_nom_min"] = 0.0
    baseline_component.loc["FR wind", "p_nom_max"] = 9.0

    module._apply_capacity_limits(
        n=n,
        component_type="Generator",
        indices=pd.Index(["FR wind"]),
        baseline_component=baseline_component,
        slack=0.1,
        nom_min=True,
        nom_max=False,
        unfix_bottlenecks=False,
    )

    assert n.generators.loc["FR wind", "p_nom_min"] == pytest.approx(6.3)
    assert n.generators.loc["FR wind", "p_nom_max"] == pytest.approx(10.0)
