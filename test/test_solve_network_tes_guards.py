from scripts.solve_network import _thermal_storage_constraints_enabled


def config_with_storage(*, ttes=False, ptes=False, ates=False):
    return {
        "sector": {
            "ttes": ttes,
            "district_heating": {
                "ptes": {"enable": ptes},
                "ates": {"enable": ates},
            },
        }
    }


def test_each_thermal_storage_enables_shared_constraints():
    assert _thermal_storage_constraints_enabled(config_with_storage(ttes=True))
    assert _thermal_storage_constraints_enabled(config_with_storage(ptes=True))
    assert _thermal_storage_constraints_enabled(config_with_storage(ates=True))
    assert not _thermal_storage_constraints_enabled(config_with_storage())
