# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build heat pump COP, heat-source boosting share and heat-pump cooling profiles.

This rule merges the former ``build_cop_profiles`` and
``build_heat_source_utilisation_profiles``: both depend on the source-side
cooling the heat pump applies to its source, so they are computed together
from one consistent value.

By default each source gets a fixed, typical source-side cooling from
``heat_source_cooling`` -- either a flat scalar or a per-source mapping with a
``default`` entry (e.g. 15 K for geothermal, 6 K for PtX waste heat).

Alternatively, ``heat_pump_cooling_iterative`` solves the cooling and the COP
consistently for preheating central sources (e.g. geothermal): with one source
stream, a preheater followed by the evaporator and equal mass flows on both
sides, the heat pump's energy balance ties the cooling to the lift and the COP:

    dT_cool = (COP - 1) / COP * (T_forward - T_source)

while the COP itself depends on dT_cool through the source outlet temperature.
:func:`compute_heat_pump_cooling` resolves the two by fixed-point (Picard)
iteration, seeded from the typical value. Non-preheating sources and decentral
systems always use the typical value.

COP values below 1 are set to zero (infeasible operating conditions). Central
heating uses the Jensen et al. (2018) thermodynamic model
(:class:`CentralHeatingCopApproximator`); decentral heating the Staffell et al.
(2012) regression (:class:`DecentralHeatingCopApproximator`).

Relevant Settings
-----------------

.. code:: yaml

    sector:
        heat_pump_sink_T_individual_heating:
        heat_sources:
            urban central:
            urban decentral:
            rural:
        district_heating:
            heat_source_cooling:
            heat_pump_cooling_iterative:
            log_heat_pump_cooling_iterations:
            heat_pump_cop_approximation:
            heat_source_temperatures:
            ptes:
                layered:

Inputs
------
- ``resources/<run_name>/central_heating_forward_temperature_profiles_base_s_{clusters}_{planning_horizons}.nc``
- ``resources/<run_name>/central_heating_return_temperature_profiles_base_s_{clusters}_{planning_horizons}.nc``
- ``resources/<run_name>/temp_<source>_total_base_s_{clusters}.nc``: temperature
  profiles of variable-temperature heat sources. PTES layer temperatures come
  from config (``ptes.layered.layer_temperatures``), not from an input file.

Outputs
-------
- ``resources/<run_name>/cop_profiles_base_s_{clusters}_{planning_horizons}.nc``:
  Heat pump COP profiles, dims (time, name, heat_source, heat_system).
- ``resources/<run_name>/heat_source_boosting_profiles_base_s_{clusters}_{planning_horizons}.nc``:
  Boosting share b in [0, 1] per urban-central heat source: the fraction of
  extracted source heat routed to the heat pump's cold side. Consumed by
  ``prepare_sector_network`` as the utilisation-link split
  (efficiency = 1 - b to DH, efficiency2 = b to the HP input bus).
- ``resources/<run_name>/heat_source_cooling_profiles_base_s_{clusters}_{planning_horizons}.nc``:
  The solved (or flat) source-side cooling dT_cool [K], dims
  (time, name, heat_source, heat_system). Consumed by
  ``build_geothermal_heat_potential`` for potential scaling.
"""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_heat_source_profiles.central_heating_cop_approximator import (
    CentralHeatingCopApproximator,
)
from scripts.build_heat_source_profiles.decentral_heating_cop_approximator import (
    DecentralHeatingCopApproximator,
)
from scripts.definitions.heat_source import HeatSource, HeatSourceType
from scripts.definitions.heat_system_type import HeatSystemType

logger = logging.getLogger(__name__)


def get_cop(
    heat_system_type: str,
    heat_source: str,
    source_inlet_temperature_celsius: xr.DataArray,
    heat_pump_cooling: float | xr.DataArray,
    sink_outlet_temperature_celsius: xr.DataArray = None,
    sink_inlet_temperature_celsius: xr.DataArray = None,
) -> xr.DataArray:
    """
    Calculate the coefficient of performance (COP) for a heating system.

    Uses different approximation methods depending on the heating system type:
    - Central heating: Thermodynamic model from Jensen et al. (2018)
    - Decentral heating: Quadratic regression from Staffell et al. (2012)

    Parameters
    ----------
    heat_system_type : str
        The type of heating system (e.g., "urban central", "urban decentral", "rural").
    heat_source : str
        The heat source used (e.g., "air", "ground", "ptes", "geothermal").
    source_inlet_temperature_celsius : xr.DataArray
        The inlet temperature of the heat source in Celsius.
    heat_pump_cooling : float | xr.DataArray
        Source-side cooling depth dT_cool in K; sets the source outlet
        temperature (central systems only).
    sink_outlet_temperature_celsius : xr.DataArray, optional
        The outlet temperature of the heat sink (forward temperature) in Celsius.
        Required for central heating systems.
    sink_inlet_temperature_celsius : xr.DataArray, optional
        The inlet temperature of the heat sink in Celsius.
        Required for central heating systems.

    Returns
    -------
    xr.DataArray
        The calculated COP values. Values below 1 are set to zero.
    """
    if HeatSystemType(heat_system_type).is_central:
        cop_params = snakemake.params.heat_pump_cop_approximation_central_heating
        return CentralHeatingCopApproximator(
            sink_outlet_temperature_celsius=sink_outlet_temperature_celsius,
            sink_inlet_temperature_celsius=sink_inlet_temperature_celsius,
            source_inlet_temperature_celsius=source_inlet_temperature_celsius,
            source_outlet_temperature_celsius=source_inlet_temperature_celsius
            - heat_pump_cooling,
            refrigerant=cop_params["refrigerant"],
            delta_t_pinch_point=cop_params[
                "heat_exchanger_pinch_point_temperature_difference"
            ],
            isentropic_compressor_efficiency=cop_params[
                "isentropic_compressor_efficiency"
            ],
            heat_loss=cop_params["heat_loss"],
            min_delta_t_lift=cop_params["min_delta_t_lift"],
        ).cop

    else:
        return DecentralHeatingCopApproximator(
            sink_outlet_temperature_celsius=snakemake.params.heat_pump_sink_T_decentral_heating,
            source_inlet_temperature_celsius=source_inlet_temperature_celsius,
            source_type=heat_source,
        ).cop


def get_source_temperature(
    snakemake_params: dict, snakemake_input: dict, heat_source_name: str
) -> float | xr.DataArray:
    """
    Retrieve the temperature of a heat source.

    Heat sources can have either constant temperatures (specified in config)
    or time-varying temperatures (loaded from input files).

    Parameters
    ----------
    snakemake_params : dict
        Snakemake parameters containing heat_source_temperatures dict.
    snakemake_input : dict
        Snakemake input files containing temperature profile paths.
    heat_source_name : str
        Name of the heat source (e.g., "air", "ground", "geothermal", "ptes", "electrolysis_waste").

    Returns
    -------
    float | xr.DataArray
        Temperature in Celsius. Returns a float for constant-temperature sources
        or an xr.DataArray for time-varying sources.

    Notes
    -----
    Presence of constant-temperature entries in ``heat_source_temperatures``
    is validated at config load time by ``SectorConfig``.
    """
    heat_source = HeatSource(heat_source_name)
    if heat_source.temperature_from_config:
        return snakemake_params["heat_source_temperatures"][heat_source_name]
    elif heat_source.source_type == HeatSourceType.STORAGE:
        # PTES layer temperatures are explicit constants from config.
        ptes_layer_temperatures = snakemake_params["ptes_layer_temperatures"]
        if heat_source_name.startswith("ptes layer"):
            layer_idx = int(heat_source_name.split()[-1])
            return float(ptes_layer_temperatures[layer_idx])
        else:
            return float(ptes_layer_temperatures[0])
    else:
        return xr.open_dataarray(snakemake_input[f"temp_{heat_source_name}"])


def get_source_inlet_temperature(
    source_temperature: float | xr.DataArray,
    central_heating_return_temperature: xr.DataArray,
) -> float | xr.DataArray:
    """
    Determine the effective source inlet temperature for the heat pump.

    For preheating heat sources (source above return temperature), the source
    first preheats the return flow; the heat pump's evaporator then sees the
    source stream at return temperature. Otherwise, the heat pump draws
    directly from the source temperature.

    Notes
    -----
    We assume ideal heat exchangers with no temperature losses.

    Parameters
    ----------
    source_temperature : float | xr.DataArray
        Temperature of the heat source in Celsius.
    central_heating_return_temperature : xr.DataArray
        District heating return temperature in Celsius.

    Returns
    -------
    float | xr.DataArray
        Effective source inlet temperature for the heat pump in Celsius.
    """
    return xr.where(
        source_temperature < central_heating_return_temperature,
        source_temperature,
        central_heating_return_temperature,
    )


def get_sink_inlet_temperature(
    source_temperature: float | xr.DataArray,
    central_heating_return_temperature: xr.DataArray,
) -> float | xr.DataArray:
    """
    Determine the effective sink inlet temperature for the heat pump.

    When the source temperature exceeds the return temperature, the preheater
    raises the return flow to the source temperature and the heat pump lifts
    from there to forward. Otherwise no preheating is used and the heat pump
    receives return-temperature water directly.

    Parameters
    ----------
    source_temperature : float | xr.DataArray
        Temperature of the heat source in Celsius.
    central_heating_return_temperature : xr.DataArray
        District heating return temperature in Celsius.

    Returns
    -------
    float | xr.DataArray
        Effective sink inlet temperature for the heat pump in Celsius.
    """
    return xr.where(
        source_temperature > central_heating_return_temperature,
        source_temperature,
        central_heating_return_temperature,
    )


def get_boosting_profile(
    source_temperature: float | xr.DataArray,
    forward_temperature: xr.DataArray,
    return_temperature: xr.DataArray,
    heat_pump_cooling: float | xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the boosting share: fraction of source heat routed to the HP cold side.

    Per unit of heat extracted from the source stream (cooled from T_source down
    to T_return − ΔT_cool, with equal mass flows on source and sink side), the
    share b enters the heat pump evaporator while (1 − b) directly heats the
    return flow:

        b = 0                                            if T_source ≥ T_forward
        b = 1                                            if T_source ≤ T_return
        b = ΔT_cool / (T_source − T_return + ΔT_cool)    otherwise

    where ΔT_cool is the source-side cooling applied by the heat pump. The
    utilisation link delivers (1 − b) directly to district heating and routes b
    to the heat pump's cold-side input bus, conserving energy:
    Q_DH = Q_source + W_el.

    Parameters
    ----------
    source_temperature : float | xr.DataArray
        Heat source temperature in °C.
    forward_temperature : xr.DataArray
        District heating forward temperature in °C, indexed by (time, name).
    return_temperature : xr.DataArray
        District heating return temperature in °C, indexed by (time, name).
    heat_pump_cooling : float | xr.DataArray
        Source-side temperature drop ΔT_cool through the heat pump evaporator [K].

    Returns
    -------
    xr.DataArray
        Boosting share profile in [0, 1]: 0 = direct use, 1 = full HP boosting.
        Shape matches forward_temperature.
    """
    return xr.where(
        source_temperature >= forward_temperature,
        # no boosting needed if source_temp >= forward_temp
        0.0,
        xr.where(
            source_temperature <= return_temperature,
            # source cannot pre-heat the return flow if at/below return temp:
            # all source heat enters the HP evaporator
            1.0,
            # between return and forward temp the source pre-heats the return
            # flow (share 1 - b) and the HP draws the evaporator share b
            heat_pump_cooling
            / (source_temperature - return_temperature + heat_pump_cooling),
        ).clip(min=0, max=1),
    )


def expand_heat_sources_for_ptes_layers(
    heat_sources_by_system: dict[str, list[str]],
    ptes_layer_temperatures: list[float] | None,
) -> dict[str, list[str]]:
    """
    Replace ``'ptes'`` by per-layer ``ptes layer {l}`` labels when multiple PTES
    layers exist (the per-layer heat-source COP). A single layer keeps the plain
    ``'ptes'`` source unchanged. The input mapping is not mutated.

    The booster heat-pump COP is no longer built here: it is computed inside
    ``PtesApproximator`` (build_ptes_operations), where the discrete deposit layer
    is known, so the COP source-outlet matches the layer the volume lands in.
    """
    num_layers = len(ptes_layer_temperatures)
    if num_layers <= 1:
        return heat_sources_by_system

    layer_sources = [f"ptes layer {layer}" for layer in range(num_layers)]
    return {
        system: [
            source
            for entry in sources
            for source in (layer_sources if entry == "ptes" else [entry])
        ]
        for system, sources in heat_sources_by_system.items()
    }


def resolve_heat_source_cooling(
    cooling_config: float | dict, heat_source_name: str
) -> float:
    """
    Typical source-side cooling dT_cool [K] for one heat source.

    ``heat_source_cooling`` is either a flat scalar for all sources or a
    per-source mapping with a required ``default`` entry, e.g.::

        heat_source_cooling:
          default: 6
          geothermal: 15

    PTES layer sources ('ptes layer N') fall back to a ``ptes`` entry before
    the default.

    Parameters
    ----------
    cooling_config : float | dict
        The ``sector.district_heating.heat_source_cooling`` setting.
    heat_source_name : str
        Name of the heat source (e.g. 'geothermal', 'ptes layer 2').

    Returns
    -------
    float
        Source-side cooling depth in K.
    """
    if not isinstance(cooling_config, dict):
        return float(cooling_config)
    if heat_source_name in cooling_config:
        return float(cooling_config[heat_source_name])
    if heat_source_name.startswith("ptes layer") and "ptes" in cooling_config:
        return float(cooling_config["ptes"])
    return float(cooling_config["default"])


def compute_heat_pump_cooling(
    heat_system_type: str,
    heat_source: str,
    forward_temperature: xr.DataArray,
    source_temperature: float | xr.DataArray,
    source_inlet_temperature: float | xr.DataArray,
    sink_inlet_temperature: float | xr.DataArray,
    initial_cooling: float = 0.0,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
    iteration_log: list | None = None,
) -> xr.DataArray:
    """
    Source-side cooling applied by the heat pump to a preheating heat source.

    The cooling equals ``(COP - 1) / COP * (T_forward - T_source)``. Since the
    COP itself depends on the cooling, we start from ``initial_cooling`` and
    re-evaluate the COP and cooling in turn until the cooling stops changing
    (within ``tolerance``). Hours that need no boost (T_forward <= T_source) and
    infeasible operating points (COP < 1) get zero cooling.

    If ``iteration_log`` is given, each iteration's full (time, name) state is
    appended to it for inspection; iteration 0 holds the ``initial_cooling``
    starting guess.
    """
    lift = (forward_temperature - source_temperature).clip(min=0)
    cooling = initial_cooling

    for iteration in range(max_iterations):
        cop = get_cop(
            heat_system_type=heat_system_type,
            heat_source=heat_source,
            source_inlet_temperature_celsius=source_inlet_temperature,
            heat_pump_cooling=cooling,
            sink_outlet_temperature_celsius=forward_temperature,
            sink_inlet_temperature_celsius=sink_inlet_temperature,
        )

        if iteration_log is not None:
            # ``* 0 +`` broadcasts scalars onto the (time, name) grid.
            frame = (
                xr.Dataset(
                    {
                        "forward_temperature": forward_temperature,
                        "source_temperature": forward_temperature * 0
                        + source_temperature,
                        "heat_pump_cooling": forward_temperature * 0 + cooling,
                        "cop": cop,
                    }
                )
                .to_dataframe()
                .reset_index()
            )
            frame.insert(0, "iteration", iteration)
            frame.insert(0, "heat_source", heat_source)
            frame.insert(0, "heat_system", heat_system_type)
            iteration_log.append(frame)

        updated = (1 - 1 / cop.clip(min=1)) * lift
        if float(abs(updated - cooling).max()) < tolerance:
            return updated
        cooling = updated

    raise RuntimeError(
        f"Heat pump cooling did not converge within {max_iterations} "
        f"iterations for '{heat_source}' ({heat_system_type})."
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_heat_source_profiles",
            clusters=48,
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Collect the per-iteration cooling solve when logging is enabled.
    cooling_iteration_log = (
        [] if snakemake.params.log_heat_pump_cooling_iterations else None
    )

    central_heating_forward_temperature: xr.DataArray = xr.open_dataarray(
        snakemake.input.central_heating_forward_temperature_profiles
    )
    central_heating_return_temperature: xr.DataArray = xr.open_dataarray(
        snakemake.input.central_heating_return_temperature_profiles
    )

    if snakemake.params.layered_ptes:
        heat_sources_by_system = expand_heat_sources_for_ptes_layers(
            heat_sources_by_system=snakemake.params.heat_sources,
            ptes_layer_temperatures=snakemake.params.ptes_layer_temperatures,
        )
    else:
        heat_sources_by_system = snakemake.params.heat_sources

    # Cache the cooling assigned to each urban-central source so the COP
    # profile and the boosting profile stay consistent.
    central_heat_pump_cooling: dict[str, float | xr.DataArray] = {}

    empty_sources = central_heating_forward_temperature.expand_dims(
        heat_source=pd.Index([], name="heat_source")
    ).isel(heat_source=slice(0, 0))

    # Sources appear in several heat systems and again in the boosting
    # profiles below; load each temperature (file) only once.
    source_temperatures: dict[str, float | xr.DataArray] = {}

    cop_all_system_types = []
    cooling_all_system_types = []
    for heat_system_type, heat_sources in heat_sources_by_system.items():
        if not heat_sources:
            cop_all_system_types.append(empty_sources)
            cooling_all_system_types.append(empty_sources)
            continue

        cop_this_system_type = []
        cooling_this_system_type = []
        for heat_source_name in heat_sources:
            if heat_source_name not in source_temperatures:
                source_temperatures[heat_source_name] = get_source_temperature(
                    snakemake_params=snakemake.params,
                    snakemake_input=snakemake.input,
                    heat_source_name=heat_source_name,
                )
            source_temperature_celsius = source_temperatures[heat_source_name]

            source_inlet_temperature_celsius = get_source_inlet_temperature(
                source_temperature=source_temperature_celsius,
                central_heating_return_temperature=central_heating_return_temperature,
            )

            sink_inlet_temperature_celsius = get_sink_inlet_temperature(
                source_temperature=source_temperature_celsius,
                central_heating_return_temperature=central_heating_return_temperature,
            )

            # Default: a fixed, source-typical cooling from heat_source_cooling
            # (scalar or per-source mapping). When heat_pump_cooling_iterative
            # is enabled, preheating central sources instead solve the
            # operating-point-consistent cooling, seeded from that typical
            # value.
            typical_cooling = resolve_heat_source_cooling(
                snakemake.params.heat_source_cooling, heat_source_name
            )
            if (
                snakemake.params.heat_pump_cooling_iterative
                and HeatSystemType(heat_system_type).is_central
                and HeatSource(heat_source_name).supports_preheating
            ):
                heat_pump_cooling = compute_heat_pump_cooling(
                    heat_system_type=heat_system_type,
                    heat_source=heat_source_name,
                    forward_temperature=central_heating_forward_temperature,
                    source_temperature=source_temperature_celsius,
                    source_inlet_temperature=source_inlet_temperature_celsius,
                    sink_inlet_temperature=sink_inlet_temperature_celsius,
                    initial_cooling=typical_cooling,
                    iteration_log=cooling_iteration_log,
                )
            else:
                heat_pump_cooling = xr.full_like(
                    central_heating_forward_temperature,
                    typical_cooling,
                )

            if heat_system_type == HeatSystemType.URBAN_CENTRAL.value:
                central_heat_pump_cooling[heat_source_name] = heat_pump_cooling

            cop_this_system_type.append(
                get_cop(
                    heat_system_type=heat_system_type,
                    heat_source=heat_source_name,
                    source_inlet_temperature_celsius=source_inlet_temperature_celsius,
                    heat_pump_cooling=heat_pump_cooling,
                    sink_outlet_temperature_celsius=central_heating_forward_temperature,
                    sink_inlet_temperature_celsius=sink_inlet_temperature_celsius,
                )
            )
            cooling_this_system_type.append(heat_pump_cooling)

        cop_all_system_types.append(
            xr.concat(
                cop_this_system_type, dim=pd.Index(heat_sources, name="heat_source")
            )
        )
        cooling_all_system_types.append(
            xr.concat(
                cooling_this_system_type,
                dim=pd.Index(heat_sources, name="heat_source"),
            )
        )

    system_dim = pd.Index(heat_sources_by_system.keys(), name="heat_system")
    xr.concat(cop_all_system_types, dim=system_dim).to_netcdf(
        snakemake.output.cop_profiles
    )
    xr.concat(cooling_all_system_types, dim=system_dim).to_netcdf(
        snakemake.output.heat_source_cooling_profiles
    )

    # Write the full per-node, per-timestep Picard trace only when logging is on.
    if cooling_iteration_log:
        iterations_path = (
            Path(snakemake.output.cop_profiles).parent
            / f"heat_pump_cooling_iterations_base_s_{snakemake.wildcards.clusters}_{snakemake.wildcards.planning_horizons}.csv"
        )
        pd.concat(cooling_iteration_log, ignore_index=True).to_csv(
            iterations_path, index=False
        )

    # Boosting profiles are built for urban-central heat sources only. PTES
    # uses dedicated routing in prepare_sector_network and is excluded from
    # the generic boosting pathway when enabled.
    urban_central_heat_sources = [
        hs
        for hs in heat_sources_by_system[HeatSystemType.URBAN_CENTRAL.value]
        if not (
            snakemake.params.ptes_enable
            and HeatSource(hs).source_type == HeatSourceType.STORAGE
        )
    ]

    if urban_central_heat_sources:
        boosting_profiles = xr.concat(
            [
                get_boosting_profile(
                    source_temperature=source_temperatures[heat_source_key],
                    forward_temperature=central_heating_forward_temperature,
                    return_temperature=central_heating_return_temperature,
                    heat_pump_cooling=central_heat_pump_cooling[heat_source_key],
                ).assign_coords(heat_source=heat_source_key)
                for heat_source_key in urban_central_heat_sources
            ],
            dim="heat_source",
        )
    else:
        boosting_profiles = xr.DataArray(
            [],
            coords={"heat_source": []},
            dims=["heat_source"],
        )

    boosting_profiles.to_netcdf(snakemake.output.heat_source_boosting_profiles)
