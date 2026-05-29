# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import xarray as xr

from scripts.build_cop_profiles.central_heating_cop_approximator import (
    CentralHeatingCopApproximator,
)

# Number of layers at/above which the layered volume model is used. Fewer layers
# fall back to the simple energy-only model (one store at the top temperature).
LAYERED_MODEL_MIN_LAYERS = 3


class PtesApproximator:
    def __init__(
        self,
        forward_temperature: xr.DataArray,
        return_temperature: xr.DataArray,
        layer_temperatures: list[float] | np.ndarray,
        top_temperature: float,
        bottom_temperature: float,
        design_top_temperature: float,
        design_bottom_temperature: float,
        design_standing_losses: float,
        interlayer_heat_transfer_coefficient: float,
        cop_approximation_params: dict,
        booster_source_dt: float,
        rho=1000,
        # Specific heat in MWh/kg/K to keep rho*c_p*DeltaT in MWh/m3.
        c_p=4184 / 3.6e9,
    ):
        self.forward_temperature = forward_temperature
        self.return_temperature = return_temperature

        layer_temperatures = np.asarray(layer_temperatures, dtype=float)

        self._layer_temperatures = layer_temperatures
        self.num_layers = int(layer_temperatures.size)
        # Operating spread comes from the configured top/bottom temperatures (the
        # physical store boundaries), not from the layer list -- a single-layer
        # configuration [90] would otherwise give a zero spread.
        self.top_temperature = float(top_temperature)
        self.bottom_temperature = float(bottom_temperature)
        # Coldest modelled layer: the m3 -> MWh reference for the volume model.
        self.coldest_layer_temperature = float(layer_temperatures[-1])
        self.design_top_temperature = design_top_temperature
        self.design_bottom_temperature = design_bottom_temperature
        self.design_standing_losses = design_standing_losses
        self.interlayer_heat_transfer_coefficient = interlayer_heat_transfer_coefficient
        self.cop_approximation_params = cop_approximation_params
        self.booster_source_dt = booster_source_dt
        self.rho = rho
        self.c_p = c_p

        self._time_dim = [d for d in forward_temperature.dims if d != "name"][0]

    @property
    def is_layered(self) -> bool:
        """Whether the layered volume model applies (else the energy-only model)."""
        return self.num_layers >= LAYERED_MODEL_MIN_LAYERS

    def _central_heating_cop(
        self,
        sink_outlet,
        sink_inlet,
        source_inlet,
        source_outlet,
    ) -> xr.DataArray:
        """Run the central-heating COP approximator with the configured params."""
        p = self.cop_approximation_params
        return CentralHeatingCopApproximator(
            sink_outlet_temperature_celsius=sink_outlet,
            sink_inlet_temperature_celsius=sink_inlet,
            source_inlet_temperature_celsius=source_inlet,
            source_outlet_temperature_celsius=source_outlet,
            refrigerant=p["refrigerant"],
            delta_t_pinch_point=p["heat_exchanger_pinch_point_temperature_difference"],
            isentropic_compressor_efficiency=p["isentropic_compressor_efficiency"],
            heat_loss=p["heat_loss"],
            min_delta_t_lift=p["min_delta_t_lift"],
        ).cop

    @property
    def layer_temperatures(self) -> np.ndarray:
        """Fixed temperature of each layer [°C], hottest first."""
        return self._layer_temperatures

    @property
    def layer_temperatures_da(self) -> xr.DataArray:
        """Fixed temperature of each layer [°C], hottest first."""
        return xr.DataArray(
            self._layer_temperatures,
            dims=["layer"],
            coords={"layer": np.arange(len(self._layer_temperatures))},
        )

    @property
    def charging_availability(self) -> xr.DataArray:
        """
        Binary charging availability Φ⁺_{l,t}.

        For each timestep, only the layer whose temperature is closest to the forward temperature can receive charge. Returns a DataArray with dimensions (snapshot, name, layer).
        """

        closest_layer_to_forward = np.abs(
            self.forward_temperature - self.layer_temperatures_da
        ).argmin(dim="layer")  # (snapshot, name, layer)

        return xr.concat(
            [closest_layer_to_forward == l for l in range(self.num_layers)],
            dim=pd.Index(np.arange(self.num_layers), name="layer"),
        )

    @property
    def layer_needs_boosting(self) -> xr.DataArray:
        """
        Binary indicator whether a layer requires boosting to reach forward temperature.

        Returns dimensions (snapshot, name, layer).
        """
        return xr.where(self.forward_temperature > self.layer_temperatures_da, 1, 0)

    @property
    def m3_to_mwh(self) -> xr.DataArray:
        """Per-layer conversion from volume state to energy equivalent [MWh/m3]."""
        return (
            self.rho
            * self.c_p
            * (self.layer_temperatures_da - self.coldest_layer_temperature)
        )

    @property
    def e_max_pu(self) -> float:
        """Scalar PTES capacity scaling against design temperature spread."""
        design_delta_t = self.design_top_temperature - self.design_bottom_temperature
        if design_delta_t <= 0:
            raise ValueError(
                "design_top_temperature must be larger than design_bottom_temperature"
            )

        operational_delta_t = self.top_temperature - self.bottom_temperature
        return max(operational_delta_t / design_delta_t, 0.0)

    @property
    def return_temp_layer(self) -> xr.DataArray:
        """Closest layer index to node-wise mean return temperature."""

        return (
            np.abs(self.layer_temperatures_da - self.return_temperature)
            .argmin(dim="layer")
            .astype(int)
        )

    @property
    def return_layer_temperature(self) -> xr.DataArray:
        """Discrete temperature [°C] of each node's return layer. Dims ``(name,)``."""
        return self.layer_temperatures_da.isel(layer=self.return_temp_layer)

    @property
    def preheater_return_layer(self) -> xr.DataArray:
        """
        Reinjection layer index for the discharger's no-boost return volume.

        For each source layer l and node:
          a) T_l > T_ret and l is not the closest-to-T_ret layer: target = closest-to-T_ret
          b) T_l > T_ret and l is the closest-to-T_ret layer: target = min(l + 1, num_layers - 1)
             (next colder modelled layer; prevents the self-loop that produces phantom heat)
          c) T_l <= T_ret: no physical feedback; placeholder = num_layers - 1.
             The discharger's efficiency2 (= 1 - layer_needs_boosting) masks this flow
             to 0 because any layer colder than T_ret always needs boosting
             (T_fwd > T_ret >= T_l).

        Returns dimensions (layer, name).

        Future work: distribute the volume across the two layers adjacent to the
        target temperature, weighted by proximity, instead of snapping to nearest.
        """
        ret_layer = self.return_temp_layer
        next_colder = xr.where(
            ret_layer < self.num_layers - 1, ret_layer + 1, self.num_layers - 1
        )
        layer_idx = self.layer_temperatures_da.layer

        return xr.where(
            layer_idx < ret_layer,
            ret_layer,
            xr.where(
                layer_idx == ret_layer,
                next_colder,
                self.num_layers - 1,
            ),
        ).astype(int)

    @property
    def _hp_inlet_temperature(self) -> xr.DataArray:
        """
        Temperature [°C] at which the discharged volume enters the booster
        evaporator, dims ``(layer, name)``.

        Layers warmer than the (discrete) return layer have already given up their
        above-return heat directly, so they enter at the return-layer temperature;
        layers at/below the return layer enter at their own temperature.
        """
        t_ret_layer = self.return_layer_temperature
        t_layer = self.layer_temperatures_da
        return xr.where(t_layer > t_ret_layer, t_ret_layer, t_layer)

    @property
    def hp_return_layer(self) -> xr.DataArray:
        """
        Layer index corresponding to the booster heat pump outlet, dims
        ``(layer, name)``.

        Non-iterative selection: the booster cools the discharged volume from the
        HP inlet by the configured ``booster_source_dt`` and deposits it at the
        nearest layer to that target temperature that is strictly colder than the
        source layer (so volume always moves down). The deposit -- and hence the
        evaporator enthalpy E_hp, and hence net-energy-exactness -- is fixed by
        temperatures alone; the COP only rescales the elec/heat split, so no
        COP <-> deposit Picard fixed point is needed.

        Future work: a Picard pass would matter only if the source ΔT itself were
        made COP-dependent, e.g. demanding
        ``(T_fwd - T_top) / (T_top - T_ret + dT) == q_hp_out / q_dis``.
        """
        target_temp = self._hp_inlet_temperature - self.booster_source_dt

        ret_val = []
        for l in range(self.num_layers):
            # The water enters the evaporator at the warmer of (source layer,
            # return layer), so the deposit must be strictly colder than that
            # inlet layer (clamped to the bottom layer for the coldest source).
            inlet_idx = np.maximum(l, self.return_temp_layer)
            colder_floor = np.minimum(inlet_idx + 1, self.num_layers - 1)
            closest_layer = abs(
                self.layer_temperatures_da - target_temp.sel(layer=l)
            ).argmin(dim="layer")
            closest_layer = xr.where(
                closest_layer <= inlet_idx, colder_floor, closest_layer
            )
            ret_val.append(closest_layer.astype(int))

        return xr.concat(
            ret_val, dim=pd.Index(np.arange(self.num_layers), name="layer")
        )

    @property
    def _deposit_layer_temperature(self) -> xr.DataArray:
        """Layer temperature [°C] of the booster deposit (snapped), dims ``(layer, name)``."""
        deposit = self.hp_return_layer
        return xr.DataArray(
            np.asarray(self._layer_temperatures)[deposit.values],
            dims=deposit.dims,
            coords=deposit.coords,
        )

    @property
    def _booster_evaporator_dt(self) -> xr.DataArray:
        """
        Per-m3 evaporator temperature drop the booster realises [K], dims
        ``(layer, name)``: ``T_HP_inlet - T_deposit_layer`` using the snapped
        deposit layer, so ``E_hp = rho*c_p*dt`` matches the bottom-referenced store
        debit exactly (net-energy-exact regardless of the COP value).
        """
        return (self._hp_inlet_temperature - self._deposit_layer_temperature).clip(
            min=0.0
        )

    @property
    def booster_cop(self) -> xr.DataArray:
        """
        Per-layer COP of the layered-PTES booster heat pump, dims
        ``(layer, name, time)``.

        Computed directly from the discrete layer geometry (rather than in
        build_cop_profiles) so the evaporator outlet is the *snapped* deposit
        layer temperature -- consistent with the evaporator enthalpy ``E_hp`` used
        for the volume bookkeeping. Geometry:
          - sink_outlet  = T_forward (DH water leaves the condenser at forward T)
          - sink_inlet   = T_layer for layers above the return layer (already
            preheated by the direct heat exchanger), else the return-layer T
          - source_inlet = T_HP_inlet (the discrete temperature the volume enters
            the evaporator at)
          - source_outlet = T(deposit layer) -- the snapped deposit
        """
        t_layer = self.layer_temperatures_da
        t_ret_layer = self.return_layer_temperature
        sink_inlet = xr.where(t_layer > t_ret_layer, t_layer, t_ret_layer)
        return self._central_heating_cop(
            sink_outlet=self.forward_temperature,
            sink_inlet=sink_inlet,
            source_inlet=self._hp_inlet_temperature,
            source_outlet=self._deposit_layer_temperature,
        )

    @property
    def booster_elec_efficiency(self) -> xr.DataArray:
        """
        Electricity drawn per unit heat delivered by the booster HP (``1/COP``),
        dims ``(layer, name, time)`` -- the bus1 efficiency on the reversed-flow
        booster link (bus0 = UCH heat).
        """
        return 1.0 / self.booster_cop.clip(min=1.0 + 1e-3)

    @property
    def booster_volume_efficiency(self) -> xr.DataArray:
        """
        Volume drawn from the hp-source bus per unit heat delivered by the booster
        HP [m3/MWh], dims ``(layer, name, time)``:
        ``(COP - 1) / (COP * E_hp)`` with ``E_hp = rho*c_p*(T_HP_inlet -
        T_deposit)``. The booster deposits the same volume at the hp-return layer
        (``efficiency3 = -booster_volume_efficiency``), conserving mass.
        """
        cop = self.booster_cop.clip(min=1.0 + 1e-3)
        e_hp = self.rho * self.c_p * self._booster_evaporator_dt
        return ((cop - 1.0) / (cop * e_hp).where(e_hp > 0)).fillna(0.0)

    @property
    def discharger_heat_efficiency(self) -> xr.DataArray:
        """
        Direct (above-return) heat delivered to DH per m3 discharged [MWh/m3],
        dims ``(layer, name)``.

        ``rho*c_p * clip(T_layer - T_return_layer, 0)``, referenced to the
        DISCRETE return layer (not the actual return temperature) so the discrete
        LP stays net-energy-exact: delivered direct heat + (booster heat - elec)
        equals the bottom-referenced store debit exactly. Layers at/below the
        return layer get 0 (all their enthalpy goes through the booster).
        """
        drop = (self.layer_temperatures_da - self.return_layer_temperature).clip(
            min=0.0
        )
        return self.rho * self.c_p * drop

    @property
    def charger_efficiency_by_source(self) -> xr.DataArray:
        """
        Volume gained per unit DH heat for the volume-trade charger [m3/MWh].

        Raising 1 m3 from a colder source layer to a hotter destination layer
        draws rho*c_p*(T_dest - T_source) of DH heat, i.e.
        ``m3_to_mwh[dest] - m3_to_mwh[source]``. The efficiency is its inverse,
        defined only where T_source < T_dest (0 otherwise). Dims
        ``(layer_dest, layer_source)`` -- layer-only, node-independent.
        """
        dest = self.m3_to_mwh.rename({"layer": "layer_dest"})
        source = self.m3_to_mwh.rename({"layer": "layer_source"})
        delta = dest - source
        return (1.0 / delta.where(delta > 0)).fillna(0.0)

    @property
    def charger_validity(self) -> xr.DataArray:
        """
        Boolean mask of which volume-trade chargers to wire, dims
        ``(name, layer_dest, layer_source)``.

        A (destination, source) pair is valid at a node iff the destination
        layer is the closest to the forward temperature in at least one snapshot
        (the charging-availability mask) and the source layer is colder than the
        destination. The destination set is therefore exactly the
        availability-mask layers (a subset of "above return level"), each paired
        with every colder source. Per-snapshot gating is handled separately by
        ``charging_availability``.
        """
        dest_available = self.charging_availability.any(dim=self._time_dim).rename(
            {"layer": "layer_dest"}
        )
        t_dest = self.layer_temperatures_da.rename({"layer": "layer_dest"})
        t_source = self.layer_temperatures_da.rename({"layer": "layer_source"})
        return dest_available & (t_source < t_dest)

    @property
    def interlayer_heat_transfer_coefficients(self) -> xr.DataArray:
        """
        Per-layer interlayer heat-transfer coefficient Ψ, dims ``(layer,)``.

        Uses the configured ``interlayer_heat_transfer_coefficient`` directly (the
        per-timestep downward-flux fraction used by the standing-loss interlayer
        constraint in solve_network, ``p_{l->l+1} = -Ψ·e_l``).
        """
        return xr.full_like(
            self.layer_temperatures_da,
            self.interlayer_heat_transfer_coefficient,
            dtype=float,
        )

    @property
    def standing_losses(self) -> np.ndarray:
        """Per-layer standing-loss design value [-]."""
        return np.full(self.num_layers, self.design_standing_losses)

    # ------------------------------------------------------------------
    # Simple energy-only model (num_layers < LAYERED_MODEL_MIN_LAYERS)
    #
    # One MWh store at the top temperature. On discharge, q_dis is split between
    # heat delivered directly to DH and heat routed to the booster HP evaporator;
    # the HP runs only when the forward temperature exceeds the top temperature.
    # No volume, no per-layer bookkeeping.
    # ------------------------------------------------------------------

    @property
    def simple_needs_boosting(self) -> xr.DataArray:
        """Binary indicator that forward T exceeds the store top T. Dims ``(time, name)``."""
        return xr.where(self.forward_temperature > self.top_temperature, 1, 0)

    @property
    def simple_discharger_direct_efficiency(self) -> xr.DataArray:
        """
        Fraction of the store debit delivered directly to DH on discharge, dims
        ``(time, name)``.

        When not boosting (T_fwd <= T_top) all the heat is delivered directly
        (= 1). When boosting, the volume is split between direct delivery as it
        cools T_top -> T_ret and the booster evaporator as it cools further by
        ``booster_source_dt``:
            direct = (T_top - T_ret) / (T_top - T_ret + dT)
        so that direct + hp = 1 and the store debit is conserved.
        """
        direct_drop = self.top_temperature - self.return_temperature
        total = direct_drop + self.booster_source_dt
        boosting_direct = (direct_drop / total.where(total > 0)).fillna(1.0)
        return xr.where(self.simple_needs_boosting == 1, boosting_direct, 1.0)

    @property
    def simple_discharger_hp_efficiency(self) -> xr.DataArray:
        """Fraction of the store debit routed to the booster HP. Dims ``(time, name)``."""
        return 1.0 - self.simple_discharger_direct_efficiency

    @property
    def simple_hp_cop(self) -> xr.DataArray:
        """
        COP of the simple-model booster HP, dims ``(time, name)``. Lifts the DH
        water from the store top temperature to the forward temperature, with the
        evaporator drawing from the mean of the return and bottom temperatures and
        cooling by ``booster_source_dt``.
        """
        source_inlet = (self.return_temperature + self.bottom_temperature) / 2.0
        return self._central_heating_cop(
            sink_outlet=self.forward_temperature,
            sink_inlet=self.top_temperature,
            source_inlet=source_inlet,
            source_outlet=source_inlet - self.booster_source_dt,
        )

    @property
    def simple_hp_elec_efficiency(self) -> xr.DataArray:
        """Electricity drawn per unit heat delivered by the simple HP (``1/COP``)."""
        return 1.0 / self.simple_hp_cop.clip(min=1.0 + 1e-3)

    @property
    def simple_hp_input_efficiency(self) -> xr.DataArray:
        """
        Store heat drawn from the hp-input bus per unit heat delivered by the
        simple HP, dims ``(time, name)``: ``(COP - 1) / COP``.
        """
        cop = self.simple_hp_cop.clip(min=1.0 + 1e-3)
        return (cop - 1.0) / cop

    def to_dataset(self) -> xr.Dataset:
        """Export the pre-computed PTES parameters as a single xr.Dataset."""
        layer_coord = np.arange(self.num_layers)

        ds = xr.Dataset(
            {
                "layer_temperatures": xr.DataArray(
                    self.layer_temperatures,
                    dims=["layer"],
                    coords={"layer": layer_coord},
                ),
                "return_temperature_layer": self.return_temp_layer,
                "m3_to_mwh": self.m3_to_mwh,
                "e_max_pu": self.e_max_pu,
            },
            attrs={
                "num_layers": self.num_layers,
                "is_layered": int(self.is_layered),
                "booster_source_dt": self.booster_source_dt,
                "top_temperature": self.top_temperature,
                "bottom_temperature": self.bottom_temperature,
                "storage_height": 15,  # m, assumed for interlayer coefficient calculation
                "design_top_temperature": self.design_top_temperature,
                "design_bottom_temperature": self.design_bottom_temperature,
                "design_standing_losses": self.design_standing_losses,
                "rho": self.rho,
                "c_p": self.c_p,
            },
        )

        if self.is_layered:
            ds["charging_availability"] = self.charging_availability
            ds["layer_needs_boosting"] = self.layer_needs_boosting
            ds["charger_efficiency_by_source"] = self.charger_efficiency_by_source
            ds["charger_validity"] = self.charger_validity
            ds["discharger_heat_efficiency"] = self.discharger_heat_efficiency
            ds["preheater_return_layer"] = self.preheater_return_layer
            ds["hp_return_layer"] = self.hp_return_layer
            ds["booster_elec_efficiency"] = self.booster_elec_efficiency
            ds["booster_volume_efficiency"] = self.booster_volume_efficiency
            ds["interlayer_heat_transfer_coefficients"] = (
                self.interlayer_heat_transfer_coefficients
            )
            ds["standing_losses"] = xr.DataArray(
                self.standing_losses,
                dims=["layer"],
                coords={"layer": layer_coord},
            )
        else:
            ds["simple_needs_boosting"] = self.simple_needs_boosting
            ds["simple_discharger_direct_efficiency"] = (
                self.simple_discharger_direct_efficiency
            )
            ds["simple_discharger_hp_efficiency"] = self.simple_discharger_hp_efficiency
            ds["simple_hp_elec_efficiency"] = self.simple_hp_elec_efficiency
            ds["simple_hp_input_efficiency"] = self.simple_hp_input_efficiency

        return ds
