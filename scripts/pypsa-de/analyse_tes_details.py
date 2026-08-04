# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Create detailed NoTES/TES comparison plots and a TES sanity audit.

The script complements ``analyse_tes_impact.py``.  It focuses on quantities
needed to diagnose the TES formulation rather than on a broad scenario
dashboard:

* spatially aggregated German district-heating dispatch and demand;
* TES charge, discharge, state of charge and auxiliary boosting electricity;
* accounting-system-cost changes by technology;
* DE versus other-country positions for TES investment, DH consumer bills and
  total system cost; and
* a technology-by-technology sanity table for TTES, PTES and ATES.

Example:
python scripts/pypsa-de/analyse_tes_details.py \
    --results-dir results/20260803_allfeatures_24H \
    --resources-dir resources/20260803_allfeatures_24H
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
import xarray as xr  # noqa: E402

logger = logging.getLogger(__name__)

TES = {
    "TTES": {
        "store": "urban central water tanks",
        "charger": "urban central water tanks charger",
        "discharger": "urban central water tanks discharger",
        "booster": None,
        "cost_row": "central water tank storage",
        "color": "#4C78A8",
    },
    "PTES": {
        "store": "urban central water pits",
        "charger": "urban central water pits charger",
        "discharger": "urban central water pits discharger",
        "booster": "urban central ptes heat pump",
        "cost_row": "central water pit storage",
        "color": "#F58518",
    },
    "ATES": {
        "store": "urban central aquifer thermal energy storage",
        "charger": "urban central aquifer thermal energy storage charger",
        "discharger": "urban central aquifer thermal energy storage discharger",
        "booster": "urban central ates heat pump",
        "cost_row": None,
        "color": "#54A24B",
    },
}

HEAT_LABELS = {
    "urban central geothermal heat utilisation": "Geothermal direct",
    "urban central geothermal heat pump": "Geothermal HP",
    "urban central river_water heat pump": "River HP",
    "urban central lake_water heat pump": "Lake HP",
    "urban central sea_water heat pump": "Sea HP",
    "urban central air heat pump": "Air HP",
    "urban central resistive heater": "Resistive heater",
    "urban central gas CHP": "Gas CHP",
    "urban central H2 CHP": "H2 CHP",
    "urban central H2 retrofit CHP": "H2 retrofit CHP",
    "urban central solid biomass CHP": "Biomass CHP",
    "urban central coal CHP": "Coal CHP",
    "urban central oil CHP": "Oil CHP",
    "waste CHP": "Waste CHP",
    "urban central water tanks discharger": "TTES discharge",
    "urban central water pits discharger": "PTES discharge",
    "urban central aquifer thermal energy storage discharger": "ATES discharge",
    "urban central electrolysis_waste heat utilisation": "Electrolysis waste heat",
    "urban central fischer_tropsch_waste heat utilisation": "FT waste heat",
    "urban central electrolysis_waste heat pump": "Electrolysis waste HP",
}

CHARGE_LABELS = {
    cfg["charger"]: f"{technology} charge" for technology, cfg in TES.items()
}

TES_COST_PATTERN = re.compile(
    r"urban central (?:water tanks|water pits|aquifer thermal energy storage|"
    r"ptes|ates)"
)


def discover_networks(results_dir: Path) -> dict[tuple[str, str], Path]:
    """Return solved networks keyed by (scenario, planning horizon)."""
    found = {}
    for path in sorted(results_dir.glob("*/networks/*.nc")):
        match = re.search(r"_(\d{4})\.nc$", path.name)
        if match:
            found[(path.parents[1].name, match.group(1))] = path
    return found


def load_cases(
    paths: dict[tuple[str, str], Path],
) -> dict[tuple[str, str], pypsa.Network]:
    cases = {}
    for key, path in paths.items():
        logger.info("Reading %s", path)
        cases[key] = pypsa.Network(path)
    return cases


def region_from_bus(bus: pd.Series | pd.Index) -> np.ndarray:
    """Split component attribution into Germany and all remaining/shared buses."""
    values = pd.Index(bus).astype(str)
    return np.where(values.str.startswith("DE"), "DE", "Other countries/shared")


def weighted_energy(n: pypsa.Network, power: pd.Series | pd.DataFrame) -> float:
    """Return objective-weighted energy in MWh."""
    weights = n.snapshot_weightings.objective
    if isinstance(power, pd.Series):
        return float(power.mul(weights, axis=0).sum())
    return float(power.mul(weights, axis=0).sum().sum())


def district_heat_timeseries(
    n: pypsa.Network, country: str = "DE"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return positive supply, negative TES charge and final demand [MW]."""
    balance = n.statistics.energy_balance(
        aggregate_time=False,
        groupby=["bus", "carrier"],
        nice_names=False,
        bus_carrier="urban central heat",
    )
    buses = balance.index.get_level_values("bus").astype(str)
    balance = balance[buses.str.startswith(country)]

    demand_rows = balance.index.get_level_values("component") == "Load"
    demand = -balance[demand_rows].sum(axis=0)

    by_carrier = balance.groupby(level="carrier").sum().T
    charge_cols = by_carrier.columns.intersection(CHARGE_LABELS)
    charge = by_carrier[charge_cols].clip(upper=0).rename(columns=CHARGE_LABELS)

    supply = by_carrier.clip(lower=0)
    supplied_energy = supply.mul(n.snapshot_weightings.objective, axis=0).sum()
    supply = supply.loc[:, supplied_energy > 1e3]
    supply = supply.rename(columns=lambda x: HEAT_LABELS.get(x, short_carrier(x)))
    supply = supply.T.groupby(level=0).sum().T
    return supply, charge, demand


def short_carrier(carrier: str) -> str:
    """Compact carrier names without hiding the underlying technology."""
    label = str(carrier).replace("urban central ", "DH ").replace("_", " ")
    label = label.replace(" heat utilisation", " direct")
    label = label.replace(" heat pump", " HP")
    return label


def reduce_plot_columns(
    df: pd.DataFrame, weights: pd.Series, keep: int = 12
) -> pd.DataFrame:
    """Keep the largest annual contributors and combine the remainder."""
    if df.shape[1] <= keep:
        return df
    annual = df.abs().mul(weights, axis=0).sum().sort_values(ascending=False)
    main = annual.index[:keep]
    out = df[main].copy()
    remainder = df.drop(columns=main).sum(axis=1)
    if remainder.abs().max() > 1e-6:
        out["Other heat"] = remainder
    return out


def tes_timeseries(n: pypsa.Network, country: str = "DE") -> dict[str, pd.DataFrame]:
    """Spatially aggregate TES operating time series [MW, MWh]."""
    charge = pd.DataFrame(index=n.snapshots)
    discharge = pd.DataFrame(index=n.snapshots)
    soc = pd.DataFrame(index=n.snapshots)
    auxiliary_heat = pd.DataFrame(index=n.snapshots)
    auxiliary_electricity = pd.DataFrame(index=n.snapshots)

    for technology, cfg in TES.items():
        stores = n.stores.index[
            (n.stores.carrier == cfg["store"])
            & n.stores.index.str.startswith(country)
            & ~n.stores.index.str.contains("layer")
        ]
        chargers = n.links.index[
            (n.links.carrier == cfg["charger"]) & n.links.index.str.startswith(country)
        ]
        dischargers = n.links.index[
            (n.links.carrier == cfg["discharger"])
            & n.links.index.str.startswith(country)
            & ~n.links.index.str.contains("layer")
        ]

        charge[technology] = n.links_t.p0.reindex(columns=chargers).sum(axis=1)
        discharge[technology] = -n.links_t.p1.reindex(columns=dischargers).sum(axis=1)
        soc[technology] = n.stores_t.e.reindex(columns=stores).sum(axis=1)

        booster = cfg["booster"]
        booster_links = (
            n.links.index[
                (n.links.carrier == booster) & n.links.index.str.startswith(country)
            ]
            if booster
            else pd.Index([])
        )
        if len(booster_links):
            auxiliary_heat[technology] = (
                -n.links_t.p0.reindex(columns=booster_links).clip(upper=0).sum(axis=1)
            )
            auxiliary_electricity[technology] = (
                n.links_t.p1.reindex(columns=booster_links).clip(lower=0).sum(axis=1)
            )
        else:
            auxiliary_heat[technology] = 0.0
            auxiliary_electricity[technology] = 0.0

    return {
        "charge_MW": charge,
        "discharge_MW": discharge,
        "soc_MWh": soc,
        "auxiliary_heat_MW": auxiliary_heat,
        "auxiliary_electricity_MW": auxiliary_electricity,
    }


def system_cost_frame(n: pypsa.Network, scenario: str, horizon: str) -> pd.DataFrame:
    """Accounting cost by component, bus, carrier and cost type [EUR/a]."""
    frames = []
    for cost_type in ("capex", "opex"):
        values = getattr(n.statistics, cost_type)(
            groupby=["bus", "carrier"], nice_names=False
        )
        frame = values.rename("value_EUR_per_a").reset_index()
        frame["cost_type"] = cost_type
        frame["scenario"] = scenario
        frame["horizon"] = horizon
        frame["region"] = region_from_bus(frame["bus"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cost_deltas(costs: pd.DataFrame, reference: str, comparison: str) -> pd.DataFrame:
    keys = ["horizon", "region", "component", "carrier", "cost_type"]
    pivot = costs.pivot_table(
        index=keys,
        columns="scenario",
        values="value_EUR_per_a",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for scenario in (reference, comparison):
        if scenario not in pivot:
            pivot[scenario] = 0.0
    pivot["delta_EUR_per_a"] = pivot[comparison] - pivot[reference]
    pivot["technology"] = pivot["carrier"].map(cost_technology)
    return pivot


def cost_technology(carrier: str) -> str:
    carrier = str(carrier)
    if "water tanks" in carrier:
        return "TTES"
    if "water pits" in carrier or "ptes" in carrier:
        return "PTES"
    if "aquifer thermal" in carrier or "ates" in carrier:
        return "ATES"
    return short_carrier(carrier)


def dh_consumer_position(n: pypsa.Network) -> pd.DataFrame:
    """DH load, bill and load-weighted price by region."""
    loads = n.loads[
        (n.loads.carrier == "urban central heat")
        & n.loads.bus.isin(n.buses.index[n.buses.carrier == "urban central heat"])
    ]
    if loads.empty:
        return pd.DataFrame()
    demand = n.get_switchable_as_dense("Load", "p_set").reindex(columns=loads.index)
    prices = n.buses_t.marginal_price.reindex(columns=loads.bus)
    prices.columns = loads.index
    energy = demand.mul(n.snapshot_weightings.objective, axis=0)

    rows = []
    load_regions = pd.Series(region_from_bus(loads.bus), index=loads.index)
    for region, selected in load_regions.groupby(load_regions).groups.items():
        region_energy = float(energy[selected].sum().sum())
        bill = float((energy[selected] * prices[selected]).sum().sum())
        rows.append(
            {
                "region": region,
                "demand_MWh": region_energy,
                "consumer_cost_EUR_per_a": bill,
                "average_price_EUR_per_MWh": bill / region_energy
                if region_energy
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def position_table(
    costs: pd.DataFrame,
    cases: dict[tuple[str, str], pypsa.Network],
    reference: str,
    comparison: str,
) -> pd.DataFrame:
    """Build DE/other positions for TES investment, DH bills and system cost."""
    rows = []
    grouped_system = costs.groupby(["scenario", "horizon", "region"], as_index=False)[
        "value_EUR_per_a"
    ].sum()
    for row in grouped_system.itertuples(index=False):
        rows.append(
            {
                "scenario": row.scenario,
                "horizon": row.horizon,
                "region": row.region,
                "position": "Accounting system cost",
                "value_EUR_per_a": row.value_EUR_per_a,
            }
        )

    tes_capex = costs[
        (costs.cost_type == "capex")
        & costs.carrier.astype(str).str.match(TES_COST_PATTERN)
    ]
    tes_capex = tes_capex.groupby(["scenario", "horizon", "region"], as_index=False)[
        "value_EUR_per_a"
    ].sum()
    for row in tes_capex.itertuples(index=False):
        rows.append(
            {
                "scenario": row.scenario,
                "horizon": row.horizon,
                "region": row.region,
                "position": "TES annualised investment",
                "value_EUR_per_a": row.value_EUR_per_a,
            }
        )

    for (scenario, horizon), n in cases.items():
        for row in dh_consumer_position(n).itertuples(index=False):
            rows.append(
                {
                    "scenario": scenario,
                    "horizon": horizon,
                    "region": row.region,
                    "position": "DH consumer bill",
                    "value_EUR_per_a": row.consumer_cost_EUR_per_a,
                }
            )

    values = pd.DataFrame(rows)
    complete_index = pd.MultiIndex.from_product(
        [
            sorted(values.horizon.unique()),
            ["DE", "Other countries/shared"],
            [
                "TES annualised investment",
                "DH consumer bill",
                "Accounting system cost",
            ],
            [reference, comparison],
        ],
        names=["horizon", "region", "position", "scenario"],
    )
    values = (
        values.groupby(["horizon", "region", "position", "scenario"], as_index=True)[
            "value_EUR_per_a"
        ]
        .sum()
        .reindex(complete_index, fill_value=0.0)
        .unstack("scenario")
        .reset_index()
    )
    values["change_EUR_per_a"] = values[comparison] - values[reference]
    values["change_pct"] = np.where(
        values[reference].abs() > 0,
        100 * values["change_EUR_per_a"] / values[reference].abs(),
        np.nan,
    )
    return values


def read_cost_row(resources_dir: Path, horizon: str, technology: str) -> pd.Series:
    path = resources_dir / "TESon" / f"costs_{horizon}_processed.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    costs = pd.read_csv(path, index_col=0)
    return (
        costs.loc[technology] if technology in costs.index else pd.Series(dtype=float)
    )


def current_resource_potential(
    resources_dir: Path, horizon: str, technology: str, country: str | None
) -> float:
    """Return the physical-potential input for the current vintage [MWh]."""
    settings = {
        "PTES": ("ptes_potentials", "ptes_potential"),
        "ATES": ("ates_potentials", "ates_potential"),
    }
    if technology not in settings:
        return np.nan
    stem, column = settings[technology]
    matches = list((resources_dir / "TESon").glob(f"{stem}_*_{horizon}.csv"))
    if not matches:
        return np.nan
    potential = pd.read_csv(matches[0], index_col=0, usecols=["name", column])[column]
    de_mask = potential.index.astype(str).str.startswith("DE")
    potential = potential[de_mask if country else ~de_mask]
    return float(potential.sum())


def temperature_summary(
    resources_dir: Path,
    horizon: str,
    n: pypsa.Network,
    country: str | None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Summarise DH temperature profiles and hot-temperature exceedance."""
    out = {
        "forward_temperature_mean_C": np.nan,
        "forward_temperature_min_C": np.nan,
        "forward_temperature_max_C": np.nan,
        "return_temperature_mean_C": np.nan,
        "forward_above_hot_node_hour_share": np.nan,
    }
    pattern = f"central_heating_{{kind}}_temperature_profiles_*_{horizon}.nc"
    data = {}
    for kind in ("forward", "return"):
        matches = list((resources_dir / "TESon").glob(pattern.format(kind=kind)))
        if not matches:
            continue
        da = xr.open_dataarray(matches[0])
        if "time" in da.dims:
            frame = da.transpose("time", "name").to_pandas()
        else:
            # Return temperature is static in the current workflow and only
            # carries the spatial ``name`` dimension.
            frame = da.to_pandas().to_frame().T
        de_mask = frame.columns.astype(str).str.startswith("DE")
        frame = frame.loc[:, de_mask if country else ~de_mask]
        data[kind] = frame
    if "forward" in data and not data["forward"].empty:
        forward = data["forward"]
        out["forward_temperature_mean_C"] = float(forward.mean().mean())
        out["forward_temperature_min_C"] = float(forward.min().min())
        out["forward_temperature_max_C"] = float(forward.max().max())
        if threshold is not None:
            weights = n.snapshot_weightings.objective.reindex(forward.index)
            exceedance = (forward > threshold).astype(float)
            out["forward_above_hot_node_hour_share"] = float(
                exceedance.mul(weights, axis=0).sum().sum()
                / weights.sum()
                / len(forward.columns)
            )
    if "return" in data and not data["return"].empty:
        out["return_temperature_mean_C"] = float(data["return"].mean().mean())
    return out


def tes_audit(
    n: pypsa.Network,
    horizon: str,
    resources_dir: Path,
    country: str = "DE",
) -> pd.DataFrame:
    """Calculate technology-level TES sanity metrics for one solved network."""
    rows = []
    water_heat_capacity = 1.16222e-3  # MWh/(m3 K)
    dh = n.meta.get("sector", {}).get("district_heating", {})
    ptes = dh.get("ptes", {})
    ates = dh.get("ates", {})
    heat_temperatures = dh.get("heat_source_temperatures", {})

    for region, prefix in (("DE", country), ("Other countries/shared", None)):
        for technology, cfg in TES.items():
            store_mask = (
                n.stores.carrier == cfg["store"]
            ) & ~n.stores.index.str.contains("layer")
            if prefix:
                store_mask &= n.stores.index.str.startswith(prefix)
            else:
                store_mask &= ~n.stores.index.str.startswith(country)
            stores = n.stores[store_mask]

            def links_for(carrier: str | None) -> pd.DataFrame:
                if carrier is None:
                    return n.links.iloc[0:0]
                mask = n.links.carrier == carrier
                mask &= (
                    n.links.index.str.startswith(prefix)
                    if prefix
                    else ~n.links.index.str.startswith(country)
                )
                return n.links[mask]

            chargers = links_for(cfg["charger"])
            dischargers = links_for(cfg["discharger"])
            boosters = links_for(cfg["booster"])

            capacity = float(stores.e_nom_opt.sum())
            charge = (
                n.links_t.p0.reindex(columns=chargers.index).clip(lower=0).sum(axis=1)
            )
            discharge = (
                -n.links_t.p1.reindex(columns=dischargers.index)
                .clip(upper=0)
                .sum(axis=1)
            )
            charge_energy = weighted_energy(n, charge)
            discharge_energy = weighted_energy(n, discharge)

            auxiliary_heat = pd.Series(0.0, index=n.snapshots)
            auxiliary_electricity = pd.Series(0.0, index=n.snapshots)
            if not boosters.empty:
                auxiliary_heat = (
                    -n.links_t.p0.reindex(columns=boosters.index)
                    .clip(upper=0)
                    .sum(axis=1)
                )
                auxiliary_electricity = (
                    n.links_t.p1.reindex(columns=boosters.index)
                    .clip(lower=0)
                    .sum(axis=1)
                )

            soc = n.stores_t.e.reindex(columns=stores.index).sum(axis=1)
            soc_swing = float(soc.max() - soc.min()) if not soc.empty else 0.0
            potential = stores.e_nom_max.replace(np.inf, np.nan)
            potential_total = (
                float(potential.sum()) if potential.notna().all() else np.nan
            )
            current_potential = current_resource_potential(
                resources_dir, horizon, technology, prefix
            )

            cost_row = (
                read_cost_row(resources_dir, horizon, cfg["cost_row"])
                if cfg["cost_row"]
                else pd.Series(dtype=float)
            )
            store_capital_cost = (
                float(np.average(stores.capital_cost, weights=stores.e_nom_opt))
                if capacity > 0
                else float(stores.capital_cost.mean())
                if not stores.empty
                else np.nan
            )
            onight = (
                float(
                    np.average(
                        stores.onight_cost.dropna(),
                        weights=stores.loc[stores.onight_cost.notna(), "e_nom_opt"],
                    )
                )
                if "onight_cost" in stores
                and stores.onight_cost.notna().any()
                and stores.loc[stores.onight_cost.notna(), "e_nom_opt"].sum() > 0
                else float(cost_row.get("investment", np.nan))
            )

            if technology == "TTES":
                delta_t = float(cost_row.get("temperature difference", 55.0))
                hot_temperature = np.nan
                cold_temperature = np.nan
                heat_capacity = water_heat_capacity
                temperature_basis = "implicit DH supply level; cost-data delta T"
            elif technology == "PTES":
                hot_temperature = float(ptes.get("top_temperature", 90.0))
                cold_temperature = float(ptes.get("bottom_temperature", 35.0))
                delta_t = hot_temperature - cold_temperature
                heat_capacity = water_heat_capacity
                temperature_basis = "fixed PTES top/bottom configuration"
            else:
                hot_temperature = float(heat_temperatures.get("ates", 90.0))
                temp = temperature_summary(
                    resources_dir, horizon, n, prefix, hot_temperature
                )
                cold_temperature = temp["return_temperature_mean_C"]
                delta_t = hot_temperature - cold_temperature
                heat_capacity = (
                    float(ates.get("aquifer_volumetric_heat_capacity", 2600)) / 3.6e6
                )
                temperature_basis = (
                    "90 C heat-source model; potential uses DH forward/return"
                )

            temp = temperature_summary(
                resources_dir,
                horizon,
                n,
                prefix,
                hot_temperature if np.isfinite(hot_temperature) else None,
            )
            energy_density = heat_capacity * delta_t if delta_t > 0 else np.nan
            volume = capacity / energy_density if energy_density > 0 else np.nan
            annual_energy_capex = float((stores.e_nom_opt * stores.capital_cost).sum())
            power_components = pd.concat([chargers, dischargers, boosters])
            annual_power_capex = float(
                (power_components.p_nom_opt * power_components.capital_cost).sum()
            )
            total_capex = annual_energy_capex + annual_power_capex
            discharge_power = float(dischargers.p_nom_opt.sum())
            charger_power = float(chargers.p_nom_opt.sum())
            standing_loss = (
                float(stores.standing_loss.mean()) if not stores.empty else 0.0
            )

            issues = []
            cycles = discharge_energy / capacity if capacity else np.nan
            potential_use = capacity / potential_total if potential_total else np.nan
            current_potential_use = (
                capacity / current_potential if current_potential else np.nan
            )
            aux_heat_energy = weighted_energy(n, auxiliary_heat)
            if capacity > 1e3 and store_capital_cost == 0:
                issues.append("zero energy-capacity CAPEX")
            if np.isfinite(potential_use) and potential_use > 0.99:
                issues.append("capacity at potential limit")
            if np.isfinite(current_potential_use) and current_potential_use > 1.01:
                issues.append("cumulative capacity exceeds current physical potential")
            if capacity > 1e3 and cycles < 0.25:
                issues.append("fewer than 0.25 cycles/a")
            if (
                discharge_energy > 1e3
                and temp["forward_above_hot_node_hour_share"] > 0.01
                and aux_heat_energy < 1.0
            ):
                issues.append("temperature lift expected but boosting is zero")
            if (
                capacity > 0
                and technology in {"PTES", "ATES"}
                and "onight_cost" in stores
                and stores.onight_cost.isna().all()
            ):
                issues.append("overnight-cost metadata missing")

            rows.append(
                {
                    "horizon": horizon,
                    "region": region,
                    "technology": technology,
                    "energy_capacity_MWh": capacity,
                    "volume_m3": volume,
                    "charger_power_MW": charger_power,
                    "discharger_power_MW": discharge_power,
                    "duration_h": capacity / discharge_power
                    if discharge_power
                    else np.nan,
                    "charged_MWh_per_a": charge_energy,
                    "discharged_MWh_per_a": discharge_energy,
                    "annual_cycles": cycles,
                    "soc_swing_MWh": soc_swing,
                    "annual_cycles_per_soc_swing": discharge_energy / soc_swing
                    if soc_swing
                    else np.nan,
                    "charge_to_discharge_efficiency": discharge_energy / charge_energy
                    if charge_energy
                    else np.nan,
                    "soc_swing_fraction": soc_swing / capacity if capacity else np.nan,
                    "potential_MWh": potential_total,
                    "potential_utilisation": potential_use,
                    "current_vintage_physical_potential_MWh": current_potential,
                    "capacity_to_current_physical_potential": current_potential_use,
                    "number_of_capacity_vintages": int(
                        stores.loc[stores.e_nom_opt > 1e-6, "build_year"].nunique()
                    ),
                    "hot_temperature_C": hot_temperature,
                    "cold_temperature_C": cold_temperature,
                    "temperature_difference_K": delta_t,
                    "temperature_basis": temperature_basis,
                    "forward_temperature_mean_C": temp["forward_temperature_mean_C"],
                    "forward_temperature_max_C": temp["forward_temperature_max_C"],
                    "forward_above_hot_node_hour_share": temp[
                        "forward_above_hot_node_hour_share"
                    ],
                    "standing_loss_per_hour": standing_loss,
                    "retained_after_one_year": (1 - standing_loss) ** 8760,
                    "store_overnight_CAPEX_EUR_per_kWh": onight / 1000,
                    "store_annualised_CAPEX_EUR_per_kWh_a": store_capital_cost / 1000,
                    "store_overnight_CAPEX_EUR_per_m3": onight * energy_density,
                    "store_annualised_CAPEX_EUR_per_m3_a": store_capital_cost
                    * energy_density,
                    "annual_energy_CAPEX_EUR_per_a": annual_energy_capex,
                    "annual_power_CAPEX_EUR_per_a": annual_power_capex,
                    "charger_annualised_CAPEX_EUR_per_kW_a": float(
                        chargers.capital_cost.mean() / 1000
                    )
                    if not chargers.empty
                    else np.nan,
                    "discharger_annualised_CAPEX_EUR_per_kW_a": float(
                        dischargers.capital_cost.mean() / 1000
                    )
                    if not dischargers.empty
                    else np.nan,
                    "total_annualised_CAPEX_EUR_per_kWh_a": total_capex
                    / capacity
                    / 1000
                    if capacity
                    else np.nan,
                    "annualised_CAPEX_per_discharged_MWh": total_capex
                    / discharge_energy
                    if discharge_energy
                    else np.nan,
                    "auxiliary_boost_heat_MWh_per_a": aux_heat_energy,
                    "auxiliary_electricity_MWh_per_a": weighted_energy(
                        n, auxiliary_electricity
                    ),
                    "booster_power_MW": float(boosters.p_nom_opt.sum()),
                    "sanity_flags": "; ".join(issues) if issues else "none",
                }
            )
    return pd.DataFrame(rows)


def plot_district_heat(
    cases: dict[tuple[str, str], pypsa.Network],
    scenarios: list[str],
    horizons: list[str],
    stem: str,
) -> None:
    fig, axes = plt.subplots(
        len(scenarios), len(horizons), figsize=(15, 7), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)
    colors = plt.get_cmap("tab20").colors
    for row, scenario in enumerate(scenarios):
        for col, horizon in enumerate(horizons):
            ax = axes[row, col]
            n = cases[(scenario, horizon)]
            supply, charge, demand = district_heat_timeseries(n)
            supply = reduce_plot_columns(supply, n.snapshot_weightings.objective)
            if not supply.empty:
                ax.stackplot(
                    supply.index,
                    *(supply[c] / 1e3 for c in supply),
                    labels=supply.columns,
                    colors=colors[: supply.shape[1]],
                    alpha=0.9,
                )
            if not charge.empty:
                ax.stackplot(
                    charge.index,
                    *(charge[c] / 1e3 for c in charge),
                    labels=charge.columns,
                    colors=[TES[c.split()[0]]["color"] for c in charge.columns],
                    alpha=0.55,
                )
            ax.plot(demand.index, demand / 1e3, color="black", lw=1.1, label="Demand")
            ax.axhline(0, color="0.25", lw=0.6)
            ax.set_title(f"{scenario}, {horizon}")
            ax.grid(axis="y", alpha=0.25)
            if col == 0:
                ax.set_ylabel("DE district heat [GW]\n(charge below zero)")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(0.995, 0.5),
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    save_figure(fig, f"tes_dh_dispatch_{stem}")


def plot_tes_operation(
    cases: dict[tuple[str, str], pypsa.Network],
    comparison: str,
    horizons: list[str],
    stem: str,
) -> None:
    fig, axes = plt.subplots(2, len(horizons), figsize=(15, 7), sharex="col")
    axes = np.atleast_2d(axes)
    for col, horizon in enumerate(horizons):
        n = cases[(comparison, horizon)]
        ts = tes_timeseries(n)
        top = axes[0, col]
        bottom = axes[1, col]
        technologies = list(TES)
        top.stackplot(
            n.snapshots,
            *(ts["discharge_MW"][t] / 1e3 for t in technologies),
            labels=[f"{t} discharge" for t in technologies],
            colors=[TES[t]["color"] for t in technologies],
            alpha=0.85,
        )
        top.stackplot(
            n.snapshots,
            *(-ts["charge_MW"][t] / 1e3 for t in technologies),
            labels=[f"{t} charge" for t in technologies],
            colors=[TES[t]["color"] for t in technologies],
            alpha=0.4,
        )
        auxiliary = ts["auxiliary_electricity_MW"].sum(axis=1) / 1e3
        top.plot(
            n.snapshots,
            auxiliary,
            color="#B279A2",
            lw=1.4,
            label="Auxiliary electricity",
        )
        top.axhline(0, color="0.25", lw=0.6)
        top.set_title(f"TES operation, {horizon}")
        top.set_ylabel("Power [GW]\n(charge below zero)")
        top.grid(axis="y", alpha=0.25)

        for technology in technologies:
            carrier = TES[technology]["store"]
            capacity = n.stores.loc[
                (n.stores.carrier == carrier)
                & n.stores.index.str.startswith("DE")
                & ~n.stores.index.str.contains("layer"),
                "e_nom_opt",
            ].sum()
            if capacity > 0:
                bottom.plot(
                    n.snapshots,
                    100 * ts["soc_MWh"][technology] / capacity,
                    color=TES[technology]["color"],
                    label=technology,
                )
        bottom.set_ylabel("Aggregate state of charge [%]")
        bottom.set_ylim(bottom=-2)
        bottom.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(0.995, 0.5),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    save_figure(fig, f"tes_operation_{stem}")


def plot_cost_deltas(deltas: pd.DataFrame, horizons: list[str], stem: str) -> None:
    values = deltas.groupby(["horizon", "technology"])["delta_EUR_per_a"].sum()
    values = values.unstack(fill_value=0.0).reindex(horizons)
    importance = values.abs().max().sort_values(ascending=False)
    main = importance.index[:14]
    plot_values = values[main].copy()
    if len(importance) > len(main):
        plot_values["Other"] = values.drop(columns=main).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(horizons))
    positive_bottom = np.zeros(len(horizons))
    negative_bottom = np.zeros(len(horizons))
    colors = plt.get_cmap("tab20").colors
    for i, technology in enumerate(plot_values):
        values_bn = plot_values[technology].to_numpy() / 1e9
        bottom = np.where(values_bn >= 0, positive_bottom, negative_bottom)
        ax.bar(
            x,
            values_bn,
            bottom=bottom,
            label=technology,
            color=colors[i % len(colors)],
            width=0.62,
        )
        positive_bottom += np.where(values_bn > 0, values_bn, 0)
        negative_bottom += np.where(values_bn < 0, values_bn, 0)
    net = values.sum(axis=1).to_numpy() / 1e9
    ax.scatter(x, net, color="black", marker="D", s=45, label="Net change", zorder=5)
    for xpos, value in zip(x, net, strict=False):
        ax.annotate(
            f"{value:+.2f}",
            (xpos, value),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xticks(x, horizons)
    ax.set_ylabel("TES - NoTES accounting cost [bn EUR/a]\n(negative = saving)")
    ax.set_title("System-cost change by technology")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_figure(fig, f"tes_system_cost_delta_{stem}")


def save_figure(fig: plt.Figure, name: str) -> None:
    images = Path("images")
    images.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(images / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote images/%s.{png,pdf}", name)


def write_position_latex(table: pd.DataFrame, path: Path) -> None:
    labels = {
        "TES annualised investment": "TES investment",
        "DH consumer bill": "DH consumer bill",
        "Accounting system cost": "System cost",
    }
    lines = [
        "% generated by scripts/pypsa-de/analyse_tes_details.py",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Horizon & Position & DE change [bn EUR/a] & Other/shared change [bn EUR/a] \\\\",
        "\\midrule",
    ]
    for horizon in sorted(table.horizon.unique()):
        subset = table[table.horizon == horizon]
        for position in labels:
            by_region = subset[subset.position == position].set_index("region")
            de = by_region.at["DE", "change_EUR_per_a"] / 1e9
            other = by_region.at["Other countries/shared", "change_EUR_per_a"] / 1e9
            lines.append(
                f"{horizon} & {labels[position]} & {de:+.3f} & {other:+.3f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_timeseries(
    cases: dict[tuple[str, str], pypsa.Network],
    output_dir: Path,
    scenarios: list[str],
    horizons: list[str],
) -> None:
    def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reset_index()
        frame.columns = [
            "::".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in frame.columns
        ]
        return frame

    dh_frames = []
    tes_frames = []
    for scenario in scenarios:
        for horizon in horizons:
            n = cases[(scenario, horizon)]
            supply, charge, demand = district_heat_timeseries(n)
            frame = pd.concat(
                {
                    "supply_MW": supply,
                    "charge_MW": charge,
                    "demand_MW": demand.to_frame("total final heat demand"),
                },
                axis=1,
            )
            frame.insert(0, "horizon", horizon)
            frame.insert(0, "scenario", scenario)
            frame.index.name = "snapshot"
            dh_frames.append(flatten_columns(frame))
            if scenario == scenarios[-1]:
                tes = tes_timeseries(n)
                tes_frame = pd.concat(tes, axis=1)
                tes_frame.insert(0, "horizon", horizon)
                tes_frame.index.name = "snapshot"
                tes_frames.append(flatten_columns(tes_frame))
    pd.concat(dh_frames, ignore_index=True).to_csv(
        output_dir / "tes_dh_dispatch_timeseries.csv", index=False
    )
    pd.concat(tes_frames, ignore_index=True).to_csv(
        output_dir / "tes_operation_timeseries.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--resources-dir", required=True, type=Path)
    parser.add_argument("--reference", default="TESoff")
    parser.add_argument("--comparison", default="TESon")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    logging.getLogger("pypsa").setLevel(logging.WARNING)

    paths = discover_networks(args.results_dir)
    cases = load_cases(paths)
    horizons = sorted({horizon for _, horizon in cases})
    scenarios = [args.reference, args.comparison]
    missing = [
        (scenario, horizon)
        for scenario in scenarios
        for horizon in horizons
        if (scenario, horizon) not in cases
    ]
    if missing:
        raise SystemExit(f"Missing solved networks: {missing}")

    output_dir = args.results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.results_dir.name

    costs = pd.concat(
        [
            system_cost_frame(n, scenario, horizon)
            for (scenario, horizon), n in cases.items()
        ],
        ignore_index=True,
    )
    deltas = cost_deltas(costs, args.reference, args.comparison)
    deltas.to_csv(output_dir / "tes_system_cost_delta_by_technology.csv", index=False)

    positions = position_table(costs, cases, args.reference, args.comparison)
    positions.to_csv(output_dir / "tes_de_other_positions.csv", index=False)
    write_position_latex(positions, output_dir / "tes_de_other_positions.tex")

    audit = pd.concat(
        [
            tes_audit(cases[(args.comparison, horizon)], horizon, args.resources_dir)
            for horizon in horizons
        ],
        ignore_index=True,
    )
    audit.to_csv(output_dir / "tes_sanity_metrics.csv", index=False)
    write_timeseries(cases, output_dir, scenarios, horizons)

    plot_district_heat(cases, scenarios, horizons, stem)
    plot_tes_operation(cases, args.comparison, horizons, stem)
    plot_cost_deltas(deltas, horizons, stem)

    print("\nDE TES sanity summary")
    columns = [
        "horizon",
        "technology",
        "energy_capacity_MWh",
        "annual_cycles",
        "store_overnight_CAPEX_EUR_per_kWh",
        "store_annualised_CAPEX_EUR_per_m3_a",
        "auxiliary_electricity_MWh_per_a",
        "sanity_flags",
    ]
    print(audit[audit.region == "DE"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
