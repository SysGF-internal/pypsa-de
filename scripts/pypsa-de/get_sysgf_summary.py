#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
This script processes PyPSA networks and generates a summary DataFrame
with key metrics for energy system analysis across scenarios.
"""

import logging
import os
import sys

sys.path.append(os.getcwd())

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def calc_ptes_cycles(n, mean=True):
    """Calculate the number of cycles for PTES (water pits) systems."""
    pits_de = n.stores.filter(regex=r"DE0.*water pits", axis=0).query("e_nom_opt > 0")
    if pits_de.empty:
        return 0

    pits_de_e_max_pu = n.stores_t.e_max_pu.reindex(
        pits_de.index, axis=1, fill_value=1
    ).mean()

    discharge = (
        n.links_t.p0.filter(regex=r"DE.*pits discharger")
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
    )
    discharge.index = discharge.index.str.replace(" discharger", "")
    no_cycles = discharge.div(pits_de.e_nom_opt * pits_de_e_max_pu)
    if mean:
        return no_cycles.mean()
    else:
        return no_cycles


def get_pit_capacities_de(n, mode="effective"):
    pits_de = n.stores.filter(regex=r"DE0.*water pits", axis=0).query("e_nom_opt > 0")
    pits_de_e_max_pu = n.stores_t.e_max_pu.reindex(
        pits_de.index, axis=1, fill_value=1
    ).mean()
    if mode == "effective":
        caps = pits_de.e_nom_opt * pits_de_e_max_pu
    elif mode == "volume":
        caps = pits_de.e_nom_opt / 4500 * 70000
    elif mode == "capex":
        caps = pits_de.e_nom_opt * pits_de.capital_cost
    else:
        caps = pits_de.e_nom_opt
    return caps


def calc_average_dh_price_t_ordered(n, aggregate_time=False):
    """Calculate time-ordered average district heating price."""
    dh_price_t = n.buses_t.marginal_price.filter(regex=r"DE0 \d+.*urban central heat")
    dh_wd_t_ordered = n.statistics.withdrawal(
        groupby=["bus", "carrier", "bus_carrier"],
        aggregate_time=False,
        bus_carrier="urban central heat",
    ).filter(like="DE0", axis=0)
    # Drop chargers
    to_drop = dh_wd_t_ordered.filter(like="charger", axis=0).index
    dh_wd_t_ordered = dh_wd_t_ordered.drop(to_drop, axis=0)
    # Drop chargers
    dh_wd_t_ordered = dh_wd_t_ordered.groupby(["bus"]).sum()

    weighted_dh_price_t = (
        dh_price_t.mul(dh_wd_t_ordered.T)
        .sum(1)
        .div(dh_wd_t_ordered.sum())
        .sort_values()
    )
    if aggregate_time:
        weighted_dh_price = (
            weighted_dh_price_t.mul(dh_wd_t_ordered.sum()).sum()
            / dh_wd_t_ordered.sum().sum()
        )
        return weighted_dh_price
    else:
        return weighted_dh_price_t


def calc_average_elec_price_t_ordered(n, aggregate_time=False):
    """Calculate time-ordered average electricity price."""
    elec_price_t = n.buses_t.marginal_price.filter(regex=r"DE0 \d+$")
    elec_wd_t_ordered = n.statistics.withdrawal(
        groupby=["bus", "carrier", "bus_carrier"],
        aggregate_time=False,
        bus_carrier="AC",
    ).filter(like="DE0", axis=0)
    # Drop carrier=='DC' and component=='Line' entries
    to_drop = elec_wd_t_ordered.index[
        (elec_wd_t_ordered.index.get_level_values("carrier") == "DC")
        | (elec_wd_t_ordered.index.get_level_values("component") == "Line")
    ]
    elec_wd_t_ordered = elec_wd_t_ordered.drop(to_drop, axis=0)
    elec_wd_t_ordered = elec_wd_t_ordered.groupby(["bus"]).sum()
    weighted_elec_price_t = (
        elec_price_t.mul(elec_wd_t_ordered.T)
        .sum(1)
        .div(elec_wd_t_ordered.sum())
        .sort_values()
    )
    if aggregate_time:
        weighted_elec_price = (
            weighted_elec_price_t.mul(elec_wd_t_ordered.sum()).sum()
            / elec_wd_t_ordered.sum().sum()
        )
        return weighted_elec_price
    else:
        return weighted_elec_price_t


def calc_curtailment_de(n):
    """Calculate curtailment in TWh."""
    try:
        curtailment = (
            n.statistics.curtailment(groupby=["bus", "carrier"], nice_names=False)
            .xs("Generator", level=0)
            .filter(regex=r"DE.*wind|solar")
            .div(1e6)
            .sum()
        )
        return curtailment
    except Exception:
        return 0


def calc_vres_gen_de(n):
    try:
        gen = (
            n.statistics.energy_balance(groupby=["bus", "carrier"], nice_names=False)
            .xs("Generator", level=0)
            .filter(regex=r"DE.*wind|solar")
            .div(1e6)
            .sum()
        )
        return gen
    except Exception:
        return 0


def calc_heat_venting_de(n):
    """Calculate heat venting in TWh."""
    try:
        heat_venting = (
            n.snapshot_weightings.generators
            @ n.generators_t.p.filter(regex=r"DE0.*heat vent")
        ).sum() / 1e6
        return heat_venting
    except Exception:
        return 0


def get_component_mask(lines_or_links, country, other_countries, bus=0):
    """Create mask for components connecting a country to others."""
    if bus == 0:
        # For links, check both buses
        return lines_or_links.bus0.str.contains(
            country
        ) & lines_or_links.bus1.str.contains("|".join(other_countries))
    elif bus == 1:
        # For links, check the second bus
        return lines_or_links.bus1.str.contains(
            country
        ) & lines_or_links.bus0.str.contains("|".join(other_countries))
    else:
        return (
            lines_or_links.bus0.str.contains(country)
            & lines_or_links.bus1.str.contains("|".join(other_countries))
        ) | (
            lines_or_links.bus0.str.contains("|".join(other_countries))
            & lines_or_links.bus0.str.contains(country)
        )


def calc_ic_capex_correction(n, country):
    """Calculate correction value for interconnection CAPEX assuming equal shares of connected countries."""
    other_countries = n.buses.country.unique()
    other_countries = other_countries[
        (other_countries != "") & (other_countries != country)
    ]

    ic_links_bus0_mask = get_component_mask(n.links, country, other_countries, bus=0)
    ic_links_bus0 = n.links.loc[ic_links_bus0_mask]
    capex_ic_links_bus0 = ic_links_bus0.p_nom_opt.mul(ic_links_bus0.capital_cost)

    ic_links_bus1_mask = get_component_mask(n.links, country, other_countries, bus=1)
    ic_links_bus1 = n.links.loc[ic_links_bus1_mask]
    capex_ic_links_bus1 = ic_links_bus1.p_nom_opt.mul(ic_links_bus1.capital_cost)

    ic_links = pd.concat([capex_ic_links_bus0, capex_ic_links_bus1], axis=0)
    ic_links["capex"] = 0.5 * (
        capex_ic_links_bus1.reindex(ic_links.index, fill_value=0)
        - capex_ic_links_bus0.reindex(ic_links.index, fill_value=0)
    )

    ic_lines_bus0_mask = get_component_mask(n.lines, country, other_countries, bus=0)
    ic_lines_bus0 = n.lines.loc[ic_lines_bus0_mask]
    capex_ic_links_bus0 = ic_lines_bus0.s_nom_opt.mul(ic_lines_bus0.capital_cost)

    ic_lines_bus1_mask = get_component_mask(n.lines, country, other_countries, bus=1)
    ic_lines_bus1 = n.lines.loc[ic_lines_bus1_mask]
    capex_ic_links_bus1 = ic_lines_bus1.s_nom_opt.mul(ic_lines_bus1.capital_cost)

    ic_lines = pd.concat([capex_ic_links_bus0, capex_ic_links_bus1], axis=0)
    ic_lines["capex"] = 0.5 * (
        capex_ic_links_bus1.reindex(ic_lines.index, fill_value=0)
        - capex_ic_links_bus0.reindex(ic_lines.index, fill_value=0)
    )

    return pd.Series(
        {
            "AC": ic_lines.capex.sum(),
            "DC": ic_links.capex.sum(),
        },
        name="capex",
    )


def calc_system_costs_country(n, country):
    """Calculate system costs for a country."""
    s = n.statistics
    capex_country = s.capex(groupby=["bus", "carrier"]).filter(regex=country, axis=0)
    opex_country = s.opex(groupby=["bus", "carrier"]).filter(regex=country, axis=0)

    system_costs_country = capex_country.sum().sum() + opex_country.sum().sum()
    ic_costs_correction = calc_ic_capex_correction(n, country).sum()

    return system_costs_country + ic_costs_correction


def calculate_district_heating_costs(n: pypsa.Network) -> float:
    dh_mp = n.buses_t.marginal_price.filter(regex=r"DE0.*urban central heat")
    dh_loads = n.loads_t.p.filter(regex=r"DE\d.*(urban central|low-temperature) heat")
    dac_load = n.links_t.p1.filter(regex=r"DE0.*urban central DAC")
    all_loads = pd.concat([dh_loads, dac_load], axis=1).fillna(0)
    # Replace all the words following central with " heat" in the column names
    all_loads.columns = all_loads.columns.str.replace(
        r"central.*", "central heat", regex=True
    ).str.replace(r"low-temperature.*", "urban central heat", regex=True)
    # Aggregate columns with same name
    all_loads = all_loads.T.groupby(level=0).sum().T

    # Get snapshot weightings
    snapshot_weightings = n.snapshot_weightings.generators

    # Calculate dh consumer costs
    dh_consumer_costs = snapshot_weightings @ (dh_mp * all_loads)

    return dh_consumer_costs.sum()


def calc_h2_store_capacity(n):
    """Calculate hydrogen storage capacity in TWh."""
    h2_stores = n.stores.filter(regex=r"DE0.*H2 Store", axis=0)
    return h2_stores.e_nom_opt.sum() / 1e6  # TWh


def extract_part(scenario):
    """Extract parameter part from scenario name."""
    for prefix in ["Low", "High", "0.5", "2"]:
        if scenario.startswith(prefix):
            return scenario[len(prefix) :]
    return scenario


def create_summary_df(networks):
    """Create summary dataframe with key metrics from networks."""
    df = pd.DataFrame(
        columns=[
            "scenario",
            "year",
            "total_system_costs_bnEUR",
            "total_system_costs_DE_bnEUR",
            "district_heating_costs_DE_bnEUR",
            "PTES_capacity_TWh",
            "PTES_capacity_TWh_scaled",
            "PTES_capacity_m3",
            "PTES_capacity_GW",
            "PTES_no_cycles",
            "PTES_investment_bn€",
            "TTES_capacity_TWh",
            "TTES_capacity_GW",
            "H2_store_TWh",
            "co2_price_EU_EUR_per_ton",
            "co2_price_DE_EUR_per_ton",
            "dh_price_EUR_per_MWh",
            "electricity_price_EUR_per_MWh",
            "peak_electricity_price_EUR_per_MWh",
            "peak_dh_price_EUR_per_MWh",
            "curtailment_TWh",
            "vres_gen_TWh",
            "relative_curtailment",
            "heat_venting_TWh",
            "solar capacity_GW",
            "onwind_capacity_GW",
            "hp_capacity_GW",
            "booster_hp_capacity",
            "electrolysis_cf",
            "RSHP_cf",
            "GSHP_cf",
            "booster_hp_cf",
            "vRES_capacity_GW",
            "CHP_capacity_GW",
            "resistive_heater_GW",
        ]
    )

    for scenario, years in networks.items():
        networks_scenario = networks[scenario]
        for year, n in networks_scenario.items():
            # Extract metrics from network
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "scenario": scenario,
                            "year": year,
                            "total_system_costs_bnEUR": (
                                n.statistics.capex().sum() + n.statistics.opex().sum()
                            )
                            / 1e9,
                            "total_system_costs_DE_bnEUR": calc_system_costs_country(
                                n, "DE"
                            )
                            / 1e9,
                            "district_heating_costs_DE_bnEUR": (
                                calculate_district_heating_costs(n)
                            )
                            / 1e9,
                            "PTES_capacity_TWh": n.stores.filter(
                                regex=r"DE.*urban central water pits", axis=0
                            )
                            .e_nom_opt.div(1e6)
                            .sum(),
                            "PTES_capacity_GW": n.links.filter(
                                regex=r"DE.*urban central water pits discharger", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "PTES_no_cycles": calc_ptes_cycles(n),
                            "TTES_capacity_TWh": n.stores.filter(
                                regex=r"DE.*urban central water tanks", axis=0
                            )
                            .e_nom_opt.div(1e6)
                            .sum(),
                            "TTES_capacity_GW": n.links.filter(
                                regex=r"DE.*urban central water tanks discharger",
                                axis=0,
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "H2_store_TWh": calc_h2_store_capacity(n),
                            "co2_price_EU_EUR_per_ton": -n.global_constraints.loc[
                                "CO2Limit", "mu"
                            ],
                            "co2_price_DE_EUR_per_ton": -n.global_constraints.loc[
                                "co2_limit-DE", "mu"
                            ],
                            "dh_price_EUR_per_MWh": calc_average_dh_price_t_ordered(
                                n, aggregate_time=True
                            ),
                            "electricity_price_EUR_per_MWh": calc_average_elec_price_t_ordered(
                                n, aggregate_time=True
                            ),
                            "peak_electricity_price_EUR_per_MWh": n.buses_t.marginal_price.filter(
                                regex=r"DE0 \d+$"
                            )
                            .max()
                            .max(),
                            "peak_dh_price_EUR_per_MWh": n.buses_t.marginal_price.filter(
                                regex=r"DE0.*urban central heat"
                            )
                            .max()
                            .max(),
                            "curtailment_TWh": calc_curtailment_de(n),
                            "vres_gen_TWh": calc_vres_gen_de(n),
                            "relative_curtailment": (
                                calc_curtailment_de(n)
                                / (calc_vres_gen_de(n) + calc_curtailment_de(n))
                                if (calc_vres_gen_de(n) + calc_curtailment_de(n)) > 0
                                else 0
                            ),
                            "heat_venting_TWh": calc_heat_venting_de(n),
                            "vRES_capacity_GW": n.generators.filter(
                                regex=r"DE.*(onwind|offwind|solar-)",
                                axis=0,
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "solar_capacity_GW": n.generators.filter(
                                regex=r"DE.*solar", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "wind_capacity_GW": n.generators.filter(
                                regex=r"DE.*(onwind|offwind)", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "CHP_capacity_GW": n.links.filter(regex=r"DE.*CHP", axis=0)
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "H2_CHP_capacity_GW": n.links.filter(
                                regex=r"DE.*H2 CHP", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "resistive_heater_GW": n.links.filter(
                                regex=r"DE.*resistive heater", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "PTES_capacity_TWh_scaled": get_pit_capacities_de(
                                n, mode="effective"
                            ).sum(),
                            "PTES_capacity_m3": get_pit_capacities_de(
                                n, mode="volume"
                            ).sum(),
                            "PTES_investment_bn€": get_pit_capacities_de(
                                n, mode="effective"
                            ).sum(),
                            "booster_hp_GW": (
                                n.links.filter(regex=r"DE.*ptes heat pump", axis=0)
                                .p_nom_opt.div(1e3)
                                .sum()
                                if "hpboost" in scenario
                                else 0
                            ),
                            "booster_hp_cf": (
                                n.statistics.capacity_factor(
                                    groupby=["country", "carrier"]
                                )
                                .xs("DE", level="country")
                                .xs("urban central ptes heat pump", level="carrier")
                                .mean()
                                if "hpboost" in scenario
                                else 0
                            ),
                            "onwind_capacity_GW": n.generators.filter(
                                regex=r"DE.*onwind", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "hp_capacity_GW": n.links.filter(
                                regex="DE.*heat pump", axis=0
                            )
                            .p_nom_opt.mul(1 / n.links_t.efficiency.max())
                            .dropna()
                            .div(1e3)
                            .sum(),
                            "electrolysis_cf": n.statistics.capacity_factor(
                                groupby=["country", "carrier"]
                            )
                            .xs("DE", level="country")
                            .xs("H2 Electrolysis", level="carrier")
                            .mean(),
                            "electrolysis_capacity_GW": n.links.filter(
                                regex="DE.*H2 Electrolysis", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            # "RSHP_cf": n.statistics.capacity_factor(
                            #     groupby=["country", "carrier"]
                            # )
                            # .xs("DE", level="country")
                            # .xs("urban central river_water heat pump", level="carrier")
                            # .mean(),
                            "GSHP_cf": n.statistics.capacity_factor(
                                groupby=["country", "carrier"]
                            )
                            .xs("DE", level="country")
                            .xs("urban central geothermal heat pump", level="carrier")
                            .mean(),
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    # Add parameter extraction for sensitivity analysis
    df["parameter"] = df.scenario.apply(extract_part)
    df["case"] = df.apply(lambda x: x.scenario.replace(x.parameter, ""), axis=1)

    return df


def process_networks(run_name, scenarios, planning_horizons):
    """Load networks and create summary DataFrame."""
    networks = {}
    networks_path = os.path.join("results", run_name)

    logger.info(f"Processing networks for run: {run_name}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Planning horizons: {planning_horizons}")

    for scenario in scenarios:
        scenario_path = os.path.join(networks_path, scenario, "networks")
        if not os.path.exists(scenario_path):
            logger.warning(f"Scenario path {scenario_path} does not exist, skipping")
            continue
        networks[scenario] = {}

        for year in planning_horizons:
            # Find network file for this year
            network_file = None
            for file in os.listdir(scenario_path):
                if file.endswith(f"{year}.nc"):
                    network_file = os.path.join(scenario_path, file)
                    break

            if not network_file:
                logger.warning(
                    f"No network file found for scenario {scenario} and year {year}"
                )
                continue

            # Load network and calculate metrics
            try:
                logger.info(f"Loading network for scenario {scenario} and year {year}")
                n = pypsa.Network(network_file)

                networks[scenario][year] = n
                logger.info(
                    f"Successfully loaded network for scenario {scenario} and year {year}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing network for scenario {scenario} and year {year}: {e}"
                )

    if not networks:
        logger.error("No networks could be loaded")
        return None, None

    summary_df = create_summary_df(networks)

    return networks, summary_df


def main(snakemake):
    """Main function to generate summary from network data."""
    configure_logging(snakemake)

    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons

    output_path = os.path.dirname(snakemake.output.sysgf_summary)
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Generating system summary for run: {run_name}")

    networks, summary_df = process_networks(run_name, scenarios, planning_horizons)

    if not networks:
        logger.error("No networks could be loaded")
        return

    summary_df.to_csv(snakemake.output.sysgf_summary)

    logger.info("Summary generation completed successfully")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
        snakemake = mock_snakemake(
            "get_sysgf_summary",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
        )
    main(snakemake)
