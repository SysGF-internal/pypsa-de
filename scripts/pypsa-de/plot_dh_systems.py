#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
This script generates district heating system energy balance comparison plots
for PyPSA networks across different scenarios.
"""

import logging
import os
import re
import sys

sys.path.append(os.getcwd())

import matplotlib
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, mock_snakemake
from scripts.sysgf_plot_helpers import (
    apply_carrier_groups,
    clean_label,
    get_colors,
    process_networks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Preferred column ordering for bar chart stacks.
# Demand-side techs on the left, supply-side on the right.
# Includes both carrier-group names and individual carrier names;
# columns absent from the data are silently ignored; unknown columns are
# appended at the end.
TECH_COL_ORDER = [
    # Demand (negative bars, left side)
    "District Heating Demand",
    "low-temperature heat for industry",
    "urban central heat",
    "urban central heat vent",
    # Storage (group names first, then individual fallbacks)
    "TTES",
    "urban central water tanks",
    "urban central water tanks charger",
    "urban central water tanks losses",
    "PTES",
    "urban central water pits",
    "urban central water pits charger",
    "urban central water pits losses",
    # Supply (group names first, then individual fallbacks)
    "Heat Pumps",
    "geothermal heat pump",
    "geothermal heat",
    "urban central electrolysis excess heat pump",
    "A/WSHP",
    "urban central river_water heat pump",
    "urban central sea_water heat pump",
    "urban central air heat pump",
    "urban central ptes heat pump",
    "urban central resistive heater",
    "CHP",
    "urban central gas CHP",
    "urban central solid biomass CHP",
    "urban central lignite CHP",
    "urban central coal CHP",
    "urban central oil CHP",
    "urban central H2 CHP",
    "PtX excess heat",
    "H2 Electrolysis",
    "waste CHP",
    "urban central gas boiler",
    "Fischer-Tropsch",
    "urban central water tanks discharger",
    "urban central water pits discharger",
    "ptes heat",
]

# Fallback colors for grouped technology categories not present in network.carriers
DEFAULT_GROUP_COLORS = {
    "CHP": "#8B4513",
    "Heat Pumps": "#FF8C00",
    "A/WSHP": "#FFA600",
    "District Heating Demand": "#CCCCCC",
}

# Group names (matching keys in the carrier_groups config) that represent
# thermal storage.  Used to exclude storage from the pie chart supply mix
# and to place these groups in the "Storage" legend section.
STORAGE_GROUPS = {"PTES", "TTES", "PTES charge", "PTES discharge", "TTES charge", "TTES discharge"}

# Maps individual storage carrier-group names to a display family name.
# Members of the same family share one colour and one legend entry.
STORAGE_FAMILIES = {
    "PTES charge": "PTES",
    "PTES discharge": "PTES",
    "TTES charge": "TTES",
    "TTES discharge": "TTES",
    "PTES": "PTES",
    "TTES": "TTES",
}

STORAGE_COLUMNS = [
    "urban central ptes heat utilisation",
    "urban central ptes heat",
    "urban central ptes heat preheater",
    "urban central water pits discharger",
    "urban central water pits charger",
    "urban central water tanks discharger",
    "urban central water tanks charger",
    "PTES",
    "PTES charge",
    "PTES discharge",
    "TTES",
    "TTES charge",
    "TTES discharge",
]


# ---------------------------------------------------------------------------
# Custom legend handler
# ---------------------------------------------------------------------------


class HandlerSquare(HandlerPatch):
    """Render legend patch handles as perfect squares."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        size = min(width, height)
        center_x = -xdescent + width / 2
        center_y = -ydescent + height / 2
        p = mpatches.Rectangle(
            (center_x - size / 2, center_y - size / 2),
            size,
            size,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=trans,
        )
        return [p]


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def calc_dh_price_range_subnodes(n, subnodes_only=True):
    """Return demand-weighted district heating price per system.

    Parameters
    ----------
    n : pypsa.Network
    subnodes_only : bool
        If True, restrict to systems with a city name (subnodes).

    Returns
    -------
    pd.Series
        Index: system name (bus label without ' urban central heat' suffix).
    """
    if subnodes_only:
        loads = n.loads_t.p.filter(
            regex=r"DE\d+ \d+ \w+.*(urban central|low-temperature) heat"
        ).clip(lower=0)
        prices = n.buses_t.marginal_price.filter(
            regex=r"DE\d+ \d+ \w+.*urban central heat"
        )
    else:
        loads = n.loads_t.p.filter(
            regex=r"DE\d+ \d+.*(urban central|low-temperature) heat"
        ).clip(lower=0)
        prices = n.buses_t.marginal_price.filter(
            regex=r"DE\d+ \d+.*urban central heat"
        )

    # Merge industry heat load into the urban central heat column
    loads.columns = loads.columns.str.replace(
        "low-temperature heat for industry", "urban central heat"
    )
    loads = loads.T.groupby(loads.columns).sum().T

    weighted_price = loads.mul(prices).sum().div(loads.sum())
    weighted_price.index = weighted_price.index.str.replace(
        " urban central heat", ""
    )
    return weighted_price


def calc_dh_demand_twh(network, systems, subnodes_only=True):
    """Return annual district heating demand in TWh for each system.

    Parameters
    ----------
    network : pypsa.Network
    systems : sequence of str
        System names (bus labels without ' urban central heat' suffix).
    subnodes_only : bool
        If True, match load columns by exact name; otherwise use regex.

    Returns
    -------
    pd.Series indexed by system name.
    """
    demands = []
    for system in systems:
        if subnodes_only:
            uch_cols = [
                c for c in network.loads_t.p.columns
                if c == f"{system} urban central heat"
            ]
            ind_cols = [
                c for c in network.loads_t.p.columns
                if c == f"{system} low-temperature heat for industry"
            ]
        else:
            uch_cols = network.loads_t.p.filter(
                regex=f"{system}.*urban central heat"
            ).columns.tolist()
            ind_cols = network.loads_t.p.filter(
                regex=f"{system}.*low-temperature heat for industry"
            ).columns.tolist()

        def _weighted_sum(cols):
            if not cols:
                return 0.0
            return (
                network.loads_t.p[cols]
                .multiply(network.snapshot_weightings.generators, axis=0)
                .sum()
                .sum()
            )

        total_mwh = _weighted_sum(uch_cols) + _weighted_sum(ind_cols)
        demands.append(abs(total_mwh) / 1e6)  # MWh -> TWh

    return pd.Series(demands, index=list(systems))


def prepare_energy_balance_data(
    network,
    carrier_groups=None,
    drop_losses=False,
    subnodes_only=True,
):
    """Prepare relative (%) DH energy balance data for a single network.

    Parameters
    ----------
    network : pypsa.Network
    carrier_groups : dict[str, list[str]] or None
        Maps group name → list of carrier names to merge into that group.
        Applied in definition order; a carrier already claimed by an earlier
        group is not re-claimed.
    drop_losses : bool
    subnodes_only : bool

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        to_plot_rel : systems × carriers, values as % of total demand.
        dh_prices   : demand-weighted DH price per system.
    """
    eb_uch = (
        network.statistics.energy_balance(groupby=["bus", "carrier", "bus_carrier"])
        .xs("urban central heat", level=3)
        .reset_index()
    )

    if subnodes_only:
        eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+ \w+.*urban"), :]
    else:
        eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+.*urban"), :]

    eb_uch["bus"] = eb_uch["bus"].str.replace(" urban central heat", "")
    eb_uch.drop("component", axis=1, inplace=True)

    # Merge CC variants into their base carrier
    eb_uch["carrier"] = eb_uch["carrier"].str.replace(" CC", "", regex=False)
    eb_uch = eb_uch.groupby(["bus", "carrier"], as_index=False).sum()

    to_plot = eb_uch.set_index(["bus", "carrier"]).unstack(-1)
    to_plot.columns = to_plot.columns.droplevel(0)

    # Drop carriers whose absolute total is < 0.01% of the grand total
    total_contribution = to_plot.abs().sum()
    to_plot = to_plot[
        total_contribution[
            total_contribution > 0.0001 * to_plot.abs().sum().sum()
        ].index
    ]

    # --- Apply carrier groups BEFORE normalization so that grouped
    #     columns (e.g. PTES, TTES) are treated as single technologies
    #     and storage is properly excluded from the 100 % base. ---
    to_plot = apply_carrier_groups(to_plot, carrier_groups)

    if drop_losses:
        loss_cols = [c for c in to_plot.columns if "losses" in c.lower()]
        if loss_cols:
            to_plot = to_plot.drop(columns=loss_cols)

    # Demand denominator: absolute sum of negative values in NON-storage
    # columns.  Storage groups are excluded so they appear as extra bars
    # beyond the ±100 % envelope.
    non_storage_cols = [c for c in to_plot.columns if c not in STORAGE_COLUMNS]
    demand_total = -to_plot[non_storage_cols].clip(upper=0).sum(axis=1)
    demand_total = demand_total.replace(0, np.nan)  # avoid division by zero

    to_plot_rel = to_plot.div(demand_total, axis=0).mul(100)

    dh_prices = calc_dh_price_range_subnodes(network, subnodes_only)
    return to_plot_rel, dh_prices


def get_absolute_energy_balance(network, subnodes_only=False, carrier_groups=None):
    """Return supply-side DH energy balance in TWh across all systems.

    Parameters
    ----------
    network : pypsa.Network
    subnodes_only : bool
    carrier_groups : dict[str, list[str]] or None
        Same carrier grouping applied to the bar charts (see
        prepare_energy_balance_data).

    Returns
    -------
    pd.DataFrame  (systems × carriers, clipped to positive values)
    """
    eb_uch = (
        network.statistics.energy_balance(groupby=["bus", "carrier", "bus_carrier"])
        .xs("urban central heat", level=3)
        .reset_index()
    )

    if subnodes_only:
        eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+ \w+.*urban"), :]
    else:
        eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+.*urban"), :]

    eb_uch["bus"] = eb_uch["bus"].str.replace(" urban central heat", "")
    eb_uch.drop("component", axis=1, inplace=True)
    eb_uch["carrier"] = eb_uch["carrier"].str.replace(" CC", "", regex=False)
    eb_uch = eb_uch.groupby(["bus", "carrier"], as_index=False).sum()

    to_plot = eb_uch.set_index(["bus", "carrier"]).unstack(-1)
    to_plot.columns = to_plot.columns.droplevel(0)
    to_plot = (to_plot / 1e6).clip(lower=0)
    to_plot = apply_carrier_groups(to_plot, carrier_groups)
    return to_plot


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def format_scenario_title(scenario):
    """Return a clean display title for a scenario name."""
    if "NoPTES" in scenario:
        return "No PTES"
    elif "hpboost" in scenario:
        if "35Ctop" in scenario:
            return "PTES with\nbooster heat pump\nto 35°C"
        elif "10Cbottom" in scenario:
            return "PTES with\nbooster heat pump\nto 10°C"
        else:
            return "PTES with\nbooster heat pump"
    elif "rhboost" in scenario:
        return "PTES with\nresistive boosting"
    elif "noboost" in scenario:
        return "PTES with\nno boosting"
    else:
        return scenario.replace("_", " ")


def _clean_label(label):
    """Convert a carrier name to a human-readable legend label."""
    return clean_label(label)


# ---------------------------------------------------------------------------
# Network loading helpers
# ---------------------------------------------------------------------------


# get_colors and process_networks are imported from sysgf_plot_helpers


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_energy_balance_comparison(
    network1,
    network2,
    scenarios,
    output_path,
    colors,
    carrier_groups=None,
    drop_losses=False,
    subnodes_only=True,
):
    """Plot side-by-side DH energy balance comparison for two scenarios.

    Produces a PDF with:
    - Top row  : horizontal stacked bar charts (supply/demand as % of demand).
                 Left subplot (network1): secondary x-axis for DH demand [TWh].
                 Right subplot (network2): secondary x-axis for ΔDH price [EUR/MWh].
    - Bottom row: pie charts of aggregated national DH supply mix.
    - Right side: shared legend.

    Parameters
    ----------
    network1, network2 : pypsa.Network
    scenarios : list[str]  – [scenario1_name, scenario2_name]
    output_path : str
    colors : dict[str, str]
    carrier_groups : dict[str, list[str]] or None
        Explicit carrier-to-group mapping (see prepare_energy_balance_data).
    drop_losses, subnodes_only : bool

    Returns
    -------
    tuple[Figure, list[Axes]]
    """
    plt.rcParams.update({"font.size": 12})

    grouping_kwargs = dict(
        carrier_groups=carrier_groups,
        drop_losses=drop_losses,
        subnodes_only=subnodes_only,
    )

    to_plot_rel1, dh_prices1 = prepare_energy_balance_data(network1, **grouping_kwargs)
    to_plot_rel2, dh_prices2 = prepare_energy_balance_data(network2, **grouping_kwargs)

    dh_price_savings = dh_prices1 - dh_prices2

    # Sort systems by DH demand (highest first), using network1 as reference
    dh_demand_series = calc_dh_demand_twh(network1, to_plot_rel1.index, subnodes_only)
    sorted_systems = dh_demand_series.sort_values(ascending=False).index

    to_plot_rel1 = to_plot_rel1.loc[sorted_systems]
    to_plot_rel2 = to_plot_rel2.loc[sorted_systems]
    dh_price_savings = dh_price_savings.loc[sorted_systems]
    dh_demand_series = dh_demand_series.loc[sorted_systems]

    max_ylim = max(
        to_plot_rel1.clip(lower=0).sum(1).max(),
        to_plot_rel2.clip(lower=0).sum(1).max(),
        -to_plot_rel1.clip(upper=0).sum(1).min(),
        -to_plot_rel2.clip(upper=0).sum(1).min(),
    ) * 1.05

    # --- Layout ---
    fig = plt.figure(figsize=(10, 12))
    ax1 = plt.subplot2grid((10, 12), (0, 0), rowspan=6, colspan=5)
    ax2 = plt.subplot2grid((10, 12), (0, 5), rowspan=6, colspan=5, sharey=ax1)
    pie_ax1 = plt.subplot2grid((10, 12), (7, 0), rowspan=3, colspan=5)
    pie_ax2 = plt.subplot2grid((10, 12), (7, 5), rowspan=3, colspan=5)

    # Reorder columns according to the preferred display order
    def _apply_col_order(df):
        ordered = [c for c in TECH_COL_ORDER if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        return df[ordered]

    to_plot_rel1 = _apply_col_order(to_plot_rel1)
    to_plot_rel2 = _apply_col_order(to_plot_rel2)

    # --- Bar charts ---
    bar_kwargs = dict(stacked=True, color=colors, legend=False, width=0.9, alpha=0.8)
    to_plot_rel1.plot.barh(ax=ax1, **bar_kwargs)
    to_plot_rel2.plot.barh(ax=ax2, **bar_kwargs)

    for ax, scenario in zip([ax1, ax2], scenarios):
        ax.set_title(
            format_scenario_title(scenario),
            fontsize=11, pad=20, ha="center", weight="bold",
        )
        ax.set_xlabel("Demand and supply [%]", fontsize=12)
        ax.axvline(x=0, color="black", linestyle="-")
        ax.axvline(x=-100, color="black", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(x=100, color="black", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlim(-max_ylim, max_ylim)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

    # Y-tick labels on left subplot only; strip "DE0 " prefix for cleaner labels
    yticks = [lbl.get_text().replace("DE0 ", "") for lbl in ax1.get_yticklabels()]
    ax1.set_yticklabels(yticks, fontsize=10)
    ax1.set_ylabel("")
    ax1.tick_params(axis="y", labelleft=True)
    ax2.tick_params(axis="y", labelleft=False, labelright=False)

    # --- Secondary axis: DH Demand [TWh] on ax1 ---
    ax1_demand = ax1.twiny()
    ax1_demand.scatter(
        dh_demand_series.values, range(len(dh_demand_series)),
        s=30, marker="s", facecolor="black", edgecolor="black",
        linewidth=0.2, zorder=20, clip_on=False,
    )
    ax1_demand.axvline(
        x=dh_demand_series.mean(), color="black", linestyle=":", linewidth=2, zorder=5
    )
    ax1_demand.set_xlabel("DH Demand\n[TWh]", fontsize=12)
    ax1_demand.tick_params(axis="x", labelsize=10)
    demand_range = dh_demand_series.max() - dh_demand_series.min()
    padding = demand_range * 0.1 if demand_range > 0 else dh_demand_series.min() * 0.05
    ax1_demand.set_xlim(
        dh_demand_series.min() - padding, dh_demand_series.max() + padding
    )

    # --- Secondary axis: ΔDH Price [EUR/MWh] on ax2 ---
    # Negate so that positive = cheaper in scenario 2 (i.e. savings)
    price_values = -dh_price_savings.values
    ax2_price = ax2.twiny()
    ax2_price.scatter(
        price_values, range(len(dh_price_savings)),
        s=30, marker="^", facecolor="black", edgecolor="black",
        linewidth=0.2, zorder=20, clip_on=False,
    )
    ax2_price.axvline(
        x=price_values.mean(), color="black", linestyle="--", linewidth=2, zorder=5
    )
    ax2_price.axvline(x=0, color="gray", linestyle=":", alpha=0.7, linewidth=1)
    ax2_price.set_xlabel("ΔDH Price\n[EUR MWh$^{-1}$]", fontsize=12, color="black")
    ax2_price.tick_params(axis="x", labelsize=10, colors="black")
    max_abs = max(abs(dh_price_savings.min()), abs(dh_price_savings.max()))
    sym_limit = max_abs + max(max_abs * 0.1, 2.0)
    ax2_price.set_xlim(-sym_limit, sym_limit)

    # --- Pie charts: aggregated national DH supply mix ---

    def _group_supply_for_pie(network):
        """Aggregate supply technologies for pie chart, applying same carrier_groups."""
        abs_data = get_absolute_energy_balance(
            network, subnodes_only=False, carrier_groups=carrier_groups
        )
        result = {}
        for col in abs_data.columns:
            if col in STORAGE_GROUPS or col in STORAGE_FAMILIES:
                continue
            total = abs_data[col].sum()
            if total > 0:
                result[col] = result.get(col, 0) + total
        return result

    for network, pie_ax in zip([network1, network2], [pie_ax1, pie_ax2]):
        grouped_supply = _group_supply_for_pie(network)
        if not grouped_supply:
            continue

        techs = list(grouped_supply)
        values = list(grouped_supply.values())
        tech_colors = [colors.get(t, "#808080") for t in techs]

        wedges, _, autotexts = pie_ax.pie(
            values,
            labels=None,
            colors=tech_colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
            startangle=90,
            textprops={"fontsize": 10},
            wedgeprops={"alpha": 0.8},
            pctdistance=0.65,
        )
        for autotext, wedge in zip(autotexts, wedges):
            autotext.set_color("black")
            autotext.set_weight("bold")
            angle = (wedge.theta1 + wedge.theta2) / 2
            rotation = angle + 180 if 90 <= angle <= 270 else angle
            autotext.set_rotation(rotation)
            autotext.set_horizontalalignment("center")
            autotext.set_verticalalignment("center")

        pie_ax.set_aspect("equal")
        pie_ax.set_title(
            f"Total: {sum(values):.1f} TWh", fontsize=12, pad=5, weight="bold"
        )

    # --- Legend ---
    # Classify plotted columns into supply / demand / storage by their data
    # values and STORAGE_GROUPS membership (driven by carrier_groups config).
    all_cols_ordered = list(dict.fromkeys(
        list(to_plot_rel1.columns) + list(to_plot_rel2.columns)
    ))

    supply_cols, demand_cols, storage_cols = [], [], []
    for col in all_cols_ordered:
        if col in STORAGE_GROUPS or col in STORAGE_FAMILIES:
            storage_cols.append(col)
        else:
            val = 0.0
            if col in to_plot_rel1.columns:
                val += to_plot_rel1[col].sum()
            if col in to_plot_rel2.columns:
                val += to_plot_rel2[col].sum()
            if val < 0:
                demand_cols.append(col)
            else:
                supply_cols.append(col)

    legend_handles, legend_labels = [], []

    def _add_legend_section(title, cols):
        if not cols:
            return
        legend_labels.append(rf"$\bf{{{title.replace(' ', r'\ ')}}}$")
        legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))
        # For storage columns, collapse families into a single legend entry
        seen_families = set()
        for col in cols:
            family = STORAGE_FAMILIES.get(col)
            if family:
                if family in seen_families:
                    continue
                seen_families.add(family)
                color = colors.get(col, "#808080")
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + family)
            else:
                color = colors.get(col, "#808080")
                clean = _clean_label(col).replace("_", " ")
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + clean)

    _add_legend_section("Supply Technologies:", supply_cols)
    _add_legend_section("Demand:", demand_cols)
    _add_legend_section("Storage:", storage_cols)

    legend_labels.append(r"$\bf{Indicators:}$")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))

    for marker, linestyle, label in [
        ("s", "None", "DH Demand [TWh]"),
        ("", ":", "Mean DH Demand"),
        ("^", "None", "ΔDH Price [EUR MWh$^{-1}$]"),
        ("", "--", "Mean ΔDH Price"),
    ]:
        legend_handles.append(
            Line2D(
                [0], [0],
                marker=marker, color="black",
                markeredgecolor="black", markeredgewidth=0.5,
                markersize=6, linestyle=linestyle,
                linewidth=2 if linestyle not in ("None", "") else 0,
            )
        )
        legend_labels.append("  " + label)

    legend_labels = [lbl.replace("_", " ") for lbl in legend_labels]

    fig.legend(
        legend_handles, legend_labels,
        bbox_to_anchor=(0.7, 0.72), loc="center left", frameon=False,
        fontsize=12, ncol=1,
        handler_map={mpatches.Patch: HandlerSquare()},
        handlelength=1.0, handleheight=1.0,
    )

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.95, wspace=0.10, left=0.12, right=0.80, bottom=0.05, hspace=0.15
    )
    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Energy balance comparison saved to {output_path}")
    return fig, [ax1, ax2]


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_dh_systems",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
            scenario1="Baseline",
            scenario2="No_PTES",
        )
    configure_logging(snakemake)

    s1 = snakemake.wildcards.scenario1
    s2 = snakemake.wildcards.scenario2
    run_name = snakemake.params.run
    plotting = snakemake.params.plotting

    networks = process_networks(
        network_files=snakemake.input.networks,
        run_name=run_name,
        scenarios=[s1, s2],
    )

    colors = get_colors(networks, plotting.get("override_tech_colors", {}))
    for group, color in DEFAULT_GROUP_COLORS.items():
        colors.setdefault(group, color)

    # Ensure every carrier group name has a color: use the first listed carrier's color
    carrier_groups = plotting.get("carrier_groups", {})
    for group_name, carrier_list in carrier_groups.items():
        if group_name not in colors:
            for carrier in carrier_list:
                if carrier in colors:
                    colors[group_name] = colors[carrier]
                    break

    # Force all members of a storage family to share one colour
    for member, family in STORAGE_FAMILIES.items():
        if family in colors and member not in colors:
            colors[member] = colors[family]
        elif member in colors and family not in colors:
            colors[family] = colors[member]
    # Ensure charge/discharge pairs within each family use the same colour
    for family_name in set(STORAGE_FAMILIES.values()):
        members = [m for m, f in STORAGE_FAMILIES.items() if f == family_name]
        family_color = None
        for m in members:
            if m in colors:
                family_color = colors[m]
                break
        if family_color:
            for m in members:
                colors[m] = family_color
            colors[family_name] = family_color

    grouping_kwargs = dict(
        carrier_groups=carrier_groups,
        drop_losses=plotting["drop_losses"],
        subnodes_only=plotting["subnodes_only"],
    )

    logger.info(f"Creating energy balance comparison: {s1} vs {s2}")
    plot_energy_balance_comparison(
        networks[s1],
        networks[s2],
        [s1, s2],
        snakemake.output.pdf,
        colors,
        **grouping_kwargs,
    )
    logger.info("District heating system plot generation completed")
