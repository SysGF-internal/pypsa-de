#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 PTES Impact authors
# SPDX-License-Identifier: MIT

"""
PTES system-cost impact overview (1-D scenario comparison).

Generates a two-panel figure:
- Left:  baseline system-cost breakdown (bn EUR/a) for the reference scenario,
         split into German cost components and EU-aggregated neighbours.
- Right: net cost difference (bn EUR/a) vs the reference for every other
         scenario.  Star markers show total net savings; triangle markers
         show delta-DH price.

Technologies that change by less than a configurable threshold are aggregated
as "other technologies".  Interconnection CAPEX is split 50/50 between the
connected countries.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._helpers import configure_logging, mock_snakemake
from scripts.sysgf_plot_helpers import (
    clean_label,
    get_colors as _get_colors_base,
    process_networks,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Technology groupings (carrier -> display group)
# ---------------------------------------------------------------------------

TECH_GROUPINGS = {
    # Transmission
    "AC": "Transmission grid",
    "DC": "Transmission grid",
    # Solar
    "solar": "PV",
    "solar-hsat": "PV",
    # Wind
    "onwind": "Wind power",
    "offwind-dc": "Wind power",
    "offwind-ac": "Wind power",
    # Battery
    "battery": "Battery",
    "home battery": "Battery",
    # Fossil primary
    "oil primary": "Fossil primary",
    "gas primary": "Fossil primary",
}


# ---------------------------------------------------------------------------
# Network loading (same pattern as plot_dh_systems)
# ---------------------------------------------------------------------------


# process_networks imported from sysgf_plot_helpers


# ---------------------------------------------------------------------------
# Interconnection CAPEX correction
# ---------------------------------------------------------------------------


def _component_mask(df, country, others, bus):
    """Mask for lines/links connecting *country* to *others* on a given bus side."""
    pat = "|".join(others)
    if bus == 0:
        return df.bus0.str.contains(country) & df.bus1.str.contains(pat)
    return df.bus1.str.contains(country) & df.bus0.str.contains(pat)


def calc_ic_capex_correction(n, country):
    """Half-share interconnection CAPEX correction for *country*."""
    others = n.buses.country.unique()
    others = others[(others != "") & (others != country)]

    def _half_diff(comp, cap_col):
        m0 = _component_mask(comp, country, others, bus=0)
        m1 = _component_mask(comp, country, others, bus=1)
        c0 = comp.loc[m0, cap_col].mul(comp.loc[m0, "capital_cost"])
        c1 = comp.loc[m1, cap_col].mul(comp.loc[m1, "capital_cost"])
        idx = c0.index.union(c1.index)
        return 0.5 * (
            c1.reindex(idx, fill_value=0) - c0.reindex(idx, fill_value=0)
        )

    link_corr = _half_diff(n.links, "p_nom_opt").sum()
    line_corr = _half_diff(n.lines, "s_nom_opt").sum()
    return pd.Series({"AC": line_corr, "DC": link_corr})


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def get_country_costs(n, country):
    """Return (country_costs, neighbour_total) as Series / scalar (EUR)."""
    capex = n.statistics.capex(groupby=["bus", "carrier"], nice_names=False)
    opex = n.statistics.opex(groupby=["bus", "carrier"], nice_names=False)

    capex_is_country = capex.index.get_level_values(1).str.startswith(country)
    opex_is_country = opex.index.get_level_values(1).str.startswith(country)

    capex_de = capex.loc[capex_is_country].groupby(level=2).sum()
    opex_de = opex.loc[opex_is_country].groupby(level=2).sum()
    capex_eu = capex.loc[~capex_is_country].groupby(level=2).sum()
    opex_eu = opex.loc[~opex_is_country].groupby(level=2).sum()

    ic = calc_ic_capex_correction(n, country)
    capex_de.loc[ic.index] = capex_de.reindex(ic.index, fill_value=0) + ic
    capex_eu.loc[ic.index] = capex_eu.reindex(ic.index, fill_value=0) - ic

    idx = capex_de.index.union(opex_de.index)
    country_costs = (
        capex_de.reindex(idx, fill_value=0) + opex_de.reindex(idx, fill_value=0)
    )

    idx_eu = capex_eu.index.union(opex_eu.index)
    neighbour_total = (
        capex_eu.reindex(idx_eu, fill_value=0)
        + opex_eu.reindex(idx_eu, fill_value=0)
    ).sum()

    return country_costs, neighbour_total


def apply_tech_groupings(s):
    """Map carrier names to display groups using TECH_GROUPINGS."""
    mapping = {carrier: TECH_GROUPINGS.get(carrier, carrier) for carrier in s.index}
    return s.groupby(mapping).sum()


def build_costs_df(networks, country="DE"):
    """Build a DataFrame (scenarios x technologies) in EUR, with 'EU aggregated' column."""
    rows = {}
    for scenario, n in networks.items():
        costs, eu_total = get_country_costs(n, country)
        costs = apply_tech_groupings(costs)
        costs["EU aggregated"] = eu_total
        rows[scenario] = costs
    return pd.DataFrame(rows).T.fillna(0)


# ---------------------------------------------------------------------------
# DH price helpers
# ---------------------------------------------------------------------------


def calc_average_dh_price(n):
    """Demand-weighted average DH price in EUR/MWh for German buses."""
    loads = n.loads_t.p.filter(
        regex=r"DE.*(urban central heat|low-temperature heat for industry)"
    )
    if loads.empty:
        return 0.0
    loads.columns = loads.columns.str.replace(
        "low-temperature heat for industry", "urban central heat"
    )
    loads = loads.T.groupby(level=0).sum().T
    prices = n.buses_t.marginal_price.filter(regex=r"DE.*urban central heat")
    if prices.empty or loads.sum().sum() == 0:
        return 0.0
    w = n.snapshot_weightings.generators
    return loads.mul(prices).mul(w, axis=0).sum().sum() / loads.mul(w, axis=0).sum().sum()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_small_techs(costs_df, savings_df, threshold):
    """Aggregate technologies below *threshold* (fraction of abs sum)
    into 'other technologies'."""
    significant = {"EU aggregated"}
    for _, row in savings_df.iterrows():
        abs_sum = row.abs().sum()
        for tech, val in row.items():
            if abs(val) > threshold * abs_sum:
                significant.add(tech)

    def _agg(df):
        small = [
            c
            for c in df.columns
            if c not in significant and c != "other technologies"
        ]
        if not small:
            return df
        out = df.drop(columns=small)
        out["other technologies"] = df[small].sum(axis=1)
        return out

    return _agg(costs_df), _agg(savings_df)


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------


def format_label(label):
    """Human-readable legend label."""
    return clean_label(label)


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

DEFAULT_COLORS = {
    "EU aggregated": "#cccccc",
    "other technologies": "#999999",
    "Transmission grid": "#00b10f",
    "PV": "#FFC629",
    "Wind power": "#83D8FF",
    "Battery": "#ace37f",
    "Fossil primary": "#8B4513",
}


def get_colors(networks, override_colors=None):
    """Extract carrier colours from the first network, with overrides."""
    colors = _get_colors_base(networks, override_colors)
    colors.update(DEFAULT_COLORS)
    if override_colors:
        colors.update(override_colors)
    return colors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_plot(
    costs_df,
    savings_df,
    reference,
    colors,
    dh_prices,
    output_path,
    threshold=0.025,
):
    """Create the two-panel system-cost impact figure.

    Parameters
    ----------
    costs_df : DataFrame  (scenarios x techs, bn EUR)
    savings_df : DataFrame  (non-ref scenarios x techs, bn EUR)
    reference : str
    colors : dict
    dh_prices : dict[str, float]  (scenario -> EUR/MWh)
    output_path : str
    threshold : float
    """
    # Aggregate small techs
    costs_agg, savings_agg = aggregate_small_techs(costs_df, savings_df, threshold)

    # Order techs: largest baseline contribution at bottom, EU aggregated on top
    baseline = costs_agg.loc[reference].drop("EU aggregated", errors="ignore")
    tech_order = baseline.sort_values(ascending=False).index.tolist()
    if "other technologies" in tech_order:
        tech_order.remove("other technologies")
        tech_order.append("other technologies")
    tech_order.append("EU aggregated")
    # Include techs only in savings
    for t in savings_agg.columns:
        if t not in tech_order:
            tech_order.insert(-1, t)

    scenarios = list(savings_agg.index)
    n_scenarios = len(scenarios)

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(3 + 1.8 * n_scenarios, 7),
        gridspec_kw={
            "width_ratios": [1, max(n_scenarios, 1)],
            "wspace": 0.35,
        },
    )

    # ---- Left panel: baseline stacked bar ----
    bottom = 0.0
    ref_costs = costs_agg.loc[reference]
    for tech in tech_order:
        val = ref_costs.get(tech, 0.0)
        if abs(val) < 1e-6:
            continue
        ax_left.bar(
            0,
            val,
            bottom=bottom,
            color=colors.get(tech, "gray"),
            edgecolor="none",
            width=0.6,
        )
        bottom += val

    de_cost = ref_costs.drop("EU aggregated", errors="ignore").sum()
    total_cost = ref_costs.sum()
    ax_left.annotate(
        f"{total_cost:.0f} (total)\n{de_cost:.0f} (DE)",
        (0, bottom * 0.5),
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        rotation=90,
        path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
    )

    # DH price marker on secondary y-axis
    ref_dh = dh_prices.get(reference, 0)
    if ref_dh:
        ax_left_dh = ax_left.twinx()
        ax_left_dh.plot(
            0,
            ref_dh,
            "o",
            color="white",
            markersize=6,
            markeredgecolor="red",
            markeredgewidth=1.5,
        )
        ax_left_dh.set_ylabel("DH Price [EUR MWh$^{-1}$]", color="red")
        ax_left_dh.tick_params(axis="y", labelcolor="red")
        ax_left_dh.set_ylim(0, ref_dh * 1.4)

    ax_left.set_xticks([0])
    ax_left.set_xticklabels(
        [reference], rotation=45, ha="right", fontweight="bold"
    )
    ax_left.set_ylabel("System Costs [bn EUR a$^{-1}$]")
    ax_left.set_title("Baseline Costs", fontweight="bold", pad=12)
    ax_left.grid(True, alpha=0.3)

    # ---- Right panel: net cost differences ----
    total_net = []
    for j, scenario in enumerate(scenarios):
        row = savings_agg.loc[scenario]
        bottom_pos, bottom_neg = 0.0, 0.0
        for tech in tech_order:
            val = row.get(tech, 0.0)
            if abs(val) < 1e-6:
                continue
            if val > 0:
                ax_right.bar(
                    j,
                    val,
                    bottom=bottom_pos,
                    color=colors.get(tech, "gray"),
                    edgecolor="none",
                    width=0.6,
                )
                bottom_pos += val
            else:
                ax_right.bar(
                    j,
                    val,
                    bottom=bottom_neg,
                    color=colors.get(tech, "gray"),
                    edgecolor="none",
                    width=0.6,
                )
                bottom_neg += val
        net = row.sum()
        total_net.append(net)

        # Annotation: % of German baseline
        pct = (net / de_cost) * 100 if de_cost else 0
        ax_right.annotate(
            f"{pct:+.1f}%",
            (j, 0),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            path_effects=[
                path_effects.withStroke(linewidth=2, foreground="white")
            ],
        )

    # Star markers for net cost difference
    ax_right.scatter(
        range(n_scenarios),
        total_net,
        marker="*",
        s=150,
        facecolor="white",
        edgecolor="black",
        linewidth=2,
        zorder=10,
    )

    # DH price deltas on secondary axis
    dh_deltas = [dh_prices.get(s, ref_dh) - ref_dh for s in scenarios]
    if any(abs(d) > 1e-3 for d in dh_deltas):
        ax_right_dh = ax_right.twinx()
        ax_right_dh.plot(
            range(n_scenarios),
            dh_deltas,
            "^",
            color="white",
            markersize=8,
            markeredgecolor="red",
            markeredgewidth=1.5,
        )
        ax_right_dh.set_ylabel(
            r"$\Delta$DH Price [EUR MWh$^{-1}$]", color="red"
        )
        ax_right_dh.tick_params(axis="y", labelcolor="red")
        max_abs = max(abs(d) for d in dh_deltas) or 1
        ax_right_dh.set_ylim(-max_abs * 1.5, max_abs * 1.5)
        ax_right_dh.axhline(
            y=0, color="red", linestyle="--", alpha=0.5, linewidth=1
        )

    ax_right.set_xticks(range(n_scenarios))
    ax_right.set_xticklabels(scenarios, rotation=45, ha="right")
    ax_right.axhline(y=0, color="black", linewidth=0.8)
    ax_right.set_ylabel("Cost Savings [bn EUR a$^{-1}$]")
    ax_right.set_title(
        f"Net Cost Difference vs {reference}", fontweight="bold", pad=12
    )
    ax_right.grid(True, alpha=0.3)

    # ---- Legend ----
    handles, labels = [], []

    def _section(title):
        handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))
        labels.append(
            rf"$\mathbf{{{title.replace(' ', chr(92) + ' ')}}}$"
        )

    _section("German cost components:")
    for tech in tech_order:
        if tech in ("EU aggregated", "other technologies"):
            continue
        handles.append(
            plt.Rectangle(
                (0, 0), 1, 1, fc=colors.get(tech, "gray"), ec="none"
            )
        )
        labels.append(f"  {format_label(tech)}")

    if (
        "other technologies" in costs_agg.columns
        or "other technologies" in savings_agg.columns
    ):
        _section("Aggregated:")
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=colors.get("other technologies", "gray"),
                ec="none",
            )
        )
        labels.append("  Other technologies")

    if "EU aggregated" in tech_order:
        _section("Other countries:")
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                fc=colors.get("EU aggregated", "gray"),
                ec="none",
            )
        )
        labels.append("  EU aggregated")

    _section("Indicators:")
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markeredgecolor="black",
            markeredgewidth=2,
            markersize=10,
            markerfacecolor="white",
            linestyle="None",
        )
    )
    labels.append("  Net system cost difference")
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="red",
            markeredgewidth=1.5,
            markersize=8,
            markerfacecolor="white",
            linestyle="None",
        )
    )
    labels.append("  DH price (baseline)")
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markeredgecolor="red",
            markeredgewidth=1.5,
            markersize=8,
            markerfacecolor="white",
            linestyle="None",
        )
    )
    labels.append(r"  $\Delta$DH price")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30, top=0.92)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(snakemake):
    configure_logging(snakemake)

    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    plotting = snakemake.params.plotting

    threshold = plotting.get("aggregate_threshold_pct", 0.01)
    if threshold > 1.0:
        threshold /= 100.0
    override_colors = plotting.get("override_tech_colors", {})

    # Determine reference scenario from scenario_comparisons config
    comparisons = plotting.get("scenario_comparisons", [])
    if comparisons:
        # Use the second element of the first comparison as reference
        reference = comparisons[0][1]
    else:
        # Fallback: use the first scenario as reference
        reference = scenarios[0]

    # Load networks
    networks = process_networks(snakemake.input.networks, run_name, scenarios)
    colors = get_colors(networks, override_colors)

    # Build costs table in bn EUR
    costs_df = build_costs_df(networks, country="DE") / 1e9

    # Savings relative to reference
    non_ref = [s for s in scenarios if s != reference]
    savings_df = costs_df.loc[non_ref].subtract(costs_df.loc[reference])

    # DH prices
    dh_prices = {}
    for name, n in networks.items():
        dh_prices[name] = calc_average_dh_price(n)

    create_plot(
        costs_df,
        savings_df,
        reference,
        colors,
        dh_prices,
        snakemake.output.pdf,
        threshold,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(REPO_ROOT)
        snakemake = mock_snakemake(
            "plot_system_cost_comparison",
            configfiles=["dev/config.sysgf.test.yaml"],
        )
    main(snakemake)
