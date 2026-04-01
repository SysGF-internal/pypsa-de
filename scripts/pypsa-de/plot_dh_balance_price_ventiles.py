#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
Plot district heating balance shares by electricity price ventile
for a single scenario.

Produces a figure showing stacked bars of DH technology shares (%)
grouped by electricity price ventile (percentile bin), with a secondary
y-axis showing cumulative heat generation (excl. storage) in TWh.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pypsa

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._helpers import configure_logging, mock_snakemake
from scripts.sysgf_plot_helpers import apply_carrier_groups, clean_label

logger = logging.getLogger(__name__)

DEFAULT_N_VENTILES = 20


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_temporal_dh_balance(n, carrier_groups=None, subnodes_only=True):
    """Return temporal DH energy balance (timesteps x carriers) in MW.

    Positive = supply, negative = demand/storage-charge.
    """
    eb = n.statistics.energy_balance(
        aggregate_time=False,
        bus_carrier="urban central heat",
        groupby=["bus", "carrier"],
    )

    if subnodes_only:
        mask = eb.index.get_level_values("bus").str.contains(r"DE\d+ \d+ \w+")
    else:
        mask = eb.index.get_level_values("bus").str.contains(r"DE\d+ \d+")
    eb = eb.loc[mask]

    # Merge CC variants
    idx = eb.index.to_frame()
    idx["carrier"] = idx["carrier"].str.replace(" CC", "", regex=False)
    eb.index = pd.MultiIndex.from_frame(idx)
    eb = eb.groupby(["carrier"]).sum()  # sum across buses and CC variants

    # Transpose: rows=timesteps, columns=carriers
    df = eb.T
    df = apply_carrier_groups(df, carrier_groups)

    # Drop tiny carriers (<0.01% of total absolute energy)
    total = df.abs().sum()
    threshold = 0.0001 * total.sum()
    df = df.loc[:, total > threshold]

    return df


def get_electricity_price_t(n):
    """Demand-weighted average electricity price per timestep (€/MWh)."""
    prices = n.buses_t.marginal_price.filter(regex=r"DE\d+ \d+$")
    loads = n.loads_t.p.filter(regex=r"DE\d+ \d+$").clip(lower=0)
    if prices.empty or loads.empty:
        return pd.Series(dtype=float, index=n.snapshots)
    total_load = loads.sum(axis=1)
    total_load = total_load.replace(0, np.nan)
    return loads.mul(prices).sum(axis=1).div(total_load)


def bin_by_price_ventile(balance, elec_price, n_bins):
    """Group the DH balance by electricity price ventiles.

    Parameters
    ----------
    balance : DataFrame (timesteps x carriers, MW)
    elec_price : Series (timesteps, €/MWh)
    n_bins : int  number of ventile bins

    Returns
    -------
    shares : DataFrame (n_bins x carriers) — percentage shares within each bin
    cumulative_twh : Series (n_bins) — cumulative heat generation (excl. storage) in TWh
    bin_labels : list[str] — tick labels showing the price range
    """
    # Align price to balance index
    price = elec_price.reindex(balance.index)

    # Determine snapshot weight (hours per timestep)
    freq = balance.index.to_series().diff().median()
    hours_per_step = freq.total_seconds() / 3600 if pd.notna(freq) else 1.0

    # Assign each timestep to a price ventile (0..n_bins-1)
    quantile_edges = np.linspace(0, 1, n_bins + 1)
    price_quantiles = price.quantile(quantile_edges).values
    # pd.cut with computed bin edges
    bin_idx = pd.cut(
        price,
        bins=price_quantiles,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    # actual number of bins may be less if duplicates were dropped
    actual_n_bins = int(bin_idx.max()) + 1 if bin_idx.notna().any() else n_bins

    # Sum energy per bin (MW * hours -> MWh)
    balance_mwh = balance.mul(hours_per_step)
    balance_mwh["_bin"] = bin_idx
    binned = balance_mwh.groupby("_bin", observed=False).sum()
    # Ensure all bins present
    binned = binned.reindex(range(actual_n_bins), fill_value=0.0)

    # Compute shares: percentage of total absolute energy within each bin
    bin_totals = binned.abs().sum(axis=1)
    shares = binned.div(bin_totals, axis=0).mul(100).fillna(0)

    # Cumulative heat generation excl. storage: only positive (supply) carriers
    supply = binned.clip(lower=0)
    cumulative_twh = supply.sum(axis=1).cumsum().div(1e6)  # MWh -> TWh

    # Build tick labels: "< X" for each bin upper edge
    bin_labels = []
    for i in range(actual_n_bins):
        lo = price_quantiles[i]
        hi = price_quantiles[min(i + 1, len(price_quantiles) - 1)]
        if i == actual_n_bins - 1:
            bin_labels.append(f"< {hi:.0f}")
        else:
            bin_labels.append(f"< {hi:.0f}")

    return shares, cumulative_twh, bin_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dh_ventiles(ax, shares, cumulative_twh, bin_labels, colors, title=""):
    """Plot stacked-bar shares + cumulative heat generation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    shares : DataFrame (n_bins x carriers, %)
    cumulative_twh : Series (n_bins, TWh)
    bin_labels : list[str]
    colors : dict carrier -> colour
    title : str

    Returns
    -------
    ax2 : secondary y-axis Axes
    """
    x = np.arange(len(shares))
    bar_width = 0.85

    # Separate positive (supply) and negative (demand) shares
    pos_cols = shares.columns[shares.sum() >= 0]
    neg_cols = shares.columns[shares.sum() < 0]

    # Sort: demand by magnitude, supply by variance (consistent with temporal plot)
    neg_sorted = (
        shares[neg_cols].sum().abs().sort_values(ascending=True).index
        if len(neg_cols) else neg_cols
    )
    pos_sorted = (
        shares[pos_cols].var().sort_values(ascending=True).index
        if len(pos_cols) else pos_cols
    )
    col_order = list(neg_sorted) + list(pos_sorted)

    # Plot positive stack
    bottom_pos = np.zeros(len(shares))
    for col in col_order:
        vals = shares[col].clip(lower=0).values
        if np.abs(vals).max() < 1e-6:
            continue
        c = colors.get(col, "#808080")
        ax.bar(x, vals, bottom=bottom_pos, width=bar_width, color=c,
               edgecolor="white", linewidth=0.3, label=col, zorder=2)
        bottom_pos += vals

    # Plot negative stack
    bottom_neg = np.zeros(len(shares))
    for col in col_order:
        vals = shares[col].clip(upper=0).values
        if np.abs(vals).max() < 1e-6:
            continue
        c = colors.get(col, "#808080")
        ax.bar(x, vals, bottom=bottom_neg, width=bar_width, color=c,
               edgecolor="white", linewidth=0.3, zorder=2)
        bottom_neg += vals

    ax.axhline(0, color="black", linewidth=0.5, zorder=3)

    # Secondary axis: cumulative heat generation
    ax2 = ax.twinx()
    ax2.plot(x, cumulative_twh.values, "o", color="white", markersize=5,
             markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    ax2.set_ylabel("Cumulative heat generation\nwithout storage [TWh]", fontsize=10)
    ax2.set_ylim(0, cumulative_twh.max() * 1.15 if cumulative_twh.max() > 0 else 1)

    # Main axis formatting
    ax.set_xlim(-0.5, len(shares) - 0.5)
    ax.set_ylabel("District heating share [%]", fontsize=10)
    ax.set_xlabel("Electricity price ventiles [€/MWh]", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=9)

    if title:
        ax.set_title(title, fontsize=12)

    return ax2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(snakemake):
    configure_logging(snakemake)

    plotting = snakemake.params.plotting
    carrier_groups = plotting.get("carrier_groups", {})
    subnodes_only = plotting.get("subnodes_only", True)
    override_colors = plotting.get("override_tech_colors", {})
    n_bins = plotting.get("n_price_ventiles", DEFAULT_N_VENTILES)

    # Load network
    n = pypsa.Network(snakemake.input.network)

    # Compute data
    logger.info("Computing temporal DH balance")
    balance = get_temporal_dh_balance(n, carrier_groups, subnodes_only)
    elec_price = get_electricity_price_t(n)

    logger.info("Binning by electricity price ventile (%d bins)", n_bins)
    shares, cumulative_twh, bin_labels = bin_by_price_ventile(
        balance, elec_price, n_bins
    )

    if shares.empty:
        logger.warning("No DH balance data — skipping plot")
        return

    # Colors from network carriers + overrides
    colors = n.carriers.color.dropna().to_dict()
    colors.update(override_colors)
    for group_name, carrier_list in carrier_groups.items():
        if group_name not in colors:
            for carrier in carrier_list:
                if carrier in colors:
                    colors[group_name] = colors[carrier]
                    break

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax2 = plot_dh_ventiles(ax, shares, cumulative_twh, bin_labels, colors)

    # Legend
    all_carriers = list(shares.columns)
    handles = []
    for carrier in all_carriers:
        c = colors.get(carrier, "#808080")
        handles.append(mpatches.Patch(facecolor=c, edgecolor="white",
                                      linewidth=0.5, label=carrier))
    labels = [clean_label(h.get_label()) for h in handles]

    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), ncol=min(5, len(handles)),
               frameon=False, fontsize=8, columnspacing=1.0,
               title="Technology", title_fontsize=9)

    fig.subplots_adjust(bottom=0.20, top=0.78)

    # Save
    output_path = snakemake.output.pdf
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved DH balance price ventiles to {output_path}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_dh_balance_price_ventiles",
            configfiles=["dev/config.sysgf.test.yaml"],
            simpl="",
            clusters="27",
            opts="",
            sector_opts="none",
            planning_horizons="2045",
        )
    main(snakemake)
