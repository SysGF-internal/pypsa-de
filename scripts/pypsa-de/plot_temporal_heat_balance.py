#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
Plot temporal district heating balance for a single scenario.

Produces a figure with two panels (winter week, summer week) showing the
stacked-area DH energy balance, overlaid with a load-weighted electricity
price line coloured by the central heating forward flow temperature.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sysgf_plot_helpers import (  # noqa: E402
    apply_carrier_groups,
    clean_label,
)

from scripts._helpers import configure_logging, mock_snakemake  # noqa: E402

logger = logging.getLogger(__name__)

# Default example-week date ranges (month-day).  Overridden via config
# sysgf.plotting.example_weeks.
DEFAULT_EXAMPLE_WEEKS = {
    "winter": {"start": "01-15", "end": "01-28"},
    "summer": {"start": "07-15", "end": "07-28"},
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


# apply_carrier_groups imported from sysgf_plot_helpers


def get_temporal_dh_balance(n, carrier_groups=None, subnodes_only=True):
    """
    Return temporal DH energy balance (timesteps x carriers) in MW.

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


def get_forward_temperature_t(ff_temp_path, snapshots):
    """Load forward flow temperature and average across DE buses."""
    da = xr.open_dataarray(ff_temp_path)
    df = da.to_pandas()
    # Filter to DE buses and compute spatial mean
    de_cols = [c for c in df.columns if "DE" in str(c)]
    if de_cols:
        df = df[de_cols]
    return df.mean(axis=1).reindex(snapshots)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _colored_line(ax, x, y, c, cmap, norm, lw=2.5):
    """Plot a line coloured by a third variable *c*."""
    points = np.column_stack([x, y])[:, np.newaxis, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw, zorder=10)
    lc.set_array(np.asarray(c)[:-1])
    ax.add_collection(lc)
    return lc


def plot_dh_balance_week(ax, balance, elec_price, ff_temp, colors, norm,
                         ylabel=True, price_label=True):
    """
    Plot one week of DH balance as stacked area + coloured price line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    balance : DataFrame  (timesteps x carriers, MW)
    elec_price : Series  (timesteps, €/MWh)
    ff_temp : Series  (timesteps, °C)
    colors : dict  carrier -> colour
    norm : mcolors.Normalize  shared color normalization for temperature
    ylabel : bool  show left y-label
    price_label : bool  show right y-label

    Returns
    -------
    lc : LineCollection (for colorbar)
    """
    df_gw = balance.div(1e3)  # MW -> GW

    # Sort: demand (negative) by magnitude, then supply (positive) by variance
    neg_cols = df_gw.columns[df_gw.sum() < 0]
    pos_cols = df_gw.columns[df_gw.sum() >= 0]
    neg_sorted = df_gw[neg_cols].sum().abs().sort_values(ascending=True).index if len(neg_cols) else neg_cols
    pos_sorted = df_gw[pos_cols].var().sort_values(ascending=True).index if len(pos_cols) else pos_cols
    col_order = list(neg_sorted) + list(pos_sorted)
    df_gw = df_gw[col_order]

    x = np.arange(len(df_gw))

    # Stacked areas
    pos = df_gw.clip(lower=0)
    neg = df_gw.clip(upper=0)

    # Plot positive stack
    bottom_pos = np.zeros(len(df_gw))
    for col in col_order:
        vals = pos[col].values
        if np.abs(vals).max() < 1e-6:
            continue
        c = colors.get(col, "#808080")
        ax.fill_between(x, bottom_pos, bottom_pos + vals, color=c,
                        linewidth=0.3, edgecolor="white", step="mid",
                        label=col, zorder=2)
        bottom_pos += vals

    # Plot negative stack
    bottom_neg = np.zeros(len(df_gw))
    for col in col_order:
        vals = neg[col].values
        if np.abs(vals).max() < 1e-6:
            continue
        c = colors.get(col, "#808080")
        ax.fill_between(x, bottom_neg, bottom_neg + vals, color=c,
                        linewidth=0.3, edgecolor="white", step="mid",
                        zorder=2)
        bottom_neg += vals

    ax.axhline(0, color="black", linewidth=0.5, zorder=3)

    # Electricity price line coloured by forward temperature
    ax2 = ax.twinx()
    cmap = plt.cm.coolwarm

    price_vals = elec_price.reindex(balance.index).values
    temp_vals = ff_temp.reindex(balance.index).values
    lc = _colored_line(ax2, x, price_vals, temp_vals, cmap, norm)

    ax2.set_ylim(0, max(1500, np.nanmax(price_vals) * 1.1))
    ax2.set_xlim(x[0], x[-1])
    if price_label:
        ax2.set_ylabel("Electricity Price [€/MWh]", fontsize=10)
    else:
        ax2.set_yticklabels([])

    # Main axis formatting
    ax.set_xlim(x[0], x[-1])
    if ylabel:
        ax.set_ylabel("District Heating Balance [GW]", fontsize=10)

    # x-tick labels as dates
    n_ticks = min(7, len(balance))
    tick_idx = np.linspace(0, len(balance) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [balance.index[i].strftime("%b %d") for i in tick_idx],
        fontsize=8,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=9)

    return lc


def build_legend_handles(carriers, colors):
    """Build legend handles for supply, demand and storage groups."""
    handles = []
    for carrier in carriers:
        c = colors.get(carrier, "#808080")
        handles.append(mpatches.Patch(facecolor=c, edgecolor="white",
                                      linewidth=0.5, label=carrier))
    # Electricity price entry (coloured by forward temperature)
    handles.append(Line2D([0], [0], color="grey", linewidth=2.5,
                          label="Electricity Price"))
    return handles


def _clean_label(label):
    """Shorten carrier label for display."""
    return clean_label(label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(snakemake):
    configure_logging(snakemake)

    plotting = snakemake.params.plotting
    carrier_groups = plotting.get("carrier_groups", {})
    subnodes_only = plotting.get("subnodes_only", True)
    override_colors = plotting.get("override_tech_colors", {})
    example_weeks = plotting.get("example_weeks", DEFAULT_EXAMPLE_WEEKS)

    # Load network
    n = pypsa.Network(snakemake.input.network)
    year = n.snapshots[0].year

    # Build date ranges from config
    winter_start = pd.Timestamp(f"{year}-{example_weeks['winter']['start']}")
    winter_end = pd.Timestamp(f"{year}-{example_weeks['winter']['end']}") + pd.Timedelta(hours=23)
    summer_start = pd.Timestamp(f"{year}-{example_weeks['summer']['start']}")
    summer_end = pd.Timestamp(f"{year}-{example_weeks['summer']['end']}") + pd.Timedelta(hours=23)

    # Compute data
    logger.info("Computing temporal DH balance")
    balance = get_temporal_dh_balance(n, carrier_groups, subnodes_only)
    elec_price = get_electricity_price_t(n)
    ff_temp = get_forward_temperature_t(snakemake.input.forward_temperature, n.snapshots)

    # Slice to example weeks
    bal_winter = balance.loc[winter_start:winter_end]
    bal_summer = balance.loc[summer_start:summer_end]
    price_winter = elec_price.loc[winter_start:winter_end]
    price_summer = elec_price.loc[summer_start:summer_end]
    temp_winter = ff_temp.loc[winter_start:winter_end]
    temp_summer = ff_temp.loc[summer_start:summer_end]

    if bal_winter.empty or bal_summer.empty:
        logger.warning("No data in one or both example weeks — check date ranges vs snapshots")

    # Colors from network carriers + overrides
    colors = n.carriers.color.dropna().to_dict()
    colors.update(override_colors)
    # Ensure carrier group names have a colour (inherit from first member)
    for group_name, carrier_list in carrier_groups.items():
        if group_name not in colors:
            for carrier in carrier_list:
                if carrier in colors:
                    colors[group_name] = colors[carrier]
                    break

    # Symmetric y-limits across both panels
    all_pos = pd.concat([bal_winter, bal_summer]).clip(lower=0).sum(axis=1).max() / 1e3
    all_neg = pd.concat([bal_winter, bal_summer]).clip(upper=0).sum(axis=1).min() / 1e3
    ylim_abs = max(abs(all_pos), abs(all_neg)) * 1.1
    ylim = (-ylim_abs, ylim_abs)

    # Create figure: 1 row x 2 columns
    fig, (ax_w, ax_s) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=False)

    # Shared temperature norm across both panels
    all_temp = pd.concat([temp_winter, temp_summer]).dropna()
    if len(all_temp) > 0:
        temp_min = float(all_temp.min()) - 1
        temp_max = float(all_temp.max()) + 1
    else:
        temp_min, temp_max = 90.0, 108.0
    if temp_min >= temp_max:
        temp_max = temp_min + 1.0
    norm = mcolors.Normalize(vmin=temp_min, vmax=temp_max)

    logger.info("Plotting winter week")
    plot_dh_balance_week(
        ax_w,
        bal_winter,
        price_winter,
        temp_winter,
        colors,
        norm,
        ylabel=True,
        price_label=False,
    )
    ax_w.set_ylim(*ylim)
    ax_w.set_title(f"{winter_start:%b %d}\u2013{winter_end:%b %d}", fontsize=12)

    logger.info("Plotting summer week")
    plot_dh_balance_week(
        ax_s,
        bal_summer,
        price_summer,
        temp_summer,
        colors,
        norm,
        ylabel=False,
        price_label=True,
    )
    ax_s.set_ylim(*ylim)
    ax_s.set_title(f"{summer_start:%b %d}\u2013{summer_end:%b %d}", fontsize=12)

    # Legend — collect unique carriers across both panels
    all_carriers = list(dict.fromkeys(
        list(bal_winter.columns) + list(bal_summer.columns)
    ))
    handles = build_legend_handles(all_carriers, colors)
    labels = [_clean_label(h.get_label()) for h in handles]

    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.08), ncol=min(6, len(handles)),
               frameon=False, fontsize=9, columnspacing=1.0)

    # Colorbar for forward temperature (reuse the shared norm)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])

    fig.subplots_adjust(bottom=0.18, top=0.82, wspace=0.25)
    cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Forward Flow Temperature [°C]", fontsize=10)

    # Save
    output_path = snakemake.output.pdf
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved temporal heat balance to {output_path}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_temporal_heat_balance",
            configfiles=["dev/config.sysgf.test.yaml"],
            simpl="",
            clusters="27",
            opts="",
            sector_opts="none",
            planning_horizons="2045",
        )
    main(snakemake)
