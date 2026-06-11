#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2026 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
Evaluate the PTES feature-matrix test scenarios.

Reads the solved networks of all scenarios defined in
config/config.test_dev.yaml (PTES off / simple / layered x heat-pump /
resistive boosting x capacity scaling) and produces a multi-page PDF with
per-scenario visualisations plus a summary CSV with functionality checks:

- component structure matches the scenario (store/charger/discharger,
  per-layer stores, booster heat pumps vs resistive booster, dummy HP)
- layered model: total stored volume is conserved over time
- capacity scaling: max SOC respects e_max_pu (static scalar or
  time-varying profile)
- boosting: boost heat only in hours with discharge; for resistive
  scenarios the boost/discharge energy ratio is reported
- layered throughput: heat-equivalent charge+discharge power against the
  invested capacity and the energy-to-power ratio

The script never aborts on a failed check; failures are recorded in the
summary table and flagged in the figures so a single run shows the state of
all feature combinations.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from matplotlib.backends.backend_pdf import PdfPages

from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def scenario_from_path(path: str) -> str:
    """Extract the run/scenario name from results/<prefix>/<run>/networks/..."""
    parts = Path(path).parts
    return parts[parts.index("networks") - 1]


def weighted_sum(series_t: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Energy totals [MWh] from power time series [MW] and snapshot weightings."""
    if series_t.empty:
        return pd.Series(dtype=float)
    return series_t.mul(weights, axis=0).sum()


class PtesComponents:
    """Locate the PTES components of one solved network by name convention."""

    def __init__(self, n: pypsa.Network):
        self.n = n
        # astype(str) keeps the .str accessor usable on empty indexes
        # (e.g. a network without any stores or links)
        stores = n.stores.index.astype(str)
        links = n.links.index.astype(str)

        self.layer_stores = stores[stores.str.contains("water pits layer")]
        self.agg_stores = stores[
            stores.str.contains("water pits") & ~stores.str.contains("layer")
        ]
        self.is_layered = not self.layer_stores.empty

        self.chargers = links[
            links.str.contains("water pits charger")
            & ~links.str.contains("resistive")
        ]
        self.dischargers = links[links.str.contains("water pits discharger")]

        # Booster heat pumps: in the layered model the per-layer links carry
        # the operation and the plain 'ptes heat pump' is the capacity-carrying
        # dummy (p == 0); in the simple model the plain link is the booster.
        ptes_hps = links[links.str.contains("ptes") & links.str.contains("heat pump")]
        layer_hps = ptes_hps[ptes_hps.str.contains("ptes layer")]
        plain_hps = ptes_hps.difference(layer_hps)
        self.booster_hps = layer_hps if self.is_layered else plain_hps
        self.dummy_hps = plain_hps if self.is_layered else pd.Index([])

        self.resistive_boosters = links[
            links.str.contains("water pits resistive booster")
        ]
        self.standalone_routers = links[
            links.str.contains("resistive heater stand-alone")
        ]

    @property
    def has_ptes(self) -> bool:
        return not self.agg_stores.empty or not self.layer_stores.empty


def evaluate_scenario(name: str, n: pypsa.Network) -> tuple[dict, dict]:
    """Compute metrics/checks and the time series needed for the figures."""
    w = n.snapshot_weightings.generators
    c = PtesComponents(n)
    m = {"scenario": name, "objective": n.objective}
    ts = {}

    m["has_ptes"] = c.has_ptes
    m["is_layered"] = c.is_layered
    m["n_layer_stores"] = len(c.layer_stores)
    m["n_booster_hps"] = len(c.booster_hps)
    m["n_resistive_boosters"] = len(c.resistive_boosters)
    m["boosting"] = (
        "resistive"
        if len(c.resistive_boosters)
        else ("heat pump" if len(c.booster_hps) else "none")
    )

    if not c.has_ptes:
        return m, ts

    # --- capacity and energy throughput -------------------------------------
    agg = n.stores.loc[c.agg_stores]
    m["e_nom_opt_twh"] = agg.e_nom_opt.sum() / 1e6

    # charger draws DH heat on bus0; discharger delivers heat on bus1
    charge_heat = weighted_sum(n.links_t.p0[c.chargers], w)
    discharge_heat = -weighted_sum(n.links_t.p1[c.dischargers], w)
    m["charged_twh"] = charge_heat.sum() / 1e6
    m["discharged_twh"] = discharge_heat.sum() / 1e6

    ts["charge_mw"] = n.links_t.p0[c.chargers].sum(axis=1)
    ts["discharge_mw"] = -n.links_t.p1[c.dischargers].sum(axis=1)

    # --- boost heat ----------------------------------------------------------
    # Booster HPs are reversed links delivering heat on bus0 (p0 <= 0); the
    # resistive booster is a forward link delivering heat on bus1.
    boost_mw = pd.Series(0.0, index=n.snapshots)
    if len(c.booster_hps):
        boost_mw = boost_mw + (-n.links_t.p0[c.booster_hps].clip(upper=0)).sum(axis=1)
    if len(c.resistive_boosters):
        boost_mw = boost_mw + (-n.links_t.p1[c.resistive_boosters]).sum(axis=1)
    ts["boost_mw"] = boost_mw
    m["boost_twh"] = float((boost_mw * w).sum()) / 1e6
    m["boost_per_discharge"] = (
        m["boost_twh"] / m["discharged_twh"] if m["discharged_twh"] > 0 else np.nan
    )

    # boost must coincide with discharge (boosting serves discharged heat)
    discharging = ts["discharge_mw"] > 1e-3
    stray_boost = float((boost_mw * w)[~discharging].sum())
    m["check_boost_with_discharge"] = stray_boost <= max(
        1e-6, 0.01 * m["boost_twh"] * 1e6
    )

    # --- state of charge and capacity scaling --------------------------------
    if c.is_layered:
        layer_soc = n.stores_t.e[c.layer_stores]
        # group by layer index for the stacked plot (no $ anchor: myopic
        # store names carry a build-year suffix, e.g. '... layer 0-2045')
        layer_idx = c.layer_stores.str.extract(r"water pits layer (\d+)")[0].astype(int)
        ts["layer_soc_m3"] = layer_soc.T.groupby(layer_idx.values).sum().T
        total = layer_soc.sum(axis=1)
        ts["total_volume_m3"] = total
        # volume conservation: the volume-capacity constraint pins the total
        rel_dev = float((total.max() - total.min()) / total.mean()) if total.mean() else 0.0
        m["volume_conservation_rel_dev"] = rel_dev
        m["check_volume_conserved"] = rel_dev < 1e-4

        # throughput vs invested capacity (heat-equivalent, using the stamped
        # energy-to-power ratio and discharger heat efficiencies)
        e2p = n.links.loc[c.chargers, "energy to power ratio"].replace(0, np.nan).dropna()
        if not e2p.empty:
            r = float(e2p.iloc[0])
            heat_equiv = n.links_t.p0[c.chargers].sum(axis=1) + (
                -n.links_t.p1[c.dischargers]
            ).sum(axis=1)
            util = float((r * heat_equiv).max() / agg.e_nom_opt.sum()) if agg.e_nom_opt.sum() else 0.0
            m["throughput_utilisation"] = util
            m["check_throughput_bound"] = util <= 1.001
    else:
        soc = n.stores_t.e[c.agg_stores]
        ts["soc_mwh"] = soc.sum(axis=1)
        # e_max_pu: static column or time-varying profile
        emp_t = n.stores_t.e_max_pu.reindex(columns=c.agg_stores)
        emp_static = n.stores.loc[c.agg_stores, "e_max_pu"]
        emp = emp_t.fillna(emp_static) if not emp_t.empty else pd.DataFrame(
            emp_static.values[None, :].repeat(len(n.snapshots), axis=0),
            index=n.snapshots,
            columns=c.agg_stores,
        )
        bound = (emp * n.stores.loc[c.agg_stores, "e_nom_opt"]).sum(axis=1)
        ts["soc_bound_mwh"] = bound
        m["e_max_pu_min"] = float(emp.min().min())
        m["e_max_pu_max"] = float(emp.max().max())
        m["e_max_pu_time_varying"] = bool((emp.nunique() > 1).any())
        with np.errstate(invalid="ignore"):
            headroom = (soc.sum(axis=1) - bound).max()
        m["check_soc_within_bound"] = bool(headroom <= max(1.0, 1e-4 * bound.max()))

    return m, ts


def expected_structure(name: str) -> dict:
    """What the scenario name implies for the component structure."""
    return {
        "has_ptes": name != "PTESoff",
        "is_layered": name.startswith("layered"),
        "boosting": (
            "none"
            if name == "PTESoff"
            else ("resistive" if name.endswith("_rh") else "heat pump")
        ),
    }


def check_structure(m: dict) -> bool:
    exp = expected_structure(m["scenario"])
    return all(m.get(k) == v for k, v in exp.items())


def plot_overview(pdf: PdfPages, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("PTES feature matrix - cross-scenario overview")
    s = summary.set_index("scenario")

    s["e_nom_opt_twh"].plot.bar(ax=axes[0, 0], color="tab:blue")
    axes[0, 0].set_title("PTES energy capacity [TWh]")

    s[["charged_twh", "discharged_twh"]].plot.bar(ax=axes[0, 1])
    axes[0, 1].set_title("Charged / discharged heat [TWh]")

    s["boost_twh"].plot.bar(ax=axes[1, 0], color="tab:red")
    axes[1, 0].set_title("Boost heat [TWh]")

    rel_cost = (s["objective"] - s["objective"].iloc[0]) / 1e9
    rel_cost.plot.bar(ax=axes[1, 1], color="tab:green")
    axes[1, 1].set_title(f"System cost vs {s.index[0]} [bn EUR]")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_scenario(pdf: PdfPages, m: dict, ts: dict) -> None:
    name = m["scenario"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Scenario: {name}")

    # (a) state of charge
    ax = axes[0, 0]
    if "layer_soc_m3" in ts:
        ts["layer_soc_m3"].plot.area(ax=ax, linewidth=0, alpha=0.8)
        ts["total_volume_m3"].plot(ax=ax, color="black", linewidth=1, label="total")
        ax.set_title("Layer volumes [m3] (total must stay constant)")
        ax.legend(fontsize=7, ncol=3)
    elif "soc_mwh" in ts:
        ts["soc_mwh"].plot(ax=ax, label="SOC")
        ts["soc_bound_mwh"].plot(
            ax=ax, color="red", linestyle="--", label="e_max_pu * e_nom_opt"
        )
        ax.set_title("Store SOC vs capacity bound [MWh]")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no PTES components", ha="center", va="center")
        ax.set_title("State of charge")

    # (b) charge / discharge dispatch
    ax = axes[0, 1]
    if "charge_mw" in ts:
        ts["charge_mw"].plot(ax=ax, label="charge", alpha=0.8)
        ts["discharge_mw"].plot(ax=ax, label="discharge", alpha=0.8)
        ax.legend(fontsize=8)
    ax.set_title("Charge / discharge heat [MW]")

    # (c) boosting
    ax = axes[1, 0]
    if "boost_mw" in ts and ts["boost_mw"].abs().max() > 0:
        ts["boost_mw"].plot(ax=ax, color="tab:red", label="boost heat")
        ts["discharge_mw"].plot(ax=ax, color="tab:orange", alpha=0.5, label="discharge")
        ax.legend(fontsize=8)
        ratio = m.get("boost_per_discharge", np.nan)
        ax.set_title(f"Boosting [MW] ({m['boosting']}, boost/discharge = {ratio:.3f})")
    else:
        ax.text(0.5, 0.5, "no boosting activity", ha="center", va="center")
        ax.set_title(f"Boosting ({m['boosting']})")

    # (d) checks panel
    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"structure as expected: {check_structure(m)}"]
    for key, val in m.items():
        if key.startswith("check_"):
            lines.append(f"{key[6:]}: {'PASS' if val else 'FAIL'}")
        elif key in (
            "e_nom_opt_twh",
            "volume_conservation_rel_dev",
            "throughput_utilisation",
            "e_max_pu_min",
            "e_max_pu_max",
            "e_max_pu_time_varying",
            "n_layer_stores",
            "n_booster_hps",
            "n_resistive_boosters",
        ):
            val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
            lines.append(f"{key}: {val_str}")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax.set_title("Functionality checks")

    for ax in axes.flat[:3]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("evaluate_ptes_feature_matrix")
    configure_logging(snakemake)

    records = []
    with PdfPages(snakemake.output.pdf) as pdf:
        scenario_pages = []
        for path in snakemake.input.networks:
            name = scenario_from_path(path)
            logger.info(f"Evaluating scenario {name} ({path})")
            n = pypsa.Network(path)
            try:
                m, ts = evaluate_scenario(name, n)
            except Exception:
                logger.exception(f"Evaluation failed for scenario {name}")
                m, ts = {"scenario": name, "objective": np.nan, "has_ptes": None}, {}
            m["check_structure"] = check_structure(m)
            records.append(m)
            scenario_pages.append((m, ts))

        summary = pd.DataFrame(records)
        plot_overview(pdf, summary.fillna(0.0))
        for m, ts in scenario_pages:
            plot_scenario(pdf, m, ts)

    summary.to_csv(snakemake.output.summary, index=False)
    failed = [
        (r["scenario"], k)
        for r in records
        for k, v in r.items()
        if k.startswith("check_") and v is False
    ]
    if failed:
        logger.warning(f"Failed functionality checks: {failed}")
    else:
        logger.info("All functionality checks passed.")
