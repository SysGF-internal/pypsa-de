# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Quantify the system impact of thermal energy storage in the all-features test.

Compares the TESoff and TESon scenarios of the SysGF all-features integration
test (see ``config/config.sysgf_allfeatures.yaml``) across all solved planning
horizons and writes a metrics table, a LaTeX table and figures.

Usage
-----
    python scripts/pypsa-de/analyse_tes_impact.py \
        --results-dir results/20260803_allfeatures_365H \
        --reference TESoff --comparison TESon

Outputs
-------
``<results-dir>/analysis/``
    ``tes_impact_metrics.csv``   one row per (scenario, horizon) with all metrics
    ``tes_impact_deltas.csv``    comparison minus reference, per horizon
    ``tes_impact_table.tex``     booktabs table for report.tex
``images/``
    ``tes_impact_<results-dir-name>.pdf`` / ``.png``
    ``tes_dh_mix_<results-dir-name>.pdf`` / ``.png``
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402

logger = logging.getLogger(__name__)

# Store carriers per TES technology. Layered PTES splits into per-layer stores,
# which are excluded here -- the aggregate "water pits" store carries e_nom.
TES_STORE_CARRIERS = {
    "TTES": "urban central water tanks",
    "PTES": "urban central water pits",
    "ATES": "urban central aquifer thermal energy storage",
}

HEAT_PUMP_PATTERN = re.compile(r"urban central (.+) heat pump$")


def _is_de(index: pd.Index) -> pd.Series:
    """Boolean mask selecting German components by bus/name prefix."""
    return pd.Series(index.str.startswith("DE"), index=index)


def _country_buses(n: pypsa.Network, country: str = "DE") -> pd.Index:
    """Buses located in `country`."""
    return n.buses.index[n.buses.index.str.startswith(country)]


def _urban_central_heat_buses(n: pypsa.Network, country: str = "DE") -> pd.Index:
    buses = n.buses.index[n.buses.carrier == "urban central heat"]
    return buses[buses.str.startswith(country)]


def total_system_cost(n: pypsa.Network) -> float:
    """Annualised total system cost [EUR/a] = capex of optimised capacity + opex."""
    capex = 0.0
    for comp, nom_opt in (
        ("generators", "p_nom_opt"),
        ("links", "p_nom_opt"),
        ("stores", "e_nom_opt"),
        ("storage_units", "p_nom_opt"),
        ("lines", "s_nom_opt"),
    ):
        df = getattr(n, comp)
        if df.empty or "capital_cost" not in df:
            continue
        capex += float((df[nom_opt] * df["capital_cost"]).sum())

    opex = 0.0
    for comp, flow_attr in (
        ("generators", "p"),
        ("links", "p0"),
        ("stores", "p"),
        ("storage_units", "p"),
    ):
        df = getattr(n, comp)
        if df.empty or "marginal_cost" not in df:
            continue
        flows = getattr(n, f"{comp}_t").get(flow_attr)
        if flows is None or flows.empty:
            continue
        cost = df["marginal_cost"].reindex(flows.columns).fillna(0.0)
        weighted = flows.abs().multiply(n.snapshot_weightings.objective, axis=0)
        opex += float((weighted * cost).sum().sum())

    return capex + opex


def tes_capacities(n: pypsa.Network, country: str = "DE") -> dict[str, float]:
    """Optimised TES energy capacity [MWh] per technology, `country` only."""
    out = {}
    for label, carrier in TES_STORE_CARRIERS.items():
        stores = n.stores[
            (n.stores.carrier == carrier) & n.stores.index.str.startswith(country)
        ]
        # Layered PTES: exclude per-layer volume stores, keep the aggregate.
        stores = stores[~stores.index.str.contains("layer")]
        out[f"tes_capacity_{label}_MWh"] = float(stores.e_nom_opt.sum())
    return out


def tes_throughput(n: pypsa.Network, country: str = "DE") -> dict[str, float]:
    """Heat discharged from each TES technology [MWh/a], `country` only."""
    out = {}
    for label, carrier in TES_STORE_CARRIERS.items():
        discharger = f"{carrier} discharger"
        links = n.links.index[
            (n.links.carrier == discharger) & n.links.index.str.startswith(country)
        ]
        links = links[~links.str.contains("layer")]
        if len(links) == 0 or "p1" not in n.links_t:
            out[f"tes_discharge_{label}_MWh"] = 0.0
            continue
        flows = n.links_t.p1.reindex(columns=links).fillna(0.0)
        weighted = flows.multiply(n.snapshot_weightings.objective, axis=0)
        # p1 is negative when heat leaves the link into the DH bus.
        out[f"tes_discharge_{label}_MWh"] = float(-weighted.clip(upper=0).sum().sum())
    return out


def dh_supply_mix(n: pypsa.Network, country: str = "DE") -> pd.Series:
    """Annual heat delivered onto German urban-central heat buses [MWh/a] by carrier."""
    heat_buses = set(_urban_central_heat_buses(n, country))
    if not heat_buses:
        return pd.Series(dtype=float)

    contributions: dict[str, float] = {}
    weights = n.snapshot_weightings.objective

    for port in ("0", "1", "2", "3", "4"):
        bus_col = f"bus{port}"
        if bus_col not in n.links:
            continue
        links = n.links.index[n.links[bus_col].isin(heat_buses)]
        if len(links) == 0:
            continue
        flow_attr = f"p{port}"
        if flow_attr not in n.links_t:
            continue
        flows = n.links_t[flow_attr].reindex(columns=links).dropna(axis=1, how="all")
        if flows.empty:
            continue
        weighted = flows.multiply(weights, axis=0)
        # Negative flow at a port means energy leaving the link into that bus.
        delivered = -weighted.clip(upper=0).sum()
        for name, value in delivered.items():
            carrier = n.links.at[name, "carrier"]
            contributions[carrier] = contributions.get(carrier, 0.0) + float(value)

    series = pd.Series(contributions).sort_values(ascending=False)
    return series[series > 0]


def heat_pump_mix(n: pypsa.Network, country: str = "DE") -> pd.Series:
    """Annual heat delivered by urban-central heat pumps [MWh/a] per source."""
    mix = dh_supply_mix(n, country)
    out = {}
    for carrier, value in mix.items():
        match = HEAT_PUMP_PATTERN.match(str(carrier))
        if match:
            out[match.group(1)] = value
    return pd.Series(out).sort_values(ascending=False)


def electrified_dh_share(n: pypsa.Network, country: str = "DE") -> float:
    """Share of German urban-central heat supplied by heat pumps and resistive heaters."""
    mix = dh_supply_mix(n, country)
    if mix.empty or mix.sum() <= 0:
        return float("nan")
    electric = mix[
        mix.index.to_series()
        .astype(str)
        .str.contains("heat pump|resistive heater", regex=True)
    ]
    return float(electric.sum() / mix.sum())


def dh_consumer_cost(n: pypsa.Network, country: str = "DE") -> float:
    """Volume-weighted marginal price of German urban-central heat [EUR/MWh]."""
    heat_buses = _urban_central_heat_buses(n, country)
    if len(heat_buses) == 0 or "urban central heat" not in set(n.loads.carrier):
        return float("nan")
    loads = n.loads.index[
        (n.loads.carrier == "urban central heat") & n.loads.bus.isin(heat_buses)
    ]
    if len(loads) == 0:
        return float("nan")
    load_p = n.loads_t.p_set.reindex(columns=loads).fillna(0.0)
    prices = n.buses_t.marginal_price.reindex(columns=n.loads.loc[loads, "bus"])
    prices.columns = loads
    weights = n.snapshot_weightings.objective
    energy = load_p.multiply(weights, axis=0)
    total_energy = float(energy.sum().sum())
    if total_energy <= 0:
        return float("nan")
    return float((energy * prices).sum().sum() / total_energy)


def curtailment(n: pypsa.Network, country: str = "DE") -> float:
    """Curtailed variable renewable energy [MWh/a]."""
    vre = ["solar", "solar-hsat", "onwind", "offwind-ac", "offwind-dc"]
    gens = n.generators.index[
        n.generators.carrier.isin(vre) & n.generators.index.str.startswith(country)
    ]
    if len(gens) == 0 or "p_max_pu" not in n.generators_t:
        return 0.0
    p_max_pu = n.generators_t.p_max_pu.reindex(columns=gens).fillna(1.0)
    available = p_max_pu.multiply(n.generators.loc[gens, "p_nom_opt"], axis=1)
    dispatched = n.generators_t.p.reindex(columns=gens).fillna(0.0)
    weights = n.snapshot_weightings.objective
    return float(
        ((available - dispatched).clip(lower=0).multiply(weights, axis=0)).sum().sum()
    )


def extract_metrics(path: Path, scenario: str, horizon: str) -> dict:
    logger.info("Reading %s", path)
    n = pypsa.Network(str(path))

    row = {
        "scenario": scenario,
        "horizon": horizon,
        "objective_EUR": float(getattr(n, "objective", float("nan"))),
        "total_system_cost_EUR": total_system_cost(n),
        "dh_consumer_cost_EUR_per_MWh": dh_consumer_cost(n),
        "electrified_dh_share": electrified_dh_share(n),
        "curtailment_DE_MWh": curtailment(n),
    }
    row.update(tes_capacities(n))
    row.update(tes_throughput(n))

    mix = dh_supply_mix(n)
    row["dh_supply_total_MWh"] = float(mix.sum())
    for carrier, value in mix.items():
        row[f"dh_supply::{carrier}"] = float(value)

    for source, value in heat_pump_mix(n).items():
        row[f"hp_heat::{source}"] = float(value)

    return row


def discover_networks(results_dir: Path) -> list[tuple[Path, str, str]]:
    """Find solved networks as (path, scenario, horizon)."""
    found = []
    for path in sorted(results_dir.glob("*/networks/*.nc")):
        scenario = path.parents[1].name
        match = re.search(r"_(\d{4})\.nc$", path.name)
        if not match:
            logger.warning("Skipping %s: no planning horizon in filename", path)
            continue
        found.append((path, scenario, match.group(1)))
    return found


def write_latex_table(deltas: pd.DataFrame, metrics: pd.DataFrame, out: Path) -> None:
    rows = []
    for horizon in sorted(metrics["horizon"].unique()):
        ref = metrics[(metrics.horizon == horizon)].set_index("scenario")
        if deltas.empty or horizon not in deltas.index:
            continue
        d = deltas.loc[horizon]
        for label in TES_STORE_CARRIERS:
            cap = ref.get(f"tes_capacity_{label}_MWh")
            cap_value = cap.max() / 1e6 if cap is not None else float("nan")
            rows.append((horizon, label, cap_value))
        rows.append((horizon, "cost", d.get("total_system_cost_EUR", float("nan"))))

    lines = [
        "% generated by scripts/pypsa-de/analyse_tes_impact.py",
        "{",
        "\\setlength{\\tabcolsep}{8pt}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Horizon & TTES [TWh] & PTES [TWh] & ATES [TWh] & $\\Delta$ cost [\\%] \\\\",
        "\\midrule",
    ]
    for horizon in sorted(metrics["horizon"].unique()):
        on = metrics[(metrics.horizon == horizon) & (metrics.scenario != "TESoff")]
        if on.empty:
            continue
        on = on.iloc[0]
        rel = (
            deltas.loc[horizon, "total_system_cost_EUR_rel_pct"]
            if not deltas.empty and horizon in deltas.index
            else float("nan")
        )
        lines.append(
            f"{horizon} & {on.get('tes_capacity_TTES_MWh', 0) / 1e6:.2f} & "
            f"{on.get('tes_capacity_PTES_MWh', 0) / 1e6:.2f} & "
            f"{on.get('tes_capacity_ATES_MWh', 0) / 1e6:.2f} & {rel:+.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "}"]
    out.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", out)


def plot_summary(metrics: pd.DataFrame, deltas: pd.DataFrame, stem: str) -> None:
    images = Path("images")
    images.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # TES capacity by technology and horizon (TESon only).
    on = metrics[metrics.scenario != "TESoff"]
    cap_cols = [f"tes_capacity_{label}_MWh" for label in TES_STORE_CARRIERS]
    if not on.empty:
        cap = on.set_index("horizon")[cap_cols] / 1e6
        cap.columns = list(TES_STORE_CARRIERS)
        cap.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_ylabel("TES energy capacity [TWh]")
    axes[0].set_title("DE thermal storage (TESon)")

    # Relative system cost change.
    if not deltas.empty and "total_system_cost_EUR_rel_pct" in deltas:
        deltas["total_system_cost_EUR_rel_pct"].plot(kind="bar", ax=axes[1], rot=0)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("Total system cost change [%]")
    axes[1].set_title("TESon vs TESoff")

    # Electrified DH share.
    share = metrics.pivot(
        index="horizon", columns="scenario", values="electrified_dh_share"
    )
    share.plot(kind="bar", ax=axes[2], rot=0)
    axes[2].set_ylabel("Electrified DH supply share [-]")
    axes[2].set_title("DE district heating")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(images / f"tes_impact_{stem}.{ext}", dpi=150)
    plt.close(fig)
    logger.info("Wrote images/tes_impact_%s.{pdf,png}", stem)


def plot_dh_mix(metrics: pd.DataFrame, stem: str) -> None:
    images = Path("images")
    images.mkdir(exist_ok=True)

    mix_cols = [c for c in metrics.columns if c.startswith("dh_supply::")]
    if not mix_cols:
        return
    mix = metrics.set_index(["scenario", "horizon"])[mix_cols] / 1e6
    mix.columns = [c.split("::", 1)[1] for c in mix_cols]
    mix = mix.loc[:, mix.max() > 0.01]  # drop negligible carriers

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(mix)), 5))
    mix.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("DE urban central heat supply [TWh/a]")
    ax.set_xlabel("")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(images / f"tes_dh_mix_{stem}.{ext}", dpi=150)
    plt.close(fig)
    logger.info("Wrote images/tes_dh_mix_%s.{pdf,png}", stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--reference", default="TESoff")
    parser.add_argument("--comparison", default="TESon")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    networks = discover_networks(args.results_dir)
    if not networks:
        raise SystemExit(f"No solved networks found under {args.results_dir}")

    metrics = pd.DataFrame(
        [extract_metrics(path, scen, hor) for path, scen, hor in networks]
    ).fillna(0.0)

    analysis = args.results_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(analysis / "tes_impact_metrics.csv", index=False)
    logger.info("Wrote %s", analysis / "tes_impact_metrics.csv")

    # Deltas: comparison minus reference, per horizon.
    delta_rows = {}
    numeric = metrics.select_dtypes("number").columns
    for horizon in sorted(metrics["horizon"].unique()):
        sub = metrics[metrics.horizon == horizon].set_index("scenario")
        if args.reference not in sub.index or args.comparison not in sub.index:
            logger.warning(
                "Horizon %s: need both %s and %s, have %s",
                horizon,
                args.reference,
                args.comparison,
                list(sub.index),
            )
            continue
        ref = sub.loc[args.reference, numeric]
        comp = sub.loc[args.comparison, numeric]
        row = (comp - ref).to_dict()
        for col in ("total_system_cost_EUR", "dh_consumer_cost_EUR_per_MWh"):
            if col in ref and ref[col]:
                row[f"{col}_rel_pct"] = 100.0 * (comp[col] - ref[col]) / abs(ref[col])
        delta_rows[horizon] = row

    deltas = pd.DataFrame(delta_rows).T
    if not deltas.empty:
        deltas.index.name = "horizon"
        deltas.to_csv(analysis / "tes_impact_deltas.csv")
        logger.info("Wrote %s", analysis / "tes_impact_deltas.csv")

    write_latex_table(deltas, metrics, analysis / "tes_impact_table.tex")
    stem = args.results_dir.name
    plot_summary(metrics, deltas, stem)
    plot_dh_mix(metrics, stem)

    # Console summary.
    print("\n=== TES impact summary ===")
    cols = [
        "scenario",
        "horizon",
        "total_system_cost_EUR",
        "tes_capacity_TTES_MWh",
        "tes_capacity_PTES_MWh",
        "tes_capacity_ATES_MWh",
        "electrified_dh_share",
    ]
    print(metrics[[c for c in cols if c in metrics]].to_string(index=False))
    if not deltas.empty:
        print("\n=== TESon - TESoff ===")
        show = [
            c
            for c in (
                "total_system_cost_EUR",
                "total_system_cost_EUR_rel_pct",
                "dh_consumer_cost_EUR_per_MWh",
                "dh_consumer_cost_EUR_per_MWh_rel_pct",
                "curtailment_DE_MWh",
            )
            if c in deltas
        ]
        print(deltas[show].to_string())


if __name__ == "__main__":
    main()
