# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Calculate per-region Aquifer Thermal Energy Storage (ATES) parameters.

Gridded ATES data (square ~1 km cells with a per-cell CAPEX and annual loss
factor) are aggregated onto the model's onshore regions (district-heating
subnodes when enabled). Each cell is assigned to the region that contains it and
aggregated with an inverse-CAPEX-weighted mean, so cheaper cells pull the
regional average more strongly. Regions without any cell are marked infeasible
(``e_nom_max = 0`` downstream).

The per-cell CAPEX is given as an overnight (non-annualised) investment, so it is
annualised here using the configured ATES lifetime and discount rate.

The script outputs, per region:

- ``capital_cost``: annualised CAPEX [EUR/MW/a], applied fully to the ATES
  charger in ``prepare_sector_network``.
- ``hourly_standing_losses``: hourly self-discharge of the ATES store
  (``1 - hourly_loss_factor``).
- ``feasible``: whether ATES can be built in the region.

Relevant Settings
-----------------

.. code:: yaml
    costs:
        fill_values:
            discount rate:
    sector:
        district_heating:
            ates:
                lifetime:

Inputs
------
- ``ates_grid``: gridded ATES data (GeoPackage) with ``CAPEX_EUR_MW`` and
  ``VERLUSTRATE`` (annual loss factor) columns.
- ``regions_onshore``: onshore (sub)region shapes.

Outputs
-------
- ``ates_potentials``: CSV indexed by region ``name`` with columns
  ``capital_cost``, ``hourly_standing_losses`` and ``feasible``.
"""

import logging

import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.add_electricity import calculate_annuity

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 8760


def aggregate_ates_to_regions(
    ates: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    capex_col: str = "CAPEX_EUR_MW",
    loss_col: str = "VERLUSTRATE",
) -> pd.DataFrame:
    """
    Inverse-CAPEX-weighted mean of CAPEX and loss rate per onshore region.

    Each ATES grid cell is assigned to the region that contains it
    (point-in-polygon on the cell's representative point). Cells with lower CAPEX
    receive a higher weight (``weight = 1 / capex``). Regions without any cell
    become NaN rows.

    Parameters
    ----------
    ates : geopandas.GeoDataFrame
        Gridded ATES data with ``capex_col`` and ``loss_col`` columns.
    regions : geopandas.GeoDataFrame
        Onshore (sub)region shapes with a ``name`` column.
    capex_col, loss_col : str
        Column names of the per-cell CAPEX and annual loss factor.

    Returns
    -------
    pandas.DataFrame
        Indexed by region ``name`` with columns ``capex_col``, ``loss_col`` and
        ``n_cells``; NaN for regions without any cell.
    """
    cells = ates.to_crs(regions.crs).copy()
    cells["geometry"] = cells.geometry.representative_point()
    joined = gpd.sjoin(
        cells,
        regions[["name", "geometry"]],
        how="inner",
        predicate="within",
    )

    # Inverse-CAPEX weighting: lower capex -> higher weight.
    joined["weight"] = 1.0 / joined[capex_col]
    df = pd.DataFrame(joined.drop(columns="geometry"))

    def _weighted_average(g):
        w = g["weight"]
        return pd.Series(
            {
                capex_col: (g[capex_col] * w).sum() / w.sum(),
                loss_col: (g[loss_col] * w).sum() / w.sum(),
                "n_cells": len(g),
            }
        )

    agg = df.groupby("name", sort=False).apply(_weighted_average, include_groups=False)
    return agg.reindex(regions["name"])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_ates_potentials", clusters="10", planning_horizons=2040
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    regions = gpd.read_file(snakemake.input.regions_onshore)

    logger.info("Loading gridded ATES data")
    ates_grid = gpd.read_file(snakemake.input.ates_grid)[
        ["CAPEX_EUR_MW", "VERLUSTRATE", "geometry"]
    ]

    logger.info("Aggregating ATES data to regions")
    ates_by_region = aggregate_ates_to_regions(ates_grid, regions)

    feasible = ates_by_region["n_cells"].notna() & (ates_by_region["n_cells"] > 0)
    logger.info(f"ATES feasible in {int(feasible.sum())} of {len(feasible)} regions")

    # Annual loss factor -> hourly loss factor -> hourly standing loss.
    hourly_losses = ates_by_region["VERLUSTRATE"] ** (1 / HOURS_PER_YEAR)
    hourly_standing_losses = 1 - hourly_losses

    # Annualise the overnight CAPEX.
    annuity = calculate_annuity(
        snakemake.params.lifetime, snakemake.params.discount_rate
    )
    capital_cost = ates_by_region["CAPEX_EUR_MW"] * annuity

    out = pd.DataFrame(
        {
            # Infeasible regions get e_nom_max=0 downstream; fill finite defaults.
            "capital_cost": capital_cost.where(feasible, 0.0),
            "hourly_standing_losses": hourly_standing_losses.where(feasible, 0.0),
            "feasible": feasible,
        }
    )

    logger.info(f"Writing results to {snakemake.output.ates_potentials}")
    out.to_csv(snakemake.output.ates_potentials, index_label="name")
