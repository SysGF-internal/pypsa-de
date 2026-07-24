# SPDX-FileCopyrightText: 2026 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT
"""Shared helpers for the SysGF district-heating reporting scripts."""

import logging

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def apply_carrier_groups(
    frame: pd.DataFrame,
    carrier_groups: dict[str, list[str]] | None,
) -> pd.DataFrame:
    """Replace carrier columns with configured aggregate groups."""
    grouped = frame.copy()
    for group_name, carriers in (carrier_groups or {}).items():
        matching = [carrier for carrier in grouped.columns if carrier in carriers]
        if matching:
            grouped[group_name] = grouped[matching].sum(axis=1)
            grouped = grouped.drop(columns=matching)
    return grouped


def clean_label(label: object) -> object:
    """Convert a carrier name to a compact human-readable plot label."""
    if pd.isna(label) or label == "":
        return label

    text = str(label)
    replacements = {
        "urban central ": "",
        " CC": "",
        "river_water": "river water",
        "water pits": "PTES",
        "water tanks": "TTES",
        "ptes": "PTES",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.split())

    acronyms = {"AC", "DC", "EU", "PV", "PTES", "TTES"}
    words = text.split()
    if words:
        if words[0].upper() in acronyms:
            words[0] = words[0].upper()
        elif words[0].islower():
            words[0] = words[0].capitalize()
    return " ".join(words)


def get_colors(
    networks: dict[str, pypsa.Network],
    override_colors: dict[str, str] | None = None,
) -> dict[str, str]:
    """Extract carrier colors from the first network and apply overrides."""
    if not networks:
        raise ValueError("At least one network is required to extract carrier colors")

    first_network = next(iter(networks.values()))
    carrier_colors = first_network.carriers.color.dropna()
    carrier_colors = carrier_colors[carrier_colors != ""]
    colors = carrier_colors.to_dict()
    colors.update(override_colors or {})
    return colors


def process_networks(
    network_files: list[str],
    run_name: str,
    scenarios: list[str],
) -> dict[str, pypsa.Network]:
    """Load exactly one solved network for every requested scenario."""
    networks = {}
    for scenario in scenarios:
        path_fragment = f"{run_name}/{scenario}"
        matches = [path for path in network_files if path_fragment in str(path)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one network for scenario '{scenario}' below "
                f"run '{run_name}', found {len(matches)}: {matches}"
            )
        logger.info("Loading network for scenario '%s' from %s", scenario, matches[0])
        networks[scenario] = pypsa.Network(matches[0])
    return networks
