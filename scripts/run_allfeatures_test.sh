#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
#
# Launch the SysGF all-features integration test (TESoff vs TESon, myopic
# 2030 -> 2045) on the TU Berlin HPC cluster.
#
#   scripts/run_allfeatures_test.sh [configfile]
#
# Defaults to the 365H tier-1 config. Pass config/config.sysgf_allfeatures_24H.yaml
# for the tier-2 run the TES impact numbers are read from.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-config/config.sysgf_allfeatures.yaml}"

# Copernicus Marine credentials for retrieve_seawater_temperature live in the
# interactive shell profile; the retrieve rule reads them from the environment.
if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.bashrc"
fi

# TU Berlin's .bashrc returns before its credential exports in non-interactive
# shells. Re-enter once through an interactive shell so those exports are
# inherited by this script. The marker prevents recursion when no credentials
# are configured at all.
if { [ -z "${COPERNICUSMARINE_USERNAME:-}" ] || [ -z "${COPERNICUSMARINE_PASSWORD:-}" ]; } &&
  [ "${SYSGF_ALLFEATURES_INTERACTIVE_SHELL:-0}" != 1 ]; then
  export SYSGF_ALLFEATURES_INTERACTIVE_SHELL=1
  exec bash -ic 'unset SNAKEMAKE_PROFILE; exec "$@"' bash "$0" "$CONFIG"
fi

# Do not inherit a user-level Snakemake profile. The solve phase uses the
# tracked profiles/slurm-tub profile explicitly, while build_scenarios is local.
unset SNAKEMAKE_PROFILE
export PATH="$HOME/scratch/pixi/bin:$PATH"

if [ -z "${COPERNICUSMARINE_USERNAME:-}" ] || [ -z "${COPERNICUSMARINE_PASSWORD:-}" ]; then
  echo "WARNING: COPERNICUSMARINE_USERNAME/PASSWORD not set." >&2
  echo "         retrieve_seawater_temperature will fail for the europe cutout." >&2
fi

mkdir -p logs/slurm

echo "=== [1/2] build_scenarios (DE production entry point) ==="
pixi run snakemake build_scenarios --cores 1 --configfile "$CONFIG" --force

echo "=== [2/2] solve_sector_networks via SLURM ==="
exec pixi run snakemake solve_sector_networks \
  --configfile "$CONFIG" \
  --profile profiles/slurm-tub \
  --keep-going \
  --rerun-incomplete
