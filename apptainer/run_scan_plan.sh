#!/bin/bash
# Wrapper script to launch scan_plan from the Apptainer container.
#
# The scan_plan source code is bind-mounted into the container and
# installed in editable mode, so `git pull` on the host updates the
# code without rebuilding the container.
#
# Usage:
#   run_scan_plan.sh                           # uses ./scan_plan_config.json
#   run_scan_plan.sh /path/to/config.json      # explicit config
#   run_scan_plan.sh config.json --debug       # with debug logging
#
# Environment variables (override defaults):
#   SCAN_PLAN_SIF   — path to scan_plan.sif
#   SCAN_PLAN_SRC   — path to scan_plan git repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults — adjust these or set env vars before running
SIF="${SCAN_PLAN_SIF:-${SCRIPT_DIR}/scan_plan.sif}"
SRC="${SCAN_PLAN_SRC:-${REPO_DIR}}"

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container image not found: ${SIF}"
    echo "  Set SCAN_PLAN_SIF or build with: ./build.sh"
    exit 1
fi

# Bind mounts:
#   1. The scan_plan source code → /opt/scan_plan (pip install -e)
#   2. The user's home (for configs, data volumes, output files)
#   3. /tmp for Qt/VTK temp files
BINDS="${SRC}:/opt/scan_plan"
BINDS="${BINDS},${HOME}:${HOME}"
# Bind /tmp and common ESRF data paths if they exist
[ -d /tmp ] && BINDS="${BINDS},/tmp:/tmp"
[ -d /data ] && BINDS="${BINDS},/data:/data"
[ -d /visitors ] && BINDS="${BINDS},/visitors:/visitors"

apptainer exec \
    --bind "${BINDS}" \
    --env "PYTHONPATH=/opt/scan_plan" \
    "${SIF}" \
    python3.13 -m scan_plan.cli "$@"
