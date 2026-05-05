#!/bin/bash
# Wrapper script to launch scan_planner from the Apptainer container.
#
# The Python source (package "scan_plan") is bind-mounted into the
# container via PYTHONPATH, so `git pull` on the host updates the code
# without rebuilding the container.
#
# Usage:
#   bash run_scan_planner.sh                            # uses ./scan_plan_config.json
#   bash run_scan_planner.sh /path/to/config.json       # explicit config
#   bash run_scan_planner.sh config.json --debug        # with debug logging
#
# Environment variables (override defaults):
#   SCAN_PLANNER_SIF   — path to scan_planner.sif
#   SCAN_PLANNER_SRC   — path to scan_plan git repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults — adjust these or set env vars before running
SIF="${SCAN_PLANNER_SIF:-${SCRIPT_DIR}/scan_planner.sif}"
SRC="${SCAN_PLANNER_SRC:-${REPO_DIR}}"

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Container image not found: ${SIF}"
    echo "  Set SCAN_PLANNER_SIF or build with: bash build.sh"
    exit 1
fi

# Create Qt runtime directory
RUN_DIR="/tmp/run_user_$(id -u)"
mkdir -p "${RUN_DIR}"

# Bind mounts:
#   1. The scan_plan source code → /opt/scan_plan (via PYTHONPATH)
#   2. The user's home (for configs, data volumes, output files)
#   3. /tmp and Qt runtime directory
BINDS="${SRC}:/opt/scan_plan"
BINDS="${BINDS},${HOME}:${HOME}"
BINDS="${BINDS},/tmp:/tmp"
BINDS="${BINDS},${RUN_DIR}:/run/user/$(id -u)"
# Bind common ESRF data paths if they exist
[ -d /data ] && BINDS="${BINDS},/data:/data"
[ -d /visitors ] && BINDS="${BINDS},/visitors:/visitors"

# --writable-tmpfs: allows fixing /etc/machine-id at runtime
# (Apptainer may mount an empty host machine-id over the container's)
apptainer exec --writable-tmpfs \
    --bind "${BINDS}" \
    --env "PYTHONPATH=/opt/scan_plan" \
    "${SIF}" \
    bash -c 'cat /proc/sys/kernel/random/uuid | tr -d "-" > /etc/machine-id && mkdir -p /var/lib/dbus && cp /etc/machine-id /var/lib/dbus/machine-id && python -m scan_plan.cli "$@"' _ "$@"
