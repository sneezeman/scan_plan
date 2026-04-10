#!/bin/bash
# Build the scan_plan Apptainer image.
#
# Requires root or fakeroot privileges.
#
# Usage:
#   bash build.sh              # builds scan_plan.sif in current directory
#   bash build.sh /some/path   # builds scan_plan.sif at /some/path/scan_plan.sif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/scan_plan.def"
OUT_DIR="${1:-.}"
OUT_FILE="${OUT_DIR}/scan_plan.sif"

echo "Building scan_plan Apptainer image..."
echo "  Definition: ${DEF_FILE}"
echo "  Output:     ${OUT_FILE}"

if [ "$(id -u)" -eq 0 ]; then
    apptainer build "${OUT_FILE}" "${DEF_FILE}"
else
    echo "  (not root — using sudo)"
    sudo apptainer build "${OUT_FILE}" "${DEF_FILE}"
fi

echo ""
echo "Done. Test with:"
echo "  apptainer exec --bind /path/to/scan_plan:/opt/scan_plan --env PYTHONPATH=/opt/scan_plan ${OUT_FILE} python -c 'import scan_plan; print(scan_plan.__version__)'"
