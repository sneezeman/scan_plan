#!/bin/bash
# Build the scan_plan Apptainer image.
#
# Run this on a machine where you have root/fakeroot privileges
# (e.g. your workstation, or a CI runner — NOT the CVMFS publisher).
#
# Usage:
#   ./build.sh              # builds scan_plan.sif in current directory
#   ./build.sh /some/path   # builds scan_plan.sif at /some/path/scan_plan.sif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/scan_plan.def"
OUT_DIR="${1:-.}"
OUT_FILE="${OUT_DIR}/scan_plan.sif"

echo "Building scan_plan Apptainer image..."
echo "  Definition: ${DEF_FILE}"
echo "  Output:     ${OUT_FILE}"

apptainer build --fakeroot "${OUT_FILE}" "${DEF_FILE}"

echo ""
echo "Done. Test with:"
echo "  apptainer exec --bind /path/to/scan_plan:/opt/scan_plan ${OUT_FILE} python3.13 -c 'import scan_plan; print(scan_plan.__version__)'"
