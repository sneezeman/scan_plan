#!/bin/bash
# Deploy scan_planner container to CVMFS on the ESRF cluster.
#
# Only the .sif image and module file go to CVMFS.
# The source code lives on a shared beamline filesystem (git-managed),
# so updates are just `git pull` — no CVMFS transaction needed.
#
# Prerequisites:
#   1. SSH access to the CVMFS publisher (scisoft10/11) as cvmfs-hpc
#   2. The .sif image (built with: sudo apptainer build scan_planner.sif apptainer/scan_planner.def)
#   3. The module file from this repo (apptainer/module/scan_planner/2026.5.0)
#
# Run this FROM the CVMFS publisher machine (scisoft10/11).
#
# Usage (logged in as cvmfs-hpc):
#   bash deploy_cvmfs.sh /path/to/scan_planner.sif /path/to/scan_plan_repo

set -euo pipefail

REPO="hpc.esrf.fr"
VERSION="2026.5.0"
BASE="/cvmfs/${REPO}/software"
PKG_DIR="${BASE}/packages/linux/x86_64/scan_planner/${VERSION}"
MOD_DIR="${BASE}/modules/linux/x86_64/scan_planner"

SIF_FILE="${1:?Usage: $0 <scan_planner.sif> <scan_plan_repo_dir>}"
REPO_DIR="${2:?Usage: $0 <scan_planner.sif> <scan_plan_repo_dir>}"

if [ ! -f "${SIF_FILE}" ]; then
    echo "ERROR: .sif file not found: ${SIF_FILE}"
    exit 1
fi
if [ ! -f "${REPO_DIR}/apptainer/module/scan_planner/${VERSION}" ]; then
    echo "ERROR: module file not found in: ${REPO_DIR}/apptainer/module/scan_planner/${VERSION}"
    exit 1
fi

echo "=== Deploying scan_planner ${VERSION} to CVMFS ==="
echo "  Repository: ${REPO}"
echo "  .sif file:  ${SIF_FILE}"
echo "  Package dir: ${PKG_DIR}"
echo "  Module dir:  ${MOD_DIR}"
echo ""
echo "  NOTE: Only the container image and module file are deployed to CVMFS."
echo "  Source code should be cloned separately to the shared beamline path"
echo "  (see module file for the expected location)."
echo ""

# Start transaction
echo "Starting CVMFS transaction..."
cvmfs_server transaction "${REPO}/software/packages/linux/x86_64/scan_planner"

# Create directories
mkdir -p "${PKG_DIR}"
mkdir -p "${MOD_DIR}"

# Copy the container image
echo "Copying container image..."
cp "${SIF_FILE}" "${PKG_DIR}/scan_planner.sif"

# Copy the module file
echo "Copying module file..."
cp "${REPO_DIR}/apptainer/module/scan_planner/${VERSION}" "${MOD_DIR}/${VERSION}"

# Publish
echo "Publishing to CVMFS..."
cd "$HOME"
cvmfs_server publish "${REPO}"

echo ""
echo "=== CVMFS deployment complete ==="
echo ""
echo "Remaining manual step — clone the source code to the shared path:"
echo "  git clone git@gitlab.esrf.fr:artem1706/scan_plan.git /data/id16a/inhouse1/sware/Python/scan_plan"
echo ""
echo "Users can then run:"
echo "  module load scan_planner/${VERSION}"
echo "  scan_planner [config.json]"
echo ""
echo "To update the code later:"
echo "  cd /data/id16a/inhouse1/sware/Python/scan_plan && git pull"
