#!/bin/bash
# Deploy scan_plan to CVMFS on the ESRF cluster.
#
# Prerequisites:
#   1. SSH access to the CVMFS publisher (scisoft10/11) as cvmfs-hpc
#   2. The .sif image (built on a machine with sudo/fakeroot)
#   3. The scan_plan git repo (cloned or copied)
#
# Run this FROM the CVMFS publisher machine (scisoft10/11).
#
# Usage (logged in as cvmfs-hpc):
#   bash deploy_cvmfs.sh /path/to/scan_plan.sif /path/to/scan_plan_repo

set -euo pipefail

REPO="hpc.esrf.fr"
VERSION="5.1.0"
BASE="/cvmfs/${REPO}/software"
PKG_DIR="${BASE}/packages/linux/x86_64/scan_plan/${VERSION}"
MOD_DIR="${BASE}/modules/linux/x86_64/scan_plan"

SIF_FILE="${1:?Usage: $0 <scan_plan.sif> <scan_plan_src_dir>}"
SRC_DIR="${2:?Usage: $0 <scan_plan.sif> <scan_plan_src_dir>}"

if [ ! -f "${SIF_FILE}" ]; then
    echo "ERROR: .sif file not found: ${SIF_FILE}"
    exit 1
fi
if [ ! -d "${SRC_DIR}/scan_plan" ]; then
    echo "ERROR: scan_plan package not found in: ${SRC_DIR}"
    exit 1
fi

echo "=== Deploying scan_plan ${VERSION} to CVMFS ==="
echo "  Repository: ${REPO}"
echo "  .sif file:  ${SIF_FILE}"
echo "  Source dir:  ${SRC_DIR}"
echo "  Package dir: ${PKG_DIR}"
echo "  Module dir:  ${MOD_DIR}"
echo ""

# Start transaction
echo "Starting CVMFS transaction..."
cvmfs_server transaction "${REPO}/software/packages/linux/x86_64/scan_plan"

# Create directories
mkdir -p "${PKG_DIR}/src"
mkdir -p "${MOD_DIR}"

# Copy the container image
echo "Copying container image..."
cp "${SIF_FILE}" "${PKG_DIR}/scan_plan.sif"

# Copy the source code (for bind-mounting at runtime)
echo "Copying source code..."
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
    --exclude='*.sif' \
    "${SRC_DIR}/" "${PKG_DIR}/src/"

# Copy the module file
echo "Copying module file..."
cp "${SRC_DIR}/apptainer/module/scan_plan/${VERSION}" "${MOD_DIR}/${VERSION}"

# Publish
echo "Publishing to CVMFS..."
cd "$HOME"
cvmfs_server publish "${REPO}"

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Users can now run:"
echo "  module load scan_plan/${VERSION}"
echo "  scan-plan [config.json]"
