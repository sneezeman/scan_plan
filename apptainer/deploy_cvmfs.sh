#!/bin/bash
# Deploy scan_plan to CVMFS on the ESRF cluster.
#
# Prerequisites:
#   1. You have SSH access to the CVMFS publisher (scisoft10/11)
#   2. The .sif image has been built (run build.sh first)
#   3. You have access to the target CVMFS repository
#
# This script is meant to be run FROM the CVMFS publisher machine,
# after scp'ing the .sif file there.
#
# Usage (on scisoft10, logged in as cvmfs-hpc):
#   ./deploy_cvmfs.sh /path/to/scan_plan.sif /path/to/scan_plan_repo
#
# Or step-by-step — see the commands below.

set -euo pipefail

REPO="hpc.esrf.fr"
VERSION="5.1.0"
BASE="/cvmfs/${REPO}/software"
PKG_DIR="${BASE}/packages/linux/x86_64/scan_plan/${VERSION}"
MOD_DIR="${BASE}/modules/linux/x86_64/scan_plan"

SIF_FILE="${1:?Usage: $0 <scan_plan.sif> <scan_plan_src_dir>}"
SRC_DIR="${2:?Usage: $0 <scan_plan.sif> <scan_plan_src_dir>}"

echo "=== Deploying scan_plan ${VERSION} to CVMFS ==="
echo "  Repository: ${REPO}"
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
