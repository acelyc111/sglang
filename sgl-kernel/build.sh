#!/bin/bash
set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION> [ARCH]"
  exit 1
fi

PYTHON_VERSION="$1"          # e.g. 3.10
CUDA_VERSION="$2"            # e.g. 12.8
ARCH="${3:-$(uname -i)}"     # optional override

if [ "${ARCH}" = "aarch64" ]; then
  BASE_IMG="pytorch/manylinuxaarch64-builder"
else
  BASE_IMG="pytorch/manylinux2_28-builder"
fi

# Create cache directories for persistent build artifacts in home directory
# Using home directory to persist across workspace cleanups/checkouts
CACHE_DIR="${HOME}/.cache/sgl-kernel"
BUILDX_CACHE_DIR="${CACHE_DIR}/buildx"
mkdir -p "${BUILDX_CACHE_DIR}"

PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

# Output directory for wheels
DIST_DIR="dist"
mkdir -p "${DIST_DIR}"

echo "----------------------------------------"
echo "Build configuration"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "ARCH:           ${ARCH}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "PYTHON_TAG:     ${PY_TAG}"
echo "Output:         ${DIST_DIR}/"
echo "Buildx cache:   ${BUILDX_CACHE_DIR}"
echo "----------------------------------------"

# Optional profiling build-args (empty string disables)
BUILD_ARGS=()
[ -n "${ENABLE_CMAKE_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_CMAKE_PROFILE="${ENABLE_CMAKE_PROFILE}")
[ -n "${ENABLE_BUILD_PROFILE:-}" ] && BUILD_ARGS+=(--build-arg ENABLE_BUILD_PROFILE="${ENABLE_BUILD_PROFILE}")

docker buildx build \
  -v $(pwd):/sgl-kernel \
  -f sgl-kernel/Dockerfile sgl-kernel \
  --build-arg BASE_IMG="${BASE_IMG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ARCH="${ARCH}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg PYTHON_TAG="${PY_TAG}" \
  "${BUILD_ARGS[@]}" \
  --cache-from type=local,src=${BUILDX_CACHE_DIR} \
  --cache-to type=local,dest=${BUILDX_CACHE_DIR},mode=max \
  --target artifact \
  --output "type=local,dest=${DIST_DIR}"

echo "Done. Wheels are in ${DIST_DIR}/"
