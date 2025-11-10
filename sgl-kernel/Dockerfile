ARG BASE_IMG=pytorch/manylinux2_28-builder
ARG CUDA_VERSION=12.8

# Dependency stage: install system deps, CMake, ccache, Python deps (including torch)
FROM ${BASE_IMG}:cuda${CUDA_VERSION} AS deps

# Overridable build arguments
ARG ARCH=x86_64
ARG PYTHON_VERSION=3.10
# Manylinux python path tag, e.g. cp310-cp310 / cp311-cp311
ARG PYTHON_TAG=cp310-cp310
ARG CMAKE_VERSION_MAJOR=3.31
ARG CMAKE_VERSION_MINOR=1
ARG CCACHE_VERSION=4.12.1

ENV PYTHON_ROOT_PATH=/opt/python/${PYTHON_TAG}
ENV PATH=/opt/cmake/bin:${PATH}
ENV NINJA_STATUS="[%f/%t %es] "
ENV FLASHINFER_CUDA_ARCH_LIST="8.0 8.9 9.0a 10.0a 12.0a"
# CUDA headers path (same as script)
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}
ENV C_INCLUDE_PATH=/usr/local/cuda/include/cccl${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}

# System dependencies
RUN yum install gcc gcc-c++ make wget tar numactl-devel libibverbs -y --nogpgcheck \
 && ln -sv /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so \
 && yum clean all && rm -rf /var/cache/yum

# Install CMake (cached download)
RUN --mount=type=cache,id=sgl-kernel-cmake,target=/cmake-downloads \
    set -eux; \
    CMAKE_TARBALL=cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz; \
    # Check if CMake is already cached
    if [ -f /cmake-downloads/${CMAKE_TARBALL} ]; then \
      echo "Using cached CMake from /cmake-downloads/${CMAKE_TARBALL}"; \
      cp /cmake-downloads/${CMAKE_TARBALL} .; \
    else \
      echo "Downloading CMake from: https://cmake.org/files/v${CMAKE_VERSION_MAJOR}/${CMAKE_TARBALL}" \
      wget -q https://cmake.org/files/v${CMAKE_VERSION_MAJOR}/${CMAKE_TARBALL}; \
      # Cache the downloaded file
      cp ${CMAKE_TARBALL} /cmake-downloads/; \
    fi; \
    tar -xzf ${CMAKE_TARBALL}; \
    mv cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}-linux-${ARCH} /opt/cmake; \
    rm -f ${CMAKE_TARBALL}; \
    cmake --version

# Install ccache (same as script, 4.12.1)
RUN set -eux; \
    cd /tmp; \
    wget -q https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}.tar.xz; \
    tar -xf ccache-${CCACHE_VERSION}.tar.xz; \
    cd ccache-${CCACHE_VERSION}; \
    mkdir build && cd build; \
    /opt/cmake/bin/cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr .. >/dev/null; \
    make -j"$(nproc)" >/dev/null; \
    make install >/dev/null; \
    ccache --version; \
    rm -rf /tmp/ccache-${CCACHE_VERSION}*

RUN set -eux; \
    if [ "${ARCH}" = "aarch64" ]; then _LIB=sbsa; else _LIB="${ARCH}"; fi; \
    mkdir -p /usr/lib/${ARCH}-linux-gnu/; \
    ln -sf /usr/local/cuda-${CUDA_VERSION}/targets/${_LIB}-linux/lib/stubs/libcuda.so /usr/lib/${ARCH}-linux-gnu/libcuda.so

# Install Python dependencies (torch + build tools)
RUN --mount=type=cache,id=sgl-kernel-pip,target=/root/.cache/pip \
    set -eux; \
    case "${CUDA_VERSION}" in \
      13.0) TORCH_VER=2.9.0; CU_TAG=cu130 ;; \
      12.9) TORCH_VER=2.8.0; CU_TAG=cu129 ;; \
      12.8) TORCH_VER=2.8.0; CU_TAG=cu128 ;; \
      *)    TORCH_VER=2.8.0; CU_TAG=cu126 ;; \
    esac; \
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir torch==${TORCH_VER} --index-url https://download.pytorch.org/whl/${CU_TAG}; \
    ${PYTHON_ROOT_PATH}/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core

# Build stage: copy source and build wheel
FROM deps AS build
WORKDIR /sgl-kernel
# Only copy sgl-kernel source so code changes only affect later layers
COPY . /sgl-kernel/

# Optional: enable CMake/Ninja profiling (pass non-empty via --build-arg ENABLE_*)
ARG ENABLE_CMAKE_PROFILE
ARG ENABLE_BUILD_PROFILE
ARG ARCH=x86_64

# ccache config (BuildKit cache mounts)
ENV CCACHE_DIR=/ccache
ENV CCACHE_BASEDIR=/sgl-kernel
ENV CCACHE_MAXSIZE=10G
ENV CCACHE_COMPILERCHECK=content
ENV CCACHE_COMPRESS=true
ENV CCACHE_SLOPPINESS=file_macro,time_macros,include_file_mtime,include_file_ctime

# Set up ccache as compiler launcher (don't use PATH to avoid -ccbin conflicts)
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

RUN --mount=type=cache,id=sgl-kernel-ccache,target=/ccache \
    --mount=type=cache,id=sgl-kernel-pip,target=/root/.cache/pip \
    set -eux; \
    ccache -sV; \
    echo "Installing CMake"; \
    # Setting these flags to reduce OOM chance only on ARM
    if [ "${ARCH}" = "aarch64" ]; then \
      export CUDA_NVCC_FLAGS="-Xcudafe --threads=2"; \
      export MAKEFLAGS="-j2"; \
      export CMAKE_BUILD_PARALLEL_LEVEL=2; \
      export NINJAFLAGS="-j2"; \
      echo "ARM detected: Using extra conservative settings (2 parallel jobs)"; \
    else \
      export CMAKE_BUILD_PARALLEL_LEVEL=$(( $(nproc)/3 < 48 ? $(nproc)/3 : 48 )); \
    fi; \
    if [ -n "${ENABLE_CMAKE_PROFILE:-}" ]; then \
      echo "CMake profiling enabled - will save to /sgl-kernel/cmake-profile.json" \
      export CMAKE_ARGS="--profiling-output=/sgl-kernel/cmake-profile.json --profiling-format=google-trace"; \
    fi; \
    ${PYTHON_ROOT_PATH}/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation; \
    ./rename_wheels.sh; \
    if [ -n "${ENABLE_BUILD_PROFILE:-}" ] && [ -f /sgl-kernel/build/.ninja_log ]; then \
      echo "Ninja build profiling enabled - will save to /sgl-kernel/build-trace.json" \
      wget -q https://raw.githubusercontent.com/cradleapps/ninjatracing/084212eaf68f25c70579958a2ed67fb4ec2a9ca4/ninjatracing -O /tmp/ninjatracing; \
      if [ -f /tmp/ninjatracing ]; then ${PYTHON_ROOT_PATH}/bin/python /tmp/ninjatracing /sgl-kernel/build/.ninja_log > /sgl-kernel/build-trace.json; fi; \
    fi;

# Artifact stage (for --output to export wheel)
FROM scratch AS artifact
COPY --from=build /sgl-kernel/*.whl /wheels/
