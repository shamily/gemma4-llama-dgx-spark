# syntax=docker/dockerfile:1.4
# llama.cpp + Gemma 4 — CUDA server image
# Target: ASUS Ascent GX10 / NVIDIA DGX Spark (GB10, SM_121, ARM64, CUDA 13)
# Builds from source to guarantee Gemma 4 support (PR #21309)
#
# Base image from NGC (not Docker Hub) — only NGC has ARM64 CUDA 13 images.
# GB10 compute capability is SM_121 (Grace Blackwell SoC, NOT SM_100).
# Fallback: if SM_121 build fails, rebuild with CUDA_DOCKER_ARCH=89 (Ada PTX).

ARG UBUNTU_VERSION=24.04
# CUDA 13.0 matches the DGX Spark system CUDA version
ARG CUDA_VERSION=13.0.1
# SM_121 = GB10 (Grace Blackwell integrated SoC)
ARG CUDA_DOCKER_ARCH=121

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG CUDA_DOCKER_ARCH
# Pin ref — bump this to pick up new llama.cpp fixes
ARG LLAMA_CPP_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-14 g++-14 build-essential cmake \
    python3 python3-pip git libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /build

RUN git clone --depth=1 --branch ${LLAMA_CPP_REF} \
    https://github.com/ggml-org/llama.cpp.git . \
    && echo "llama.cpp commit: $(git rev-parse HEAD)"

RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        ${CMAKE_ARGS} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j$(nproc)

# Collect shared libs
RUN mkdir -p /app/lib && find build -name "*.so*" -exec cp -P {} /app/lib \;

# Collect binaries + Python conversion/quantization tools
RUN mkdir -p /app/bin \
    && cp build/bin/* /app/bin \
    && cp *.py /app/ \
    && cp -r gguf-py /app/ \
    && cp -r requirements /app/ \
    && cp requirements.txt /app/

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl python3 python3-pip \
    && pip install --break-system-packages huggingface_hub[cli] \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/lib/ /usr/local/lib/
COPY --from=build /app/bin/ /app/
COPY --from=build /app/*.py /app/
COPY --from=build /app/gguf-py /app/gguf-py
COPY --from=build /app/requirements* /app/

RUN ldconfig

WORKDIR /app

# Server defaults (override via env or CLI flags)
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080
ENV LLAMA_ARG_N_GPU_LAYERS=-1
ENV LLAMA_ARG_CTX_SIZE=8192
ENV LLAMA_ARG_FLASH_ATTN=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -sf http://localhost:${LLAMA_ARG_PORT}/health || exit 1

# Default: run server. Override entrypoint for bench/convert.
ENTRYPOINT ["/app/llama-server"]
