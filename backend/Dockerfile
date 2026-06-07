# QIE Studio — CUDA backend image (DESIGN §8). Primary, fully-featured image: Python 3.13 +
# PyTorch CUDA wheels + diffusers stack (+ best-effort Nunchaku SVDQuant int4). Serves the
# built SPA as static assets.
#
# Build context is the REPO ROOT (see docker-compose `backend.build.context: .`).

# --- Stage 1: build the frontend SPA -------------------------------------------------------
FROM node:24-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build   # -> /fe/dist

# --- Stage 2: CUDA devel + Python 3.13 backend ---------------------------------------------
# `devel` (not `runtime`): optimum-quanto's fp8 path and Nunchaku JIT-compile CUDA kernels at
# load time, which needs nvcc + CUDA headers. The devel base ships them so fp8 works on this
# Ampere card (emulated, no native fp8 < SM 8.9, but functional) — DESIGN §9.1.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS backend

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models \
    QIE_FRONTEND_DIST=/app/frontend_dist \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.6 \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Python 3.13 (deadsnakes) + minimal build/runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev \
        libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv on Python 3.13.
RUN python3.13 -m venv /opt/venv && pip install --upgrade pip setuptools wheel

WORKDIR /app

# 1) Torch CUDA wheels first (cu126), pinned to the index so we don't pull a CPU build.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch torchvision

# 2) Backend core + inference extras (torch already satisfied above).
COPY backend/pyproject.toml /app/pyproject.toml
COPY backend/app/__init__.py /app/app/__init__.py
RUN pip install -e ".[inference]"

# 3) Nunchaku (SVDQuant int4/fp4, CUDA-only) — the workstation/gaming-card path (DESIGN §9.1).
#    Pinned prebuilt wheel matched to this Python (cp313) + torch (2.12) combo; the cu12.8
#    build is runtime-compatible with the torch cu126 runtime (CUDA 12.x minor compat,
#    validated on the A6000). NOTE: the PyPI `nunchaku` is an unrelated placeholder — do not
#    use it. Update NUNCHAKU_WHEEL when bumping torch/python. (DESIGN §9.4)
ARG NUNCHAKU_WHEEL=https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260306/nunchaku-1.3.0.dev20260306%2Bcu12.8torch2.12-cp313-cp313-linux_x86_64.whl
RUN pip install "${NUNCHAKU_WHEEL}"

# 4) Application source + built SPA.
COPY backend/ /app/
COPY --from=frontend /fe/dist /app/frontend_dist

EXPOSE 8000

# Migrations run on startup inside the app lifespan (DESIGN §8). Single worker: the
# in-process job queue serializes GPU work, so multiple uvicorn workers would break the
# single-accelerator guarantee.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
