# syntax=docker/dockerfile:1.7
# CUDA 12.6 matches the PyTorch 2.7.1 wheels used by the Nero/LeRobot stack.
ARG CUDA_VERSION=12.6.3
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG APT_MIRROR=archive.ubuntu.com

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}

RUN set -eux; \
    for source_file in /etc/apt/sources.list /etc/apt/sources.list.d/*; do \
        if [ -f "${source_file}" ]; then \
            sed -i \
                -e "s|archive.ubuntu.com|${APT_MIRROR}|g" \
                -e "s|security.ubuntu.com|${APT_MIRROR}|g" \
                "${source_file}"; \
        fi; \
    done; \
    printf '%s\n' \
        'Acquire::Retries "5";' \
        'Acquire::http::Timeout "120";' \
        'Acquire::https::Timeout "120";' \
        > /etc/apt/apt.conf.d/80-retries

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libusb-1.0-0 \
    libxext6 \
    libxrender1 \
    pkg-config \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade "pip<24.1" "setuptools<80" wheel

WORKDIR /opt/build
COPY requirements/uv-common-py310-lerobot.txt requirements/uv-common-py310-lerobot.txt
COPY setup.py pyproject.toml README.md ./
COPY diffusion_policy ./diffusion_policy

# Install the CUDA-enabled framework before the rest of the pinned runtime.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1 torchvision==0.22.1

RUN python -m pip install --no-cache-dir \
    -r requirements/uv-common-py310-lerobot.txt \
    && python -m pip install --no-cache-dir --no-deps lerobot==0.4.0 \
    && python -m pip install --no-cache-dir -e ".[training,test]"

WORKDIR /workspace
COPY . /workspace

ENV PYTHONPATH=/workspace \
    WANDB_DIR=/workspace/data/wandb \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# The compose service keeps the container open. Run torchrun through
# `docker compose exec trainer bash` after the image is loaded on the server.
CMD ["sleep", "infinity"]
