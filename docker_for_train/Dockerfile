# syntax=docker/dockerfile:1.7

ARG CUDA_TAG=12.4.1-cudnn-devel-ubuntu22.04
ARG CUDA_RUNTIME_TAG=12.4.1-cudnn-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.10
ARG CUDA_IMAGE=docker.m.daocloud.io/nvidia/cuda

FROM ${CUDA_IMAGE}:${CUDA_TAG} AS builder

ARG PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    cmake \
    ffmpeg \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    ninja-build \
    libavutil-dev \
    libffi-dev \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libosmesa6 \
    libosmesa6-dev \
    libsm6 \
    libswresample-dev \
    libswscale-dev \
    libvulkan1 \
    libxkbcommon-x11-0 \
    libxext6 \
    libxrender1 \
    patchelf \
    pkg-config \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv --seed --python python${PYTHON_VERSION} ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build

COPY requirements ./requirements
COPY docker_for_train/wheelhouse/torch/ /wheelhouse/torch/

# Keep heavyweight CUDA wheels in their own stable layer. As long as these
# pinned versions do not change, ordinary source-code rebuilds reuse this layer.
# If docker_for_train/wheelhouse/torch already contains local wheels, pip will
# reuse them and only download missing dependencies.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /wheelhouse/torch \
    && PIP_NO_CACHE_DIR=0 python -m pip download \
    --dest /wheelhouse/torch \
    --find-links=/wheelhouse/torch \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    numpy==1.26.4

RUN uv pip install \
    --no-index \
    --find-links=/wheelhouse/torch \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    numpy==1.26.4
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-strategy unsafe-best-match -r requirements/uv-common-py310-lerobot.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps lerobot==0.4.0

COPY setup.py ./
COPY diffusion_policy ./diffusion_policy
RUN uv pip install .

FROM ${CUDA_IMAGE}:${CUDA_RUNTIME_TAG} AS runtime

ARG PYTHON_VERSION
ARG USER_ID=1000
ARG GROUP_ID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    MUJOCO_GL=osmesa \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    ffmpeg \
    git \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libgomp1 \
    libosmesa6 \
    libsm6 \
    libusb-1.0-0 \
    libvulkan1 \
    libxkbcommon-x11-0 \
    libxext6 \
    libxrender1 \
    patchelf \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    && groupadd --gid "${GROUP_ID}" trainer \
    && useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --create-home --shell /bin/bash trainer \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /opt/venv /opt/venv
COPY --chown=trainer:trainer . /workspace

ENV HOME=/home/trainer \
    WANDB_DIR=/workspace/wandb \
    MPLCONFIGDIR=/tmp/matplotlib

USER trainer

CMD ["/bin/bash"]
