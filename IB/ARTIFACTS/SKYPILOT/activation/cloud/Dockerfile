FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04@sha256:0230b7f243483cb15969fa3cc724a9459599604427052fc2a0d4291c7c0647dd

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/activation/.venv \
    UV_PYTHON_PREFERENCE=only-managed \
    CUDA_HOME=/usr/local/cuda \
    HF_HOME=/root/.cache/huggingface

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        openssh-server \
        rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/activation/image-lock
COPY pyproject.toml uv.lock ./
RUN uv python install 3.14 \
    && uv sync --frozen --no-install-project \
    && uv cache clean

ENV UV_FROZEN=1 \
    PATH=/opt/activation/.venv/bin:/usr/local/bin:${PATH}

RUN mkdir -p \
    /root/activation_artifacts \
    /root/.cache/flashinfer \
    /root/.cache/huggingface \
    /root/sky_workdir

WORKDIR /root/sky_workdir

