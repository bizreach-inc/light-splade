ARG BASE_IMAGE_URL=nvidia/cuda
ARG BASE_IMAGE_TAG=12.5.0-devel-ubuntu22.04
ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.8.16

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_base

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG} AS base

ARG PYTHON_VERSION
ARG UV_VERSION

COPY --from=uv_base /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1

# Update packages
RUN export DEBIAN_FRONTEND=noninteractive && \
    export TZ="Asia/Tokyo" && \
    apt-get update && \
    apt upgrade -y && \
    apt clean

# Set working directory
WORKDIR /opt/ml/app

COPY uv.lock pyproject.toml .python-version README.md ./
COPY src ./src

# Install uv venv
RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \
    uv venv --python ${PYTHON_VERSION} /opt/ml/app/.venv && \
    . /opt/ml/app/.venv/bin/activate && \
    uv sync

COPY config ./config
COPY examples ./examples
