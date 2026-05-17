ARG BUILD_FROM
FROM $BUILD_FROM

ARG BUILD_ARCH

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Pull the uv binary directly from astral's image — avoids a separate
# python3-pip bootstrap layer.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# litert-lm-api ships only manylinux wheels (glibc) — Alpine/musl fails,
# Debian works. libgomp1 is the OpenMP runtime its bundled C library needs.
# python3-dev is install-time only (compile-from-source fallbacks); purged
# after the install layer to keep the final image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Single COPY+install — earlier attempt to split deps/source layers
# fell over because the install layer wheels only an empty __init__.py
# stub; subsequent COPY app/src/ leaves site-packages out of sync. Two
# minutes of rebuild is a smaller cost than a stale-package class of bug.
COPY app/ /opt/app/
WORKDIR /opt/app
RUN uv pip install --system --no-cache --break-system-packages . \
    && apt-get purge -y python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY rootfs/ /

LABEL io.hass.version="0.1.0" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}"
