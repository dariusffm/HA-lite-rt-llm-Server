ARG BUILD_FROM
FROM $BUILD_FROM

ARG BUILD_ARCH

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# litert-lm-api ships only manylinux wheels (glibc) — Alpine/musl fails,
# Debian works. libgomp1 is the OpenMP runtime its bundled C library needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages uv

# Single COPY+install — earlier attempt to split deps/source layers
# fell over because the install layer wheels only an empty __init__.py
# stub; subsequent COPY app/src/ leaves site-packages out of sync. Two
# minutes of rebuild is a smaller cost than a stale-package class of bug.
COPY app/ /opt/app/
WORKDIR /opt/app
RUN uv pip install --system --no-cache --break-system-packages .

COPY rootfs/ /

LABEL io.hass.version="0.1.0" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}"
