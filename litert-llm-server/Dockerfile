ARG BUILD_FROM
FROM $BUILD_FROM

ARG BUILD_ARCH

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# litert-lm-api's bundled C library needs libstdc++ + OpenMP.
# If litert-lm-api ships only manylinux (glibc) wheels and the base
# image is musl-based (Alpine), add `gcompat` here so the wheel loads.
RUN apk add --no-cache libstdc++ libgomp

# --- deps layer (only invalidated when pyproject.toml / uv.lock changes) ---
COPY app/pyproject.toml app/uv.lock app/README.md /opt/app/
WORKDIR /opt/app
RUN pip install --no-cache-dir uv \
    && mkdir -p src/litert_server \
    && touch src/litert_server/__init__.py \
    && uv pip install --system --no-cache .

# --- source layer (invalidated on every code edit, but deps are cached) ---
COPY app/src/ /opt/app/src/

COPY rootfs/ /

LABEL io.hass.version="0.1.0" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}"
