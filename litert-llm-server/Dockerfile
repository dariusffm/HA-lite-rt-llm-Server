ARG BUILD_FROM
FROM $BUILD_FROM

ARG BUILD_ARCH

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apk add --no-cache libstdc++ libgomp

COPY app/ /opt/app/
WORKDIR /opt/app

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache .

COPY rootfs/ /

LABEL io.hass.version="0.1.0" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}"
