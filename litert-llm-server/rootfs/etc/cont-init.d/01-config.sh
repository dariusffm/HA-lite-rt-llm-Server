#!/usr/bin/with-contenv bashio
bashio::log.info "Reading add-on configuration..."

export LITERT_LOG_LEVEL="$(bashio::config 'log_level')"
export LITERT_DEFAULT_MODEL="$(bashio::config 'default_model')"
export LITERT_MAX_TOKENS="$(bashio::config 'max_tokens')"
export LITERT_TEMPERATURE="$(bashio::config 'temperature')"
export LITERT_MODELS_DIR="/data/models"
export LITERT_PORT="8080"

LITERT_PRELOAD_MODELS="$(bashio::config 'preload_models')"
export LITERT_PRELOAD_MODELS

# HuggingFace token: exported as HF_TOKEN (the env var huggingface_hub reads
# by default). Required for downloading gated google/*-litert-lm repos.
HF_TOKEN_VALUE="$(bashio::config 'hf_token')"
if [ -n "${HF_TOKEN_VALUE}" ]; then
    export HF_TOKEN="${HF_TOKEN_VALUE}"
fi

printenv | grep -E '^(LITERT_|HF_TOKEN$)' > /var/run/s6/container_environment/litert.env
