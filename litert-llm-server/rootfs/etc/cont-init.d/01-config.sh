#!/usr/bin/with-contenv bashio
set -euo pipefail

bashio::log.info "Reading add-on configuration..."

export LITERT_LOG_LEVEL="$(bashio::config 'log_level')"
export LITERT_DEFAULT_MODEL="$(bashio::config 'default_model')"
export LITERT_MAX_TOKENS="$(bashio::config 'max_tokens')"
export LITERT_TEMPERATURE="$(bashio::config 'temperature')"
export LITERT_TOOL_CALLING="$(bashio::config 'tool_calling')"
export LITERT_CONTEXT_LENGTH="$(bashio::config 'context_length')"
export LITERT_PROMPT_COMPACTION="$(bashio::config 'prompt_compaction')"
export LITERT_CONVERSATION_TTL="$(bashio::config 'conversation_ttl')"
export LITERT_MODELS_DIR="/data/models"
export LITERT_PORT="8080"

LITERT_PRELOAD_MODELS="$(bashio::config 'preload_models')"
export LITERT_PRELOAD_MODELS

# HuggingFace token: exported as HF_TOKEN (the env var huggingface_hub
# reads by default). Required for downloading gated google/*-litert-lm
# repos; left unset otherwise so blank values do not surface as 401.
if bashio::config.has_value 'hf_token'; then
    export HF_TOKEN="$(bashio::config 'hf_token')"
fi

# Persist for s6-overlay v3 (base-python:14.0.2 ships v3).
mkdir -p /run/s6/container_environment
printenv | grep -E '^(LITERT_|HF_TOKEN$)' | while IFS='=' read -r key value; do
    printf '%s' "${value}" > "/run/s6/container_environment/${key}"
done
