#!/usr/bin/with-contenv bashio
set -euo pipefail

bashio::log.info "Reading add-on configuration..."

# bashio prints a list as one raw element per line. The loop that persists the
# env vars for s6 keeps only a value's first line, so every list option has to
# become single-line JSON here — which is also what pydantic-settings parses.
config_list() {
    bashio::config "${1}" | jq -R -s -c 'split("\n") | map(select(length > 0))'
}

export LITERT_LOG_LEVEL="$(bashio::config 'log_level')"
export LITERT_DEFAULT_MODEL="$(bashio::config 'default_model')"
export LITERT_MAX_TOKENS="$(bashio::config 'max_tokens')"
export LITERT_TEMPERATURE="$(bashio::config 'temperature')"
export LITERT_TOOL_CALLING="$(bashio::config 'tool_calling')"
export LITERT_CONTEXT_LENGTH="$(bashio::config 'context_length')"
export LITERT_PROMPT_COMPACTION="$(bashio::config 'prompt_compaction')"
export LITERT_CONVERSATION_TTL="$(bashio::config 'conversation_ttl')"
export LITERT_TOOL_CALL_REPAIR="$(bashio::config 'tool_call_repair')"
LITERT_SWITCHING_TOOL_NAMES="$(config_list 'switching_tool_names')"
export LITERT_SWITCHING_TOOL_NAMES
bashio::log.info "switching tools: ${LITERT_SWITCHING_TOOL_NAMES}"
export LITERT_SWITCHING_BLOCK_REPLY="$(bashio::config 'switching_block_reply')"
export LITERT_ENGINE_WAIT_TIMEOUT="$(bashio::config 'engine_wait_timeout')"
export LITERT_GENERATION_TIMEOUT="$(bashio::config 'generation_timeout')"
export LITERT_MODELS_DIR="/data/models"
export LITERT_PORT="8080"

LITERT_PRELOAD_MODELS="$(config_list 'preload_models')"
export LITERT_PRELOAD_MODELS

# HuggingFace token: exported as HF_TOKEN (the env var huggingface_hub
# reads by default). Required for downloading gated google/*-litert-lm
# repos; left unset otherwise so blank values do not surface as 401.
if bashio::config.has_value 'hf_token'; then
    export HF_TOKEN="$(bashio::config 'hf_token')"
fi

# Persist for s6-overlay v3 (base-python:14.0.2 ships v3).
# The filter matches printenv's "NAME=value" lines, so HF_TOKEN needs the
# "=" — anchoring with "HF_TOKEN$" matched a bare name that never occurs and
# silently dropped the token, leaving gated model downloads unauthenticated.
# umask keeps the token file from being world-readable.
mkdir -p /run/s6/container_environment
chmod 700 /run/s6/container_environment
umask 077
printenv | grep -E '^(LITERT_|HF_TOKEN=)' | while IFS='=' read -r key value; do
    printf '%s' "${value}" > "/run/s6/container_environment/${key}"
done
