import json
import os

from litert_server.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LITERT_LOG_LEVEL", "debug")
    monkeypatch.setenv("LITERT_DEFAULT_MODEL", "gemma-4-e2b")
    monkeypatch.setenv("LITERT_MAX_TOKENS", "256")
    monkeypatch.setenv("LITERT_TEMPERATURE", "0.5")
    monkeypatch.setenv("LITERT_MODELS_DIR", "/data/models")
    monkeypatch.setenv("LITERT_PORT", "8080")
    monkeypatch.setenv("LITERT_PRELOAD_MODELS", json.dumps(["gemma-4-e2b"]))
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    s = Settings()
    assert s.log_level == "debug"
    assert s.default_model == "gemma-4-e2b"
    assert s.max_tokens == 256
    assert s.temperature == 0.5
    assert str(s.models_dir) == "/data/models"
    assert s.port == 8080
    assert s.preload_models == ["gemma-4-e2b"]
    assert s.hf_token == "hf_xxx"


def test_settings_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LITERT_") or k == "HF_TOKEN":
            monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.log_level == "info"
    assert s.preload_models == []
    assert s.hf_token is None


def test_empty_hf_token_becomes_none(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LITERT_") or k == "HF_TOKEN":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_TOKEN", "")
    s = Settings()
    assert s.hf_token is None


def test_tool_calling_env_false(monkeypatch):
    monkeypatch.setenv("LITERT_TOOL_CALLING", "false")
    from litert_server.config import Settings

    assert Settings().tool_calling is False


def test_tool_calling_defaults_true(monkeypatch):
    monkeypatch.delenv("LITERT_TOOL_CALLING", raising=False)
    from litert_server.config import Settings

    assert Settings().tool_calling is True


def test_context_length_env(monkeypatch):
    monkeypatch.setenv("LITERT_CONTEXT_LENGTH", "16384")
    assert Settings().context_length == 16384


def test_context_length_defaults(monkeypatch):
    monkeypatch.delenv("LITERT_CONTEXT_LENGTH", raising=False)
    assert Settings().context_length == 8192


def test_prompt_compaction_env(monkeypatch):
    monkeypatch.setenv("LITERT_PROMPT_COMPACTION", "off")
    assert Settings().prompt_compaction == "off"


def test_prompt_compaction_defaults_auto(monkeypatch):
    monkeypatch.delenv("LITERT_PROMPT_COMPACTION", raising=False)
    assert Settings().prompt_compaction == "auto"


def test_log_level_accepts_critical(monkeypatch):
    # config.yaml's schema offers `critical` (MINOR 7); the Literal must match.
    monkeypatch.setenv("LITERT_LOG_LEVEL", "critical")
    assert Settings().log_level == "critical"


def test_conversation_ttl_env(monkeypatch):
    monkeypatch.setenv("LITERT_CONVERSATION_TTL", "120")
    assert Settings().conversation_ttl == 120


def test_conversation_ttl_defaults_to_300(monkeypatch):
    monkeypatch.delenv("LITERT_CONVERSATION_TTL", raising=False)
    assert Settings().conversation_ttl == 300


def test_conversation_ttl_rejects_out_of_range(monkeypatch):
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("LITERT_CONVERSATION_TTL", "-1")
    with pytest.raises(ValidationError):
        Settings()


def test_tool_call_repair_env_false(monkeypatch):
    monkeypatch.setenv("LITERT_TOOL_CALL_REPAIR", "false")
    assert Settings().tool_call_repair is False


def test_tool_call_repair_defaults_true(monkeypatch):
    monkeypatch.delenv("LITERT_TOOL_CALL_REPAIR", raising=False)
    assert Settings().tool_call_repair is True


def test_generation_timeout_env(monkeypatch):
    monkeypatch.setenv("LITERT_GENERATION_TIMEOUT", "60")
    assert Settings().generation_timeout == 60


def test_generation_timeout_defaults_to_240(monkeypatch):
    monkeypatch.delenv("LITERT_GENERATION_TIMEOUT", raising=False)
    assert Settings().generation_timeout == 240


def test_generation_timeout_rejects_out_of_range(monkeypatch):
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("LITERT_GENERATION_TIMEOUT", "601")
    with pytest.raises(ValidationError):
        Settings()


def test_switching_defaults_match_addon_schema(monkeypatch):
    from pathlib import Path

    import yaml

    monkeypatch.delenv("LITERT_SWITCHING_TOOL_NAMES", raising=False)
    addon = yaml.safe_load((Path(__file__).parents[2] / "config.yaml").read_text())
    assert Settings().switching_tool_names == addon["options"]["switching_tool_names"]
