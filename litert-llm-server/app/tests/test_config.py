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
