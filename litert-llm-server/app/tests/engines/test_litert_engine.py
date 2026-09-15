from pathlib import Path

from litert_server.engines.litert import LiteRTEngine


def test_engine_construction_does_not_load_model(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path)
    assert engine.models_dir == tmp_path
    assert engine.current_model is None


def test_engine_max_num_tokens_defaults_to_8192(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path)
    assert engine.max_num_tokens == 8192


def test_engine_max_num_tokens_stores_constructor_arg(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path, max_num_tokens=16384)
    assert engine.max_num_tokens == 16384
