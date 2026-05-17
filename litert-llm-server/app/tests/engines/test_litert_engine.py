from pathlib import Path

from litert_server.engines.litert import LiteRTEngine


def test_engine_construction_does_not_load_model(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path)
    assert engine.models_dir == tmp_path
    assert engine.current_model is None
