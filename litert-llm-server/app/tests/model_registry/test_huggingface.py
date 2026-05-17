from pathlib import Path
from unittest.mock import patch

import pytest

from litert_server.model_registry.filesystem import FilesystemCache
from litert_server.model_registry.huggingface import (
    MODEL_CATALOG,
    HuggingFaceRegistry,
)


async def test_list_reflects_filesystem(tmp_path: Path):
    (tmp_path / "gemma-4-e2b.litertlm").write_bytes(b"x" * 512)
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    models = await reg.list()
    assert {m.name for m in models} == {"gemma-4-e2b"}


async def test_get_unknown_raises(tmp_path: Path):
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    with pytest.raises(KeyError):
        await reg.get("does-not-exist")


async def test_pull_known_emits_progress(tmp_path: Path):
    def fake_download(repo_id, filename, cache_dir, token, **_):
        out = Path(cache_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"y" * 1024)
        return str(out)

    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    target = next(iter(MODEL_CATALOG))
    with patch(
        "litert_server.model_registry.huggingface.hf_hub_download",
        side_effect=fake_download,
    ):
        progress = [p async for p in reg.pull(target)]
    assert progress[-1].status == "done"
    assert (tmp_path / f"{target}.litertlm").exists()


async def test_pull_forwards_hf_token(tmp_path: Path):
    captured: dict[str, object] = {}

    def fake_download(repo_id, filename, cache_dir, token, **_):
        captured["token"] = token
        out = Path(cache_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"y" * 16)
        return str(out)

    reg = HuggingFaceRegistry(
        cache=FilesystemCache(root=tmp_path), hf_token="hf_test123"
    )
    with patch(
        "litert_server.model_registry.huggingface.hf_hub_download",
        side_effect=fake_download,
    ):
        async for _ in reg.pull(next(iter(MODEL_CATALOG))):
            pass
    assert captured["token"] == "hf_test123"


async def test_pull_unknown_emits_error(tmp_path: Path):
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    progress = [p async for p in reg.pull("nope")]
    assert progress[-1].status == "error"


def test_empty_token_normalized_to_none(tmp_path: Path):
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path), hf_token="")
    assert reg.hf_token is None
    reg2 = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path), hf_token="   ")
    assert reg2.hf_token is None
    reg3 = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path), hf_token="hf_x")
    assert reg3.hf_token == "hf_x"


async def test_list_enriches_quantization_from_catalog(tmp_path: Path):
    (tmp_path / "gemma-4-e2b.litertlm").write_bytes(b"x" * 16)
    (tmp_path / "unknown-model.litertlm").write_bytes(b"x" * 16)
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    models = {m.name: m for m in await reg.list()}
    assert models["gemma-4-e2b"].quantization == "int4"
    assert models["unknown-model"].quantization == "unknown"
