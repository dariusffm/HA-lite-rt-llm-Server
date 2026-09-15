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

    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path), hf_token="hf_test123")
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


async def test_list_enriches_quantization_from_catalog(tmp_path: Path):
    (tmp_path / "gemma-4-e2b.litertlm").write_bytes(b"x" * 16)
    (tmp_path / "unknown-model.litertlm").write_bytes(b"x" * 16)
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    models = {m.name: m for m in await reg.list()}
    assert models["gemma-4-e2b"].quantization == "int4"
    assert models["unknown-model"].quantization == "unknown"


async def test_pull_materializes_regular_file_from_hf_symlink(tmp_path: Path):
    """hf_hub_download returns a *relative* symlink (snapshots/ -> blobs/).
    On Linux, os.link() on a symlink hardlinks the symlink itself, which
    then dangles once placed in the models dir. The pulled file must be a
    regular file with the blob's size.
    """

    def fake_download(repo_id, filename, cache_dir, token, **_):
        blobs = Path(cache_dir) / "blobs"
        snap = Path(cache_dir) / "snapshots" / "abc"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        (blobs / "h").write_bytes(b"y" * 1024)
        link = snap / filename
        link.symlink_to(Path("..") / ".." / "blobs" / "h")
        return str(link)

    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    target = next(iter(MODEL_CATALOG))
    with patch(
        "litert_server.model_registry.huggingface.hf_hub_download",
        side_effect=fake_download,
    ):
        progress = [p async for p in reg.pull(target)]
    out = tmp_path / f"{target}.litertlm"
    assert progress[-1].status == "done"
    assert not out.is_symlink()
    assert out.stat().st_size == 1024
