from pathlib import Path

from litert_server.model_registry.filesystem import FilesystemCache


def test_lists_files_with_known_extension(tmp_path: Path):
    (tmp_path / "gemma-4-e2b.litertlm").write_bytes(b"x" * 1024)
    (tmp_path / "random.txt").write_text("ignore me")
    cache = FilesystemCache(root=tmp_path)
    names = [m.name for m in cache.scan()]
    assert names == ["gemma-4-e2b"]


def test_size_reflects_file_size(tmp_path: Path):
    (tmp_path / "x.litertlm").write_bytes(b"a" * 4096)
    cache = FilesystemCache(root=tmp_path)
    [m] = cache.scan()
    assert m.size_bytes == 4096
    assert m.path == tmp_path / "x.litertlm"


def test_delete_removes_file(tmp_path: Path):
    f = tmp_path / "y.litertlm"
    f.write_bytes(b"data")
    cache = FilesystemCache(root=tmp_path)
    cache.delete("y")
    assert not f.exists()


def test_delete_unknown_is_noop(tmp_path: Path):
    cache = FilesystemCache(root=tmp_path)
    cache.delete("nope")  # must not raise
