"""Model names come from untrusted HTTP bodies and become filesystem paths."""

from pathlib import Path

import pytest

from litert_server.domain.model_names import (
    InvalidModelNameError,
    model_path,
    validate_model_name,
)
from litert_server.model_registry.filesystem import MODEL_EXT, FilesystemCache


@pytest.mark.parametrize("name", ["gemma-4-e2b", "qwen3-0.6b", "A", "a_b.c-d", "x" * 64])
def test_catalog_style_names_pass(name: str):
    assert validate_model_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "../evil",
        "../../share/evil",
        "sub/evil",
        "sub\\evil",
        ".hidden",
        "",
        "x" * 65,
        "evil$(whoami)",
        "a b",
    ],
)
def test_traversal_and_odd_names_are_rejected(name: str):
    with pytest.raises(InvalidModelNameError):
        validate_model_name(name)


def test_model_path_rejects_symlink_pointing_outside(tmp_path: Path):
    """The pattern alone is not enough: a symlink can satisfy it and still escape."""
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / f"escape{MODEL_EXT}").symlink_to(outside / f"escape{MODEL_EXT}")

    with pytest.raises(InvalidModelNameError):
        model_path(root, "escape", MODEL_EXT)


def test_delete_cannot_unlink_a_file_outside_the_models_dir(tmp_path: Path):
    """The actual attack: DELETE /api/delete passes ``name`` through unchanged."""
    root = tmp_path / "models"
    root.mkdir()
    victim = tmp_path / f"victim{MODEL_EXT}"
    victim.write_text("important")
    cache = FilesystemCache(root=root)

    with pytest.raises(InvalidModelNameError):
        cache.delete("../victim")

    assert victim.exists(), "a file outside the models dir was removed"
