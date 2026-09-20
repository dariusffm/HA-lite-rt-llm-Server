"""The s6 env-persist loop in ``rootfs/etc/cont-init.d/01-config.sh``.

s6-overlay v3 reads the service environment from files in
``/run/s6/container_environment``. Whatever ``01-config.sh`` exports but
fails to write there never reaches uvicorn. A filter of
``^(LITERT_|HF_TOKEN$)`` matched a bare ``HF_TOKEN`` line that printenv
never emits, so the token was silently dropped and gated model downloads
came back 401.

The loop is extracted from the real script rather than restated here: a
copy would keep passing after someone edits the script.

No real token is ever handled. The fixture value deliberately does not
match Hugging Face's own format (``hf_`` + 34 alphanumerics) so secret
scanners do not flag this file, and values are never printed — only
compared.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

FAKE_TOKEN = "hf_TESTONLY_not_a_real_token"
SCRIPT = Path(__file__).resolve().parents[2] / "rootfs" / "etc" / "cont-init.d" / "01-config.sh"


def _persist_loop(target: Path) -> str:
    """The real persist block, rewritten to use ``target`` as its directory."""
    text = SCRIPT.read_text()
    match = re.search(r"^mkdir -p /run/s6/container_environment$.*?^done$", text, re.M | re.S)
    assert match, "persist block not found in 01-config.sh — did its shape change?"
    return match.group(0).replace("/run/s6/container_environment", str(target))


def _run(target: Path, env: dict[str, str]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["bash", "-c", _persist_loop(target)],
        env=env,  # no inherited environment: a developer's real HF_TOKEN must not leak in
        check=True,
        capture_output=True,
    )


def test_hf_token_is_persisted_for_s6(tmp_path: Path):
    target = tmp_path / "container_environment"
    _run(target, {"PATH": os.environ["PATH"], "HF_TOKEN": FAKE_TOKEN, "LITERT_PORT": "8080"})

    written = target / "HF_TOKEN"
    assert written.exists(), "HF_TOKEN was not persisted; uvicorn would start without it"
    assert written.read_text() == FAKE_TOKEN  # compared, never printed


def test_litert_vars_are_still_persisted(tmp_path: Path):
    target = tmp_path / "container_environment"
    _run(target, {"PATH": os.environ["PATH"], "LITERT_PORT": "8080", "HOME": "/root"})

    assert (target / "LITERT_PORT").read_text() == "8080"
    assert not (target / "HOME").exists(), "unrelated variables must not be persisted"


def test_absent_token_writes_no_file(tmp_path: Path):
    """``01-config.sh`` exports HF_TOKEN only when the option has a value, so a
    blank token stays unset rather than surfacing as a 401."""
    target = tmp_path / "container_environment"
    _run(target, {"PATH": os.environ["PATH"], "LITERT_PORT": "8080"})

    assert not (target / "HF_TOKEN").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_token_file_is_not_world_readable(tmp_path: Path):
    target = tmp_path / "container_environment"
    _run(target, {"PATH": os.environ["PATH"], "HF_TOKEN": FAKE_TOKEN})

    mode = (target / "HF_TOKEN").stat().st_mode & 0o077
    assert mode == 0, f"token file is group/world accessible (extra bits {mode:03o})"
