"""Architecture-rule enforcement via import-linter.

Runs the contracts defined in ``.importlinter`` as a normal pytest test
so violations show up in the same test run as everything else.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_import_linter_contracts_pass() -> None:
    app_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "run", "lint-imports", "--config", str(app_root / ".importlinter")],
        cwd=app_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
