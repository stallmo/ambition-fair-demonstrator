"""Verify the ported core/utils modules have no PyQt5 dependency.

Automates the acceptance-criteria grep check: none of the four ported
files (``core/data_generator.py``, ``core/anomaly_detector.py``,
``core/simulation_loader.py``, ``utils/ring_buffer.py``) may reference
``QObject``, ``pyqtSignal``, ``QTimer``, or import ``PyQt5`` anywhere
in their source (not just as top-level imports), since the webapp must
run without a Qt/GUI dependency.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pytest

_WEBAPP_ROOT = Path(__file__).resolve().parent.parent

_PORTED_FILES: List[Path] = [
    _WEBAPP_ROOT / "core" / "data_generator.py",
    _WEBAPP_ROOT / "core" / "anomaly_detector.py",
    _WEBAPP_ROOT / "core" / "simulation_loader.py",
    _WEBAPP_ROOT / "utils" / "ring_buffer.py",
]

# Any of these substrings appearing anywhere in a ported file's source
# indicates a lingering Qt dependency.
_FORBIDDEN_PATTERNS = [
    r"\bQObject\b",
    r"\bpyqtSignal\b",
    r"\bQTimer\b",
    r"from PyQt5",
    r"import PyQt5",
]


@pytest.mark.parametrize("path", _PORTED_FILES, ids=lambda p: p.name)
def test_ported_file_has_no_qt_references(path: Path) -> None:
    """Assert a ported file contains none of the forbidden Qt tokens.

    :param path: Path to the ported source file under test.
    :type path: pathlib.Path
    :returns: None
    :rtype: None
    """
    assert path.exists(), f"expected ported file to exist: {path}"
    source = path.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, source)
        assert not matches, f"{path} unexpectedly references Qt symbol matching {pattern!r}"
