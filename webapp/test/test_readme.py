"""Static regression checks for :file:`webapp/README.md` (story H3).

These are lightweight content-presence checks, not a doc-quality
evaluation -- the goal is only to guard against someone accidentally
deleting the README or a load-bearing command/section from it without
noticing (e.g. the exact ``streamlit run``/``docker build``/``docker
compose`` commands, or the documented fidelity gaps).
"""
from __future__ import annotations

from pathlib import Path

import pytest

_WEBAPP_ROOT = Path(__file__).resolve().parent.parent
_README = _WEBAPP_ROOT / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    """Read the README's contents once for all tests in this module.

    :returns: The README's full text content.
    :rtype: str
    """
    assert _README.exists(), f"expected README at {_README}"
    return _README.read_text(encoding="utf-8")


def test_readme_exists() -> None:
    """The README must exist at ``webapp/README.md``.

    :returns: None
    :rtype: None
    """
    assert _README.is_file()


def test_readme_documents_local_run_commands(readme_text: str) -> None:
    """The README must give the exact local venv + install + run commands.

    :param readme_text: Fixture providing the README's contents.
    :type readme_text: str
    :returns: None
    :rtype: None
    """
    assert "pip install -r requirements.txt" in readme_text
    assert "streamlit run app.py" in readme_text
    assert "http://localhost:8501" in readme_text


def test_readme_documents_docker_commands(readme_text: str) -> None:
    """The README must give exact ``docker build``/``docker run`` commands.

    :param readme_text: Fixture providing the README's contents.
    :type readme_text: str
    :returns: None
    :rtype: None
    """
    assert "docker build -t pm-demonstrator-web" in readme_text
    assert "docker run -p 8501:8501 pm-demonstrator-web" in readme_text


def test_readme_documents_docker_compose_commands(readme_text: str) -> None:
    """The README must give ``docker compose up`` (and note the v1 alias).

    :param readme_text: Fixture providing the README's contents.
    :type readme_text: str
    :returns: None
    :rtype: None
    """
    assert "docker compose up" in readme_text
    assert "docker-compose up" in readme_text


def test_readme_lists_all_six_fidelity_gaps(readme_text: str) -> None:
    """The README must explicitly cover each of the 6 named fidelity gaps.

    Uses loose substring checks (not exact plan wording) since the README
    is meant to explain each gap in reader-friendly prose, not quote the
    plan verbatim.

    :param readme_text: Fixture providing the README's contents.
    :type readme_text: str
    :returns: None
    :rtype: None
    """
    lower = readme_text.lower()
    # 1. 300ms cadence (story P2, raised from an original 150ms) vs
    #    original 100ms/50ms dual timers.
    assert "300ms" in lower
    assert "100ms" in lower and "50ms" in lower
    # 2. Simulation replay burstiness above ~5x speed.
    assert "5×" in readme_text or "5x" in lower
    assert "bursty" in lower or "burst" in lower
    # 3. No per-row background tinting in the logbook (emoji-coded instead).
    assert "tint" in lower
    assert "⚪ unclassified".lower() in lower or "unclassified" in lower
    # 4. Popover vs free-floating dashboard window.
    assert "popover" in lower
    assert "free-floating" in lower
    # 5. No keyboard shortcuts.
    assert "keyboard shortcut" in lower
    # 6. Single-browser-tab assumption / no shared live state across tabs.
    assert "tab" in lower and "session" in lower


def test_readme_mentions_no_manual_csv_setup_needed(readme_text: str) -> None:
    """The README must clarify simulation CSVs auto-generate on boot.

    :param readme_text: Fixture providing the README's contents.
    :type readme_text: str
    :returns: None
    :rtype: None
    """
    assert "auto-generate" in readme_text.lower()
