"""Static regression checks for :file:`webapp/docker-compose.yml` (story H2).

These are lightweight content/existence checks, not an actual
``docker compose up`` (that would be slow, require a Docker daemon, and
be flaky in CI, so it is instead verified manually -- see the story H2
acceptance criteria). The goal here is only to guard against someone
accidentally deleting/mangling load-bearing Compose directives (e.g.
``restart: unless-stopped`` or the ``8501:8501`` port mapping) without
noticing.

Deliberately implemented with plain string checks (mirroring
:file:`test_dockerfile.py`'s established pattern) rather than parsing
the file with PyYAML, since PyYAML is not currently a project
dependency (see ``requirements.txt`` / ``requirements-dev.txt``) and
adding one solely for a static test is unnecessary.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_WEBAPP_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE_FILE = _WEBAPP_ROOT / "docker-compose.yml"
_DOCKERFILE = _WEBAPP_ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def compose_text() -> str:
    """Read the Compose file's contents once for all tests in this module.

    :returns: The Compose file's full text content.
    :rtype: str
    """
    assert _COMPOSE_FILE.exists(), f"expected docker-compose.yml at {_COMPOSE_FILE}"
    return _COMPOSE_FILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose_lines(compose_text: str) -> list[str]:
    """Split the Compose file into stripped, non-empty, non-comment lines.

    :param compose_text: Fixture providing the Compose file's raw text.
    :type compose_text: str
    :returns: List of stripped lines with comments and blank lines removed.
    :rtype: list[str]
    """
    lines = []
    for raw_line in compose_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def test_compose_file_exists() -> None:
    """The Compose file must exist at ``webapp/docker-compose.yml``.

    :returns: None
    :rtype: None
    """
    assert _COMPOSE_FILE.is_file()


def test_compose_file_has_no_deprecated_version_key(compose_lines: list[str]) -> None:
    """The Compose file must omit the obsolete top-level ``version:`` key.

    Modern ``docker compose`` (Compose Specification / v2 plugin) treats
    the file as version-less and warns that ``version:`` is obsolete, so
    it is intentionally not present.

    :param compose_lines: Fixture providing the Compose file's non-comment lines.
    :type compose_lines: list[str]
    :returns: None
    :rtype: None
    """
    assert not any(line.startswith("version:") for line in compose_lines)


def test_compose_defines_services_section(compose_text: str) -> None:
    """The Compose file must define a ``services:`` top-level section.

    :param compose_text: Fixture providing the Compose file's raw text.
    :type compose_text: str
    :returns: None
    :rtype: None
    """
    assert "services:" in compose_text


def test_compose_service_builds_from_local_dockerfile_context(compose_text: str) -> None:
    """The service must build from the local ``webapp/Dockerfile``, not a pre-built image pull.

    Guards the "no manual ``docker build``/``docker run`` steps" acceptance
    criterion: ``docker compose up`` alone must be able to produce the image.

    :param compose_text: Fixture providing the Compose file's raw text.
    :type compose_text: str
    :returns: None
    :rtype: None
    """
    assert "build:" in compose_text
    assert "context: ." in compose_text
    assert _DOCKERFILE.is_file(), "compose build context relies on webapp/Dockerfile existing"


def test_compose_service_maps_port_8501(compose_text: str) -> None:
    """The service must map host port 8501 to container port 8501, matching H1.

    :param compose_text: Fixture providing the Compose file's raw text.
    :type compose_text: str
    :returns: None
    :rtype: None
    """
    assert "8501:8501" in compose_text


def test_compose_service_has_restart_unless_stopped(compose_lines: list[str]) -> None:
    """The service must set ``restart: unless-stopped`` for booth resilience.

    This is the core H2 requirement: an unattended booth laptop must
    self-recover if the app process dies mid-demo.

    :param compose_lines: Fixture providing the Compose file's non-comment lines.
    :type compose_lines: list[str]
    :returns: None
    :rtype: None
    """
    assert "restart: unless-stopped" in compose_lines


def test_compose_service_does_not_mount_simulations_volume(compose_lines: list[str]) -> None:
    """The compose file must NOT declare a ``volumes:`` mapping at all (H1 design).

    Per H1's design, scenario CSVs are regenerated fresh inside the
    container's own filesystem on every boot, so they are intentionally
    ephemeral and must not be persisted via a host volume mount. This
    checks the actual (non-comment) directive lines only, so explanatory
    prose in comments referencing ``assets/simulations`` does not trip
    a false positive.

    :param compose_lines: Fixture providing the Compose file's non-comment lines.
    :type compose_lines: list[str]
    :returns: None
    :rtype: None
    """
    assert not any(line.startswith("volumes:") for line in compose_lines)
