"""Static regression checks for :file:`webapp/Dockerfile` (story H1).

These are lightweight content/existence checks, not a full
``docker build`` (that would be slow and require a Docker daemon in
CI, so it is instead verified manually/in a deploy pipeline -- see the
story H1 acceptance criteria). The goal here is only to guard against
someone accidentally deleting/mangling a load-bearing Dockerfile
instruction (e.g. ``EXPOSE 8501`` or the ``streamlit run`` entrypoint)
without noticing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_WEBAPP_ROOT = Path(__file__).resolve().parent.parent
_DOCKERFILE = _WEBAPP_ROOT / "Dockerfile"
_DOCKERIGNORE = _WEBAPP_ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    """Read the Dockerfile's contents once for all tests in this module.

    :returns: The Dockerfile's full text content.
    :rtype: str
    """
    assert _DOCKERFILE.exists(), f"expected Dockerfile at {_DOCKERFILE}"
    return _DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_exists() -> None:
    """The Dockerfile must exist at ``webapp/Dockerfile``.

    :returns: None
    :rtype: None
    """
    assert _DOCKERFILE.is_file()


def test_dockerfile_uses_python_311_slim_base(dockerfile_text: str) -> None:
    """The image must be based on ``python:3.11-slim`` per the H1 spec.

    :param dockerfile_text: Fixture providing the Dockerfile's contents.
    :type dockerfile_text: str
    :returns: None
    :rtype: None
    """
    assert "FROM python:3.11-slim" in dockerfile_text


def test_dockerfile_installs_requirements_txt(dockerfile_text: str) -> None:
    """The Dockerfile must copy and pip-install ``requirements.txt``.

    :param dockerfile_text: Fixture providing the Dockerfile's contents.
    :type dockerfile_text: str
    :returns: None
    :rtype: None
    """
    assert "COPY requirements.txt" in dockerfile_text
    assert "RUN pip install" in dockerfile_text
    assert "requirements.txt" in dockerfile_text.split("RUN pip install", 1)[1].splitlines()[0]


def test_dockerfile_requirements_layer_precedes_app_code_copy(dockerfile_text: str) -> None:
    """``requirements.txt`` must be installed before the rest of the app is copied.

    Guards the layer-caching design goal: code-only changes should not
    invalidate the (slow) pip install layer.

    :param dockerfile_text: Fixture providing the Dockerfile's contents.
    :type dockerfile_text: str
    :returns: None
    :rtype: None
    """
    requirements_copy_idx = dockerfile_text.index("COPY requirements.txt")
    pip_install_idx = dockerfile_text.index("RUN pip install")
    app_copy_idx = dockerfile_text.index("COPY . .")
    assert requirements_copy_idx < pip_install_idx < app_copy_idx


def test_dockerfile_exposes_8501(dockerfile_text: str) -> None:
    """The container must declare port 8501 via ``EXPOSE``.

    :param dockerfile_text: Fixture providing the Dockerfile's contents.
    :type dockerfile_text: str
    :returns: None
    :rtype: None
    """
    assert "EXPOSE 8501" in dockerfile_text


def test_dockerfile_entrypoint_runs_streamlit_app(dockerfile_text: str) -> None:
    """The entrypoint/cmd must run ``streamlit run app.py`` bound to all interfaces.

    Also guards against a hanging first-run prompt (no TTY in
    ``docker run``) by requiring headless mode.

    :param dockerfile_text: Fixture providing the Dockerfile's contents.
    :type dockerfile_text: str
    :returns: None
    :rtype: None
    """
    assert "streamlit" in dockerfile_text
    assert "run" in dockerfile_text
    assert "app.py" in dockerfile_text
    assert "--server.address=0.0.0.0" in dockerfile_text
    assert "--server.port=8501" in dockerfile_text
    assert "--server.headless=true" in dockerfile_text


def test_dockerignore_exists_and_excludes_dev_only_artifacts() -> None:
    """``.dockerignore`` must exist and exclude dev-only/generated paths.

    Prevents accidentally baking the local dev venv, test suite, caches,
    or dev-only requirements file into the production image.

    :returns: None
    :rtype: None
    """
    assert _DOCKERIGNORE.is_file()
    text = _DOCKERIGNORE.read_text(encoding="utf-8")
    for excluded in (".venv/", "__pycache__", "test/", "requirements-dev.txt"):
        assert excluded in text, f"expected .dockerignore to exclude {excluded!r}"


def test_dockerignore_does_not_exclude_streamlit_config() -> None:
    """``.streamlit/`` must NOT be excluded -- it ships the theme config.

    :returns: None
    :rtype: None
    """
    text = _DOCKERIGNORE.read_text(encoding="utf-8")
    # A bare ".streamlit/" (or ".streamlit") ignore line would strip the
    # theme config the running app depends on (see ui_web/styling.py /
    # story G1); assert no such line is present.
    excluded_lines = {line.strip() for line in text.splitlines()}
    assert ".streamlit/" not in excluded_lines
    assert ".streamlit" not in excluded_lines
