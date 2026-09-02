"""Pytest root conftest for the webapp package.

Having a ``conftest.py`` at the ``webapp/`` root (rather than only under
``webapp/test/``) makes pytest add this directory to ``sys.path`` in its
default "prepend" import mode, so tests can do plain
``from core.data_generator import ...`` / ``from utils.ring_buffer import
...`` imports without needing a ``pyproject.toml``/``pytest.ini``
``pythonpath`` setting or manual ``sys.path`` hacks in every test module.
"""
