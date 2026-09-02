"""Streamlit UI package for the web port of the demonstrator.

Modules under this package hold Streamlit-specific rendering and
``st.fragment`` polling loops built on top of the Qt-free ``core``/
``state`` packages.

.. note::
   This file is created defensively (only if absent) since more than
   one story may add modules to this package around the same time; do
   not overwrite it if it already exists with additional content.
"""
