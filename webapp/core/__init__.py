"""Core (Qt-free) business logic for the web port of the demonstrator.

This package mirrors the desktop app's ``core/`` package but with all
PyQt5 dependencies removed: methods return values directly instead of
emitting Qt signals, so they can be driven from a Streamlit script re-run
loop instead of a Qt event loop.
"""
