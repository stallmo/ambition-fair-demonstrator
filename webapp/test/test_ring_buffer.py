"""Tests for :mod:`utils.ring_buffer` (ported near-verbatim from the desktop app)."""
from __future__ import annotations

import numpy as np
import pytest

from utils.ring_buffer import RingBuffer


def test_append_and_size_track_capacity() -> None:
    """Size should grow with appends up to capacity, then stay capped."""
    buf = RingBuffer(capacity=3)
    assert buf.size == 0
    buf.append(1.0)
    buf.append(2.0)
    assert buf.size == 2
    buf.append(3.0)
    buf.append(4.0)  # overwrites the oldest (1.0)
    assert buf.size == 3


def test_get_window_returns_chronological_order_with_wraparound() -> None:
    """get_window() must return oldest-to-newest even when the ring has wrapped."""
    buf = RingBuffer(capacity=3)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        buf.append(v)
    # Only the last 3 appended values should remain, oldest first.
    np.testing.assert_array_equal(buf.get_window(), np.array([3.0, 4.0, 5.0]))
    np.testing.assert_array_equal(buf.get_window(2), np.array([4.0, 5.0]))


def test_get_window_n_none_or_larger_than_size_returns_all() -> None:
    """Requesting more items than are stored should not error, just cap at size."""
    buf = RingBuffer(capacity=5)
    buf.append(10.0)
    buf.append(20.0)
    np.testing.assert_array_equal(buf.get_window(None), np.array([10.0, 20.0]))
    np.testing.assert_array_equal(buf.get_window(100), np.array([10.0, 20.0]))


def test_get_window_empty_buffer_returns_empty_array() -> None:
    """An empty buffer must return a zero-length array, not raise."""
    buf = RingBuffer(capacity=4)
    result = buf.get_window()
    assert result.shape == (0,)


def test_clear_resets_without_reallocating() -> None:
    """clear() drops all entries; subsequent appends behave like a fresh buffer."""
    buf = RingBuffer(capacity=3)
    buf.append(1.0)
    buf.append(2.0)
    buf.clear()
    assert buf.size == 0
    np.testing.assert_array_equal(buf.get_window(), np.array([]))
    buf.append(9.0)
    assert buf.size == 1
    np.testing.assert_array_equal(buf.get_window(), np.array([9.0]))


def test_non_positive_capacity_rejected() -> None:
    """Defensive guard: capacity must be positive."""
    with pytest.raises(ValueError):
        RingBuffer(capacity=0)
