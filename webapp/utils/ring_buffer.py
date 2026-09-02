"""O(1) numpy ring buffer for chart data.

Ported verbatim (algorithm unchanged) from
``core/utils/ring_buffer.py`` in the desktop app. This module has no Qt
dependency in the original either, so relocation is the only change;
type hints and docstrings were added to satisfy this repo's coding
standards.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

#: Scalar types accepted by :meth:`RingBuffer.append`.
_Number = Union[int, float, np.floating, np.integer]


class RingBuffer:
    """Fixed-capacity ring buffer backed by a numpy array.

    Supports O(1) append and O(n) window retrieval, used to feed the
    rolling chart window without unbounded memory growth.

    :param capacity: Maximum number of elements the buffer can hold.
    :type capacity: int
    :param dtype: Numpy dtype for the backing array, defaults to ``float``.
    :type dtype: type
    """

    def __init__(self, capacity: int, dtype: type = float) -> None:
        # Defensive check: a non-positive capacity would make append/get_window
        # divide-by-zero or index incorrectly.
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buf: npt.NDArray = np.empty(capacity, dtype=dtype)
        self._head = 0   # next write position
        self._size = 0   # number of valid entries

    def append(self, value: _Number) -> None:
        """Append a single value, overwriting the oldest entry once full.

        :param value: The value to store.
        :type value: int | float | numpy.floating | numpy.integer
        :returns: None
        :rtype: None
        """
        self._buf[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def get_window(self, n: Optional[int] = None) -> npt.NDArray:
        """Return the last *n* items in chronological order.

        :param n: Number of most-recent items to return. If ``None`` or
            larger than the current size, all stored items are returned.
        :type n: int | None
        :returns: Array of the last *n* items, oldest first.
        :rtype: numpy.ndarray
        """
        if n is None or n >= self._size:
            n = self._size
        if n == 0:
            return np.empty(0, dtype=self._buf.dtype)
        # tail end first, then head portion
        start = (self._head - n) % self._capacity
        if start + n <= self._capacity:
            return self._buf[start: start + n].copy()
        # wraps around
        part1 = self._buf[start:]
        part2 = self._buf[: n - len(part1)]
        return np.concatenate([part1, part2])

    @property
    def size(self) -> int:
        """Number of valid entries currently stored.

        :returns: Current element count (<= capacity).
        :rtype: int
        """
        return self._size

    def clear(self) -> None:
        """Reset the buffer to empty without reallocating memory.

        :returns: None
        :rtype: None
        """
        self._head = 0
        self._size = 0
