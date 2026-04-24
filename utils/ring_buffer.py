"""O(1) numpy ring buffer for chart data."""
import numpy as np


class RingBuffer:
    """Fixed-capacity ring buffer using a numpy array.

    Supports O(1) append and O(n) window retrieval.
    """

    def __init__(self, capacity: int, dtype=float):
        self._capacity = capacity
        self._buf = np.empty(capacity, dtype=dtype)
        self._head = 0   # next write position
        self._size = 0   # number of valid entries

    def append(self, value) -> None:
        self._buf[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def get_window(self, n: int | None = None) -> np.ndarray:
        """Return the last *n* items (or all if n is None/larger than size)."""
        if n is None or n >= self._size:
            n = self._size
        if n == 0:
            return np.empty(0, dtype=self._buf.dtype)
        # tail end first, then head portion
        start = (self._head - n) % self._capacity
        if start + n <= self._capacity:
            return self._buf[start : start + n].copy()
        # wraps around
        part1 = self._buf[start:]
        part2 = self._buf[: n - len(part1)]
        return np.concatenate([part1, part2])

    @property
    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._head = 0
        self._size = 0
