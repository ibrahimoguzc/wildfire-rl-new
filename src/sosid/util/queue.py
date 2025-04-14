"""Custom queue implementations."""

import bisect
from collections import deque
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Protocol, TypeVar

from typing_extensions import Self

T = TypeVar("T")

__all__ = ["SortedQueue"]


class Comparable(Protocol):
    """A protocol for comparable objects."""

    def __lt__(self, other: Self) -> bool: ...


class SortedQueue(deque[T]):
    """Sorted queue based on the bisect algorithm.

    This class is a sorted queue that maintains the order of the
    items based on a key function that gets a comparable value from an
    item. If an item is already in the queue, it will be inserted to the
    right of the existing item resulting in a FIFO fallback.

    Args:
        items: The items to initialize the queue with.
        sort_input: Whether to sort the input.
        key: A function that gets a comparable value from an item. If
            None the items will be compared directly. The default is
            None.
    """

    __slots__ = ("_key",)

    def __init__(
        self,
        items: Iterable[T] = (),
        *,
        sort_input: bool = True,
        key: Callable[[T], Comparable] | None = None,
    ):
        self._key = key
        super().__init__(sorted(items, key=key) if sort_input else items)

    def bisect(self, item: T) -> int:
        """Find the index where the item should be inserted."""
        return bisect.bisect_right(self, self._key(item), key=self._key)

    def insort(self, item: T) -> None:
        """Insert a item while respecting order."""
        bisect.insort_right(self, item, key=self._key)

    def copy(self) -> Self:
        """Make a shallow copy of the queue."""
        return self.__class__(self, sort_input=False, key=self._key)

    def islice(self, start: int | None, stop: int | None) -> Self:
        """Get a slice of the queue."""
        return islice(self, start, stop)
