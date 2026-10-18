"""Small bounded LRU cache for provider client objects.

Keys use a SHA-256 of the API key, never the raw key, so the key cannot leak
through cache introspection. The cached client itself necessarily holds the key.
"""

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar


T = TypeVar("T")


def key_hash(api_key: str | None) -> str:
    return hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()


class ClientCache(Generic[T]):
    def __init__(self, maxsize: int = 64):
        self._maxsize = maxsize
        self._items: OrderedDict[Hashable, T] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, key: Hashable, factory: Callable[[], T]) -> T:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        client = factory()
        with self._lock:
            self._items[key] = client
            self._items.move_to_end(key)
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)
        return client

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
