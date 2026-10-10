"""In-memory cache of displayed 2D blocks, so panning back and forth and
prefetched surroundings show without touching the disk.

Every display level is a Dask array of ``(1, 512, 512)`` blocks over a
sliceable source. :class:`CachedSource` sits between Dask and that source and
keeps whole blocks (one Z plane, 512 x 512 in Y and X) in one process-wide,
least-recently-used store with a memory budget:

* A display read that misses reads only the requested crop, as before, so
  the first view of an area is never slower. A read of a whole block (what
  the prefetcher asks for) is kept.
* The view controller (:mod:`._scroll`) finds the source and plane napari
  shows with :func:`locate`, then prefetches whole blocks around the view
  with :meth:`CachedSource.fill_block` in a background thread (no Dask
  there) and checks with :meth:`CachedSource.has_region` whether a view is
  already in memory.

Nothing is written to disk. Settings:

``NAPARI_LUXENDO_TILE_CACHE_MB``  memory budget in MiB. Default: 1024, or
    a tenth of physical memory if that is less. ``0`` turns it off.
"""
from __future__ import annotations

import os
import threading
import uuid
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Optional

import numpy as np

BLOCK_YX = 512

_LOCAL = threading.local()




def budget_bytes() -> int:
    env = os.environ.get('NAPARI_LUXENDO_TILE_CACHE_MB')
    if env is not None:
        try:
            return max(0, int(float(env) * 2**20))
        except ValueError:
            pass
    default = 1024 * 2**20
    try:
        import psutil

        default = min(default, psutil.virtual_memory().total // 10)
    except Exception:
        pass
    return int(default)


class TileStore:
    """Thread-safe LRU of numpy blocks with a byte budget."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.nbytes = 0
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._items

    def put(self, key, value: np.ndarray) -> None:
        if value.nbytes > self.capacity:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.nbytes -= old.nbytes
            self._items[key] = value
            self.nbytes += value.nbytes
            while self.nbytes > self.capacity and self._items:
                _, dropped = self._items.popitem(last=False)
                self.nbytes -= dropped.nbytes

    def drop(self, token: str) -> None:
        with self._lock:
            for key in [k for k in self._items if k[0] == token]:
                self.nbytes -= self._items.pop(key).nbytes

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.nbytes = 0


_STORE: Optional[TileStore] = None
_STORE_LOCK = threading.Lock()


def store() -> Optional[TileStore]:
    """The process-wide block store, or None when the budget is 0."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            capacity = budget_bytes()
            if capacity <= 0:
                return None
            _STORE = TileStore(capacity)
        return _STORE


def clear() -> None:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.clear()
        _STORE = None


@contextmanager
def locate():
    """Reads in this thread record ``(source, z)`` in the yielded list and
    return zeros instead of reading. Use with Dask's synchronous scheduler."""
    previous = getattr(_LOCAL, 'located', None)
    found: list = []
    _LOCAL.located = found
    try:
        yield found
    finally:
        _LOCAL.located = previous


class CachedSource:
    """Wrap a sliceable ``(Z, Y, X)`` source with the block store.

    *cacheable* says whether what *inner* returns right now may be kept (a
    preview whose averaged cache is still building returns samples that
    would go stale).
    """

    ndim = 3

    def __init__(self, inner, cacheable: Optional[Callable[[], bool]] = None) -> None:
        self.inner = inner
        self.shape, self.dtype = tuple(inner.shape), np.dtype(inner.dtype)
        self.cacheable = cacheable
        self.token = uuid.uuid4().hex
        weakref.finalize(self, _drop, self.token)

    def __getattr__(self, name):
        # cache_key() and friends of the wrapped source.
        if name == 'inner':
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __getitem__(self, key):
        located = getattr(_LOCAL, 'located', None)
        if located is not None:
            key = tuple(slice(*s.indices(n)) for s, n in zip(key, self.shape))
            located.append((self, key[0].start))
            return np.zeros(tuple(len(range(s.start, s.stop, s.step)) for s in key), self.dtype)
        tiles = store()
        if tiles is None or not isinstance(key, tuple) or len(key) != 3 \
                or any(not isinstance(s, slice) for s in key):
            return self._read(key)
        key = tuple(slice(*s.indices(n)) for s, n in zip(key, self.shape))
        zs, ys, xs = key
        if zs.stop - zs.start != 1 or ys.step != 1 or xs.step != 1 \
                or ys.stop <= ys.start or xs.stop <= xs.start:
            return self._read(key)
        z = zs.start
        b = BLOCK_YX
        by0, by1 = ys.start // b, (ys.stop - 1) // b
        bx0, bx1 = xs.start // b, (xs.stop - 1) // b
        if by0 != by1 or bx0 != bx1:
            # Dask asks per block; anything else is assembled block by block.
            out = np.empty((1, ys.stop - ys.start, xs.stop - xs.start), dtype=self.dtype)
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    y0, y1 = max(ys.start, by * b), min(ys.stop, (by + 1) * b)
                    x0, x1 = max(xs.start, bx * b), min(xs.stop, (bx + 1) * b)
                    out[:, y0 - ys.start:y1 - ys.start, x0 - xs.start:x1 - xs.start] = \
                        self[(zs, slice(y0, y1), slice(x0, x1))]
            return out
        tile_key = (self.token, z, by0, bx0)
        tile = tiles.get(tile_key)
        if tile is not None:
            return tile[:, ys.start - by0 * b:ys.stop - by0 * b,
                        xs.start - bx0 * b:xs.stop - bx0 * b].copy()
        full = (ys.start == by0 * b and ys.stop == min((by0 + 1) * b, self.shape[1])
                and xs.start == bx0 * b and xs.stop == min((bx0 + 1) * b, self.shape[2]))
        if not full:
            return self._read(key)  # the first view of an area reads only its crop
        allowed = self.cacheable is None or self.cacheable()
        data = np.asarray(self._read(key))
        if allowed:
            tiles.put(tile_key, data)
            return data.copy()
        return data

    def _read(self, key):
        return self.inner[key]

    def _block(self, by: int, bx: int):
        b = BLOCK_YX
        return (slice(by * b, min((by + 1) * b, self.shape[1])),
                slice(bx * b, min((bx + 1) * b, self.shape[2])))

    def has_region(self, z: int, y0: int, y1: int, x0: int, x1: int) -> bool:
        """True if every block of plane *z* touching ``[y0:y1, x0:x1]`` is in memory."""
        tiles = store()
        if tiles is None:
            return False
        b = BLOCK_YX
        return all((self.token, z, by, bx) in tiles
                   for by in range(max(0, y0) // b, (min(y1, self.shape[1]) - 1) // b + 1)
                   for bx in range(max(0, x0) // b, (min(x1, self.shape[2]) - 1) // b + 1))

    def fill_block(self, z: int, by: int, bx: int) -> None:
        """Read block (*by*, *bx*) of plane *z* into memory unless it is there."""
        tiles = store()
        if tiles is None or (self.token, z, by, bx) in tiles:
            return
        if self.cacheable is not None and not self.cacheable():
            return
        ys, xs = self._block(by, bx)
        tiles.put((self.token, z, by, bx), np.asarray(self._read((slice(z, z + 1), ys, xs))))


def _drop(token: str) -> None:
    tiles = _STORE
    if tiles is not None:
        tiles.drop(token)
