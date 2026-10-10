"""Background-built, on-disk cache of averaged display pyramid levels.

A raw acquisition without pyramid levels gets nearest-neighbour preview
levels (:mod:`._preview`). Those are a strided view of full resolution, so a
coarse plane read from disk costs as much as a full-resolution one. The first
time any preview level of a source is read, this module averages all of its
levels in one pass over the source, in a single background thread, and writes
them to a raw memory-mapped file in a per-user cache directory. Previews read
from that file plane by plane as soon as the planes they need are written, and
from then on whenever the same data is opened again.

The builder thread never calls h5py: h5py 3.14 to 3.16 can deadlock napari
when one thread is inside h5py while another frees an h5py object. It reads
the source through prepared direct chunk readers (see :mod:`._fastio`), so
sources that need h5py for every read (compressed data) are not cached. It
also steps aside while the display is reading, so it never slows a Z step.

The cache never touches the data folder. Settings:

``NAPARI_LUXENDO_PYRAMID_CACHE``  ``0`` turns the cache off (previews stay
    nearest-neighbour samples).
``NAPARI_LUXENDO_CACHE_DIR``  where cache files go. Default: the user cache
    directory (``%LOCALAPPDATA%\\napari-luxendo\\cache`` on Windows,
    ``~/Library/Caches/napari-luxendo`` on macOS, ``~/.cache/napari-luxendo``
    elsewhere).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import queue
from pathlib import Path

import numpy as np

from ._fastio import _BACKGROUND, seconds_since_foreground_read

logger = logging.getLogger(__name__)

# Bump when the cache file layout or the averaging changes.
_VERSION = 2
# Full-resolution planes and YX edge of one brick of the build pass.
_BRICK_Z = 16
_BRICK_YX = 1024
# Free space to leave on the cache volume, beyond the cache itself.
_FREE_MARGIN = 2 * 2**30

# Partial cache files older than this are left over from a closed napari.
_STALE_TMP_S = 24 * 3600
# The builder waits until the display has not read for this long (seconds).
_IDLE_S = 0.3

_QUEUE: queue.Queue | None = None
_EXECUTOR_LOCK = threading.Lock()
_CACHES: dict[str, PreviewCache] = {}


def cache_enabled() -> bool:
    return os.environ.get('NAPARI_LUXENDO_PYRAMID_CACHE', '1') != '0'


def cache_dir() -> Path:
    env = os.environ.get('NAPARI_LUXENDO_CACHE_DIR')
    if env:
        return Path(env)
    if sys.platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
        return base / 'napari-luxendo' / 'cache'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Caches' / 'napari-luxendo'
    return Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'napari-luxendo'


def _submit(job) -> None:
    """Run *job* on the single cache-building thread.

    The thread is a daemon so closing napari never waits for a build; an
    interrupted build leaves only a ``.tmp`` file, removed by a later build.
    """
    global _QUEUE
    with _EXECUTOR_LOCK:
        if _QUEUE is None:
            _QUEUE = queue.Queue()
            threading.Thread(target=_worker, args=(_QUEUE,), name='luxendo-preview-cache',
                             daemon=True).start()
    _QUEUE.put(job)


def _worker(jobs: queue.Queue) -> None:
    _BACKGROUND.active = True
    while True:
        jobs.get()()


def _remove_stale_partials(folder: Path) -> None:
    now = time.time()
    for tmp in [*folder.glob('*.tmp'), *folder.glob('*.part')]:
        try:
            if now - tmp.stat().st_mtime > _STALE_TMP_S:
                tmp.unlink()
        except OSError:
            pass


def wait_for_builds(timeout: float | None = None) -> None:
    """Block until every queued cache build has finished (tests, benchmarks)."""
    deadline = None if timeout is None else time.monotonic() + timeout
    for cache in list(_CACHES.values()):
        cache.wait(None if deadline is None else max(0.0, deadline - time.monotonic()))


def close_caches() -> None:
    """Close cache files; running builds stop at the next brick."""
    for cache in list(_CACHES.values()):
        cache.close()
    _CACHES.clear()


class PreviewCache:
    """Averaged copies of one source's preview levels, filled in the background.

    Level ``i`` has factor ``factors[i]`` on every axis and shape
    ``ceil(n / factor)``, like the nearest-neighbour preview it replaces;
    each voxel is the mean of the (possibly clipped) block it covers.

    On disk: ``<key>.preview`` holds the levels one after another as raw C-order
    arrays, and ``<key>.json`` (written last) says the file is complete.
    """

    def __init__(self, source, factors, path: Path):
        self.source = source
        self.factors = list(factors)
        self.shapes = [tuple(-(-n // f) for n in source.shape) for f in self.factors]
        self.dtype = np.dtype(source.dtype)
        self.path = path
        self.marker = path.with_suffix('.json')
        self.ready = False
        self.failed = False
        self.done_z = 0  # full-resolution planes whose cached planes are written
        self._lock = threading.RLock()
        self._maps: list[np.memmap] = []
        self._started = False
        self._closed = False
        self._finished = threading.Event()
        if self.marker.is_file():
            self._open_finished()

    @classmethod
    def for_source(cls, source, factors):
        """The shared cache for *source*, or None if it cannot or should not be cached."""
        if not cache_enabled() or not hasattr(source, 'cache_key'):
            return None
        try:
            key = {'version': _VERSION, 'source': source.cache_key(), 'factors': list(factors),
                   'shape': list(source.shape), 'dtype': np.dtype(source.dtype).str}
        except (OSError, AttributeError, TypeError) as exc:
            logger.debug('No preview cache: %s', exc)
            return None
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
        with _EXECUTOR_LOCK:
            cache = _CACHES.get(digest)
            if cache is None or cache._closed:
                cache = _CACHES[digest] = cls(source, factors, cache_dir() / f'{digest}.preview')
            return cache

    # ------------------------------------------------------------------ reads

    def read(self, level: int, key) -> np.ndarray | None:
        """Cached pixels for *key* of *level*, or None if not cached (yet).

        The first call queues the build.
        """
        if not self.ready:
            self._start()
        with self._lock:
            if not self._maps or self._closed:
                return None
            if not self.ready:
                f = self.factors[level]
                zs = range(*key[0].indices(self.shapes[level][0]))
                if len(zs) and (max(zs) + 1) * f > self.done_z \
                        and self.done_z < self.source.shape[0]:
                    return None
            try:
                # A copy, so no view keeps the file mapped once it is replaced.
                return np.array(self._maps[level][key])
            except Exception as exc:  # never break the display over the cache
                logger.debug('Preview cache read failed: %s', exc)
                return None

    def wait(self, timeout: float | None = None) -> bool:
        if not self._started and not self.ready:
            return self.ready
        return self._finished.wait(timeout)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._maps = []

    def _offsets(self):
        offsets, total = [], 0
        for shape in self.shapes:
            offsets.append(total)
            total += int(np.prod(shape)) * self.dtype.itemsize
        return offsets, total

    def _map(self, path: Path, mode: str) -> list[np.memmap]:
        offsets, _ = self._offsets()
        return [np.memmap(path, dtype=self.dtype, mode=mode, offset=offset, shape=shape)
                for offset, shape in zip(offsets, self.shapes)]

    def _open_finished(self) -> None:
        try:
            info = json.loads(self.marker.read_text())
            _, total = self._offsets()
            if not info.get('complete') or info.get('dtype') != self.dtype.str \
                    or [tuple(s) for s in info.get('shapes', [])] != self.shapes \
                    or self.path.stat().st_size != total:
                raise ValueError('incomplete or mismatched cache file')
            maps = self._map(self.path, 'r')
        except Exception as exc:
            logger.info('Ignoring preview cache %s: %s', self.path, exc)
            for stale in (self.marker, self.path):
                try:
                    stale.unlink()
                except OSError:
                    pass
            return
        with self._lock:
            self._maps = maps
            self.ready = True
            self.done_z = self.source.shape[0]
        self._finished.set()

    # ------------------------------------------------------------------ build

    def _start(self) -> None:
        with self._lock:
            if self._started or self.ready or self.failed or self._closed:
                return
            self._started = True
        try:
            # Any h5py work (mapping, chunk index) happens here, on the
            # reading thread, so the builder thread never needs h5py.
            ok = getattr(self.source, 'prepare_background', lambda: False)()
        except Exception as exc:
            logger.debug('Preview cache source not ready: %s', exc)
            ok = False
        if not ok:
            self.failed = True
            self._finished.set()
            logger.info('Not caching previews of %s: its reads need h5py', self.path.stem)
            return
        _submit(self._build_safely)

    def _build_safely(self) -> None:
        try:
            if not self._closed:
                self._build()
        except Exception as exc:
            self.failed = True
            logger.warning('Could not build preview cache %s: %s', self.path, exc)
            with self._lock:
                self._maps = []
        finally:
            self._finished.set()

    def _build(self) -> None:
        _, needed = self._offsets()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_partials(self.path.parent)
        free = shutil.disk_usage(self.path.parent).free
        if free < needed + _FREE_MARGIN:
            self.failed = True
            logger.warning('Not caching previews: %d MB needed, %d MB free in %s',
                           needed >> 20, free >> 20, self.path.parent)
            return
        part = self.path.with_name(f'{self.path.stem}.{os.getpid()}.{threading.get_ident()}.part')
        t0 = time.perf_counter()
        with open(part, 'wb') as handle:
            handle.truncate(needed)
        maps = self._map(part, 'r+')
        with self._lock:
            self._maps = maps
        self._fill(maps)
        with self._lock:
            # Readers copy under this lock, so once the maps are dropped
            # nothing holds the file mapped (Windows cannot replace it then).
            self._maps = []
        for m in maps:
            m.flush()
        del maps, m
        if self._closed:
            part.unlink(missing_ok=True)
            return
        os.replace(part, self.path)
        tmp = self.marker.with_name(f'{self.marker.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps({'complete': True, 'dtype': self.dtype.str,
                                   'shapes': [list(s) for s in self.shapes]}))
        os.replace(tmp, self.marker)
        self._open_finished()
        logger.info('Cached %d preview levels (%d MB) in %.1f s: %s', len(self.factors),
                    needed >> 20, time.perf_counter() - t0, self.path)

    def _wait_for_idle_display(self) -> None:
        """Let display reads go first: wait until they have paused."""
        while not self._closed:
            idle = seconds_since_foreground_read()
            if idle >= _IDLE_S:
                return
            time.sleep(_IDLE_S - idle)

    def _fill(self, maps) -> None:
        nz, ny, nx = self.source.shape
        top = max(self.factors)
        depth = top * max(1, -(-_BRICK_Z // top))
        edge = top * max(1, -(-_BRICK_YX // top))
        for z0 in range(0, nz, depth):
            z1 = min(nz, z0 + depth)
            for y0 in range(0, ny, edge):
                for x0 in range(0, nx, edge):
                    self._wait_for_idle_display()
                    if self._closed:
                        return
                    brick = np.asarray(self.source[(slice(z0, z1), slice(y0, min(ny, y0 + edge)),
                                                    slice(x0, min(nx, x0 + edge)))])
                    for m, f, mean in zip(maps, self.factors, block_means(brick, self.factors)):
                        m[z0 // f: z0 // f + mean.shape[0], y0 // f: y0 // f + mean.shape[1],
                          x0 // f: x0 // f + mean.shape[2]] = mean
            with self._lock:
                self.done_z = z1


def block_means(brick: np.ndarray, factors) -> list[np.ndarray]:
    """Means of ``f x f x f`` blocks of *brick* for each factor (clipped at the edges).

    Block sums are built coarse from fine, which is exact because the factors
    are nested (each divides the next) and integer sums add up.
    """
    kind = brick.dtype.kind
    acc = np.uint64 if kind in 'bu' else np.int64 if kind == 'i' else np.float64
    sums = brick
    prev = 1
    out = []
    for f in factors:
        rel = f // prev
        if f % prev:
            sums, rel = brick, f
        sums = _block_sum(sums, rel, acc)
        counts = [np.diff(np.append(np.arange(0, n, f), n)) for n in brick.shape]
        mean = sums / (counts[0][:, None, None] * counts[1][None, :, None] * counts[2][None, None, :])
        if kind in 'biu':
            info = np.iinfo(brick.dtype)
            mean = np.clip(np.rint(mean), info.min, info.max)
        out.append(mean.astype(brick.dtype))
        prev = f
    return out


def _block_sum(a: np.ndarray, f: int, acc) -> np.ndarray:
    """Sums of ``f x f x f`` blocks, the last block on each axis possibly smaller."""
    if f == 1:
        return a.astype(acc)
    if all(n % f == 0 for n in a.shape):
        nz, ny, nx = a.shape
        return a.reshape(nz // f, f, ny // f, f, nx // f, f).sum(axis=(1, 3, 5), dtype=acc)
    for axis in range(3):
        a = np.add.reduceat(a, np.arange(0, a.shape[axis], f), axis=axis, dtype=acc)
    return a
