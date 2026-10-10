"""Background-built, on-disk cache of averaged display pyramid levels.

A raw acquisition without pyramid levels gets nearest-neighbour preview
levels (:mod:`._preview`). Those are a strided view of full resolution, so a
coarse plane read from disk costs as much as a full-resolution one. The first
time any preview level of a source is read, this module averages all of its
levels in one pass over the source, in a single background thread, and writes
them to an HDF5 file in a per-user cache directory. Previews read from that
file plane by plane as soon as the planes they need are written, and from then
on whenever the same data is opened again.

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

logger = logging.getLogger(__name__)

# Bump when the cache file layout or the averaging changes.
_VERSION = 1
# Full-resolution planes and YX edge of one brick of the build pass.
_BRICK_Z = 16
_BRICK_YX = 1024
# Free space to leave on the cache volume, beyond the cache itself.
_FREE_MARGIN = 2 * 2**30

# Partial cache files older than this are left over from a closed napari.
_STALE_TMP_S = 24 * 3600

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
    while True:
        jobs.get()()


def _remove_stale_partials(folder: Path) -> None:
    now = time.time()
    for tmp in folder.glob('*.tmp'):
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
    """

    def __init__(self, source, factors, path: Path):
        self.source = source
        self.factors = list(factors)
        self.shapes = [tuple(-(-n // f) for n in source.shape) for f in self.factors]
        self.path = path
        self.ready = False
        self.failed = False
        self.done_z = 0  # full-resolution planes whose cached planes are written
        self._lock = threading.RLock()
        self._file = None
        self._datasets: list = []
        self._started = False
        self._closed = False
        self._finished = threading.Event()
        if path.is_file():
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
                cache = _CACHES[digest] = cls(source, factors, cache_dir() / f'{digest}.preview.h5')
            return cache

    # ------------------------------------------------------------------ reads

    def read(self, level: int, key) -> np.ndarray | None:
        """Cached pixels for *key* of *level*, or None if not cached (yet).

        The first call queues the build.
        """
        if not self.ready:
            self._start()
        with self._lock:
            if self._file is None or not self._datasets or self._closed:
                return None
            if not self.ready:
                f = self.factors[level]
                zs = range(*key[0].indices(self.shapes[level][0]))
                if len(zs) and (max(zs) + 1) * f > self.done_z \
                        and self.done_z < self.source.shape[0]:
                    return None
            try:
                return np.asarray(self._datasets[level][key])
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
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
            self._file, self._datasets = None, []

    def _open_finished(self) -> None:
        import h5py

        try:
            f = h5py.File(self.path, 'r')
            datasets = [f[f'level_{i}'] for i in range(len(self.factors))]
            if any(ds.shape != s for ds, s in zip(datasets, self.shapes)) \
                    or not f.attrs.get('complete', False):
                f.close()
                raise ValueError('incomplete or mismatched cache file')
        except Exception as exc:
            logger.info('Ignoring preview cache %s: %s', self.path, exc)
            try:
                self.path.unlink()
            except OSError:
                pass
            return
        with self._lock:
            old = self._file
            self._file, self._datasets = f, datasets
            self.ready = True
            self.done_z = self.source.shape[0]
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        self._finished.set()

    # ------------------------------------------------------------------ build

    def _start(self) -> None:
        with self._lock:
            if self._started or self.ready or self.failed or self._closed:
                return
            self._started = True
        _submit(self._build_safely)

    def _build_safely(self) -> None:
        try:
            if not self._closed:
                self._build()
        except Exception as exc:
            self.failed = True
            logger.warning('Could not build preview cache %s: %s', self.path, exc)
        finally:
            self._finished.set()

    def _build(self) -> None:
        import h5py

        itemsize = np.dtype(self.source.dtype).itemsize
        needed = sum(int(np.prod(s)) for s in self.shapes) * itemsize
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_partials(self.path.parent)
        free = shutil.disk_usage(self.path.parent).free
        if free < needed + _FREE_MARGIN:
            self.failed = True
            logger.warning('Not caching previews: %d MB needed, %d MB free in %s',
                           needed >> 20, free >> 20, self.path.parent)
            return
        tmp = self.path.with_name(f'{self.path.stem}.{os.getpid()}.{threading.get_ident()}.tmp')
        t0 = time.perf_counter()
        f = h5py.File(tmp, 'w')
        try:
            datasets = [f.create_dataset(f'level_{i}', shape=s, dtype=self.source.dtype)
                        for i, s in enumerate(self.shapes)]
            with self._lock:
                self._file, self._datasets = f, datasets
            self._fill(datasets)
            if not self._closed:
                f.attrs['complete'] = True
                f.flush()
        finally:
            with self._lock:
                self._file, self._datasets = None, []
                f.close()
        if self._closed:
            tmp.unlink(missing_ok=True)
            return
        os.replace(tmp, self.path)
        self._open_finished()
        logger.info('Cached %d preview levels (%d MB) in %.1f s: %s', len(self.factors),
                    needed >> 20, time.perf_counter() - t0, self.path)

    def _fill(self, datasets) -> None:
        nz, ny, nx = self.source.shape
        top = max(self.factors)
        depth = top * max(1, -(-_BRICK_Z // top))
        edge = top * max(1, -(-_BRICK_YX // top))
        for z0 in range(0, nz, depth):
            z1 = min(nz, z0 + depth)
            for y0 in range(0, ny, edge):
                for x0 in range(0, nx, edge):
                    if self._closed:
                        return
                    brick = np.asarray(self.source[(slice(z0, z1), slice(y0, min(ny, y0 + edge)),
                                                    slice(x0, min(nx, x0 + edge)))])
                    for ds, f, mean in zip(datasets, self.factors, block_means(brick, self.factors)):
                        ds[z0 // f: z0 // f + mean.shape[0], y0 // f: y0 // f + mean.shape[1],
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
