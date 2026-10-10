"""On-demand display pyramids for large raw acquisitions.

Level zero is unchanged. Until a level is cached, it is a nearest-neighbour
sample of full resolution, which is free to set up but costs as much disk I/O
per plane as full resolution (sampled rows still touch every page). The first
time a preview is shown, its levels are averaged once in a background thread
into a cache file (see :mod:`._preview_cache`), and reads switch to it plane
by plane as it fills. No sidecar files are written next to the data.
"""
from __future__ import annotations

import os
import uuid

import dask.array as da
import numpy as np

from ._preview_cache import PreviewCache
from ._tilecache import BLOCK_YX, CachedSource


def preview_factors(shape):
    if os.environ.get('NAPARI_LUXENDO_PREVIEW', '1') == '0':
        return []
    if np.prod(shape) <= 16_000_000 or max(shape[-2:]) <= 1024:
        return []
    factors = []
    factor = 1
    while max(-(-n // factor) for n in shape[-2:]) > 512 \
            or np.prod([-(-n // factor) for n in shape]) > 8_000_000:
        factor *= 2
        factors.append(factor)
    return factors


class VolumeSource:
    ndim = 3

    def __init__(self, vol):
        self.vol = vol
        self.shape, self.dtype = vol.shape, vol.dtype

    def __getitem__(self, key):
        return self.vol.read(0, key)

    def cache_key(self):
        return [dataset_identity(self.vol.datasets[0])]


def dataset_identity(ds):
    """What identifies the pixels of an HDF5 dataset across sessions."""
    path = os.path.abspath(ds.file.filename)
    st = os.stat(path)
    return [path, ds.name, st.st_size, st.st_mtime_ns, list(ds.shape), ds.dtype.str]


class SampledSource:
    ndim = 3

    def __init__(self, source, factor, cache=None, level=0):
        self.source, self.factor = source, factor
        self.shape = tuple(-(-n // factor) for n in source.shape)
        self.dtype = source.dtype
        self.cache, self.level = cache, level

    def __getitem__(self, key):
        if self.cache is not None:
            cached = self.cache.read(self.level, key)
            if cached is not None:
                return cached
        mapped = []
        for s, n, full in zip(key, self.shape, self.source.shape):
            start, stop, step = s.indices(n)
            mapped.append(slice(start*self.factor, min(stop*self.factor, full), step*self.factor))
        return self.source[tuple(mapped)]


def preview_levels(source):
    factors = preview_factors(source.shape)
    cache = PreviewCache.for_source(source, factors) if factors else None
    return [da.from_array(_cached(SampledSource(source, f, cache, i)), chunks=(1, BLOCK_YX, BLOCK_YX),
                          name='luxendo-preview-' + uuid.uuid4().hex,
                          asarray=False, fancy=False,
                          meta=np.empty((0, 0, 0), dtype=source.dtype))
            for i, f in enumerate(factors)]


def _cached(sampled):
    # Samples served while the averaged cache builds would go stale in memory.
    return CachedSource(sampled, lambda: sampled.cache is None or sampled.cache.ready)


_FOUND: dict = {}


def preview_caches(array):
    """The preview caches behind a (possibly stacked) display level, if any."""
    # Walking the graph materialises every block key, so it is done once per
    # array (Dask names are unique per array content).
    found = _FOUND.get(array.name)
    if found is None:
        found = _FOUND[array.name] = _find_caches(array)
        if len(_FOUND) > 1024:
            _FOUND.pop(next(iter(_FOUND)))
    return found


def forget_caches() -> None:
    _FOUND.clear()


def _find_caches(array):
    graph = array.__dask_graph__()
    found = []
    for name, layer in getattr(graph, 'layers', {}).items():
        if not str(name).startswith('original-luxendo-preview-'):
            continue
        for value in dict(layer).values():
            value = value.inner if isinstance(value, CachedSource) else value
            if isinstance(value, SampledSource) and value.cache is not None:
                found.append(value.cache)
            elif isinstance(value, SampledSource):
                found.append(None)
    return found


def level_is_cheap(array):
    """True if reading a plane of *array* does not fall back to full-resolution I/O.

    Native pyramid levels are always cheap; preview levels are cheap once every
    cache behind them has finished building.
    """
    caches = preview_caches(array)
    return all(c is not None and c.ready for c in caches)
