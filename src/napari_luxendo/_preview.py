"""On-demand display pyramids for large raw acquisitions.

Level zero is unchanged. Until a level is cached, it is a nearest-neighbour
sample of full resolution, which is free to set up but costs as much disk I/O
per plane as full resolution (sampled rows still touch every page). The first
time a preview is shown, its levels are averaged once in a background thread
into a cache file (see :mod:`._preview_cache`), and reads switch to it plane
by plane as it fills. No sidecar files are written next to the data.
"""
from __future__ import annotations

import itertools
import os
import uuid

import dask.array as da
import numpy as np

from ._fastio import note_foreground_read
from ._preview_cache import PreviewCache


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

    def prepare_background(self):
        """Ready the reader; True if reads then make no h5py calls (see ._fastio)."""
        reader = self.vol.reader(0)
        return reader is not None and reader.prepare()


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
        note_foreground_read()
        if self.cache is not None:
            cached = self.cache.read(self.level, key)
            if cached is not None:
                return cached
        # Sample each block at its middle, the centre the display puts it at
        # (see ._center). A clipped last block is sampled at its last pixel.
        f, mid = self.factor, self.factor // 2
        parts = []
        for s, n, full in zip(key, self.shape, self.source.shape):
            start, stop, step = s.indices(n)
            count = len(range(start, stop, step))
            first = start * f + mid
            inside = max(0, min(count, -(-(full - first) // (step * f))))
            axis = [(slice(first, first + max(inside - 1, 0) * step * f + 1, step * f),
                     slice(0, inside))] if inside else []
            if inside < count:
                axis.append((slice(full - 1, full), slice(inside, count)))
            parts.append(axis)
        out = np.empty(tuple(len(range(*s.indices(n))) for s, n in zip(key, self.shape)),
                       dtype=self.dtype)
        for combo in itertools.product(*parts):
            out[tuple(d for _, d in combo)] = self.source[tuple(src for src, _ in combo)]
        return out


def preview_levels(source):
    factors = preview_factors(source.shape)
    cache = PreviewCache.for_source(source, factors) if factors else None
    return [da.from_array(SampledSource(source, f, cache, i), chunks=(1, 512, 512),
                          name='luxendo-preview-' + uuid.uuid4().hex,
                          asarray=False, fancy=False,
                          meta=np.empty((0, 0, 0), dtype=source.dtype))
            for i, f in enumerate(factors)]


def preview_caches(array):
    """The preview caches behind a (possibly stacked) display level, if any."""
    graph = array.__dask_graph__()
    found = []
    for name, layer in getattr(graph, 'layers', {}).items():
        if not str(name).startswith('original-luxendo-preview-'):
            continue
        for value in dict(layer).values():
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
