"""On-demand nearest-neighbour display pyramids for large raw acquisitions.

These are sampled views, not averaged scientific resampling. Level zero is
unchanged. No preprocessing, sidecar files or full-volume reads are needed.
"""
from __future__ import annotations

import os
import uuid

import dask.array as da
import numpy as np


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


class SampledSource:
    ndim = 3

    def __init__(self, source, factor):
        self.source, self.factor = source, factor
        self.shape = tuple(-(-n // factor) for n in source.shape)
        self.dtype = source.dtype

    def __getitem__(self, key):
        mapped = []
        for s, n, full in zip(key, self.shape, self.source.shape):
            start, stop, step = s.indices(n)
            mapped.append(slice(start*self.factor, min(stop*self.factor, full), step*self.factor))
        return self.source[tuple(mapped)]


def preview_levels(source):
    return [da.from_array(SampledSource(source, f), chunks=(1, 512, 512),
                          name='luxendo-preview-' + uuid.uuid4().hex,
                          asarray=False, fancy=False,
                          meta=np.empty((0, 0, 0), dtype=source.dtype))
            for f in preview_factors(source.shape)]
