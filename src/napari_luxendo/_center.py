"""Draw coarse pyramid levels centred on the full-resolution pixels they cover.

napari places pixel ``j`` of a level downsampled by ``s`` at full-resolution
coordinate ``j * s``, so the first coarse pixel and the first full-resolution
pixel share a centre and the coarse image hangs up and left by ``(s - 1) / 2``
pixels. Every coarse voxel here stands for the block ``[j * s, (j + 1) * s)``
(averaged in the preview cache, sampled at its middle otherwise), whose centre
is ``j * s + (s - 1) / 2``. This module adds that half-block offset to the
tile-to-data transform napari builds for each slice, so a coarse level sits
exactly on the full-resolution bounds, and picks the coarse plane whose block
contains the current full-resolution plane.

It hooks one layer instance at a time (napari 0.5 keeps slicing state on the
layer, napari 0.9 on ``layer._slicing_state``) and leaves level 0 untouched.
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

logger = logging.getLogger(__name__)


def center_levels(layer) -> None:
    """Make *layer* draw its coarse levels centred over full resolution (idempotent)."""
    if not getattr(layer, 'multiscale', False):
        return
    _hook(layer)
    events = getattr(getattr(layer, 'events', None), 'data', None)
    if events is not None and not getattr(layer, '_luxendo_center_data_hook', False):
        # Replacing the data can replace the slicing state (napari 0.9).
        events.connect(lambda event=None: _hook(layer))
        layer._luxendo_center_data_hook = True


def _hook(layer) -> None:
    state = getattr(layer, '_slicing_state', layer)
    for name, wrap in (('_update_slice_response', _wrap_response),
                       ('_make_slice_request_internal', _wrap_request)):
        method = getattr(state, name, None)
        if method is None or getattr(method, '_luxendo_centered', False):
            continue
        wrapped = wrap(layer, method)
        wrapped._luxendo_centered = True
        setattr(state, name, wrapped)


def centered_tile_to_data(tile_to_data, displayed):
    """*tile_to_data* with displayed axes shifted by half a block, or None if level 0."""
    scale = np.asarray(tile_to_data.scale, dtype=float)
    displayed = [d for d in displayed if scale[d] != 1]
    if not displayed:
        return None
    translate = np.array(tile_to_data.translate, dtype=float)
    translate[displayed] += (scale[displayed] - 1) / 2
    return type(tile_to_data)(scale=scale, translate=translate, ndim=len(scale),
                              name=tile_to_data.name)


def _wrap_response(layer, method):
    def update_slice_response(response, *args, **kwargs):
        try:
            response = _centered_response(response)
        except Exception as exc:  # never break slicing
            logger.debug('Centering coarse level: %s', exc)
        return method(response, *args, **kwargs)

    return update_slice_response


def _centered_response(response):
    tile_to_data = getattr(response, 'tile_to_data', None)
    slice_input = getattr(response, 'slice_input', None)
    if tile_to_data is None or slice_input is None or slice_input.ndisplay != 2:
        return response
    centered = centered_tile_to_data(tile_to_data, slice_input.displayed)
    if centered is None:
        return response
    return dataclasses.replace(response, tile_to_data=centered)


def _wrap_request(layer, method):
    def make_slice_request_internal(*args, **kwargs):
        request = method(*args, **kwargs)
        try:
            _center_planes(request)
        except Exception as exc:  # never break slicing
            logger.debug('Centering coarse plane: %s', exc)
        return request

    return make_slice_request_internal


def _center_planes(request) -> None:
    """Pick, on coarse levels, the plane whose block holds the current plane.

    napari rounds ``z / s``, which shows block ``j + 1`` from the middle of
    block ``j`` on. Shifting the point by ``(s - 1) / (2 s)`` coarse planes
    rounds to the block instead.
    """
    at_level = getattr(request, '_thick_slice_at_level', None)
    if at_level is None or not getattr(request, 'multiscale', False):
        return
    factors = np.asarray(request.downsample_factors, dtype=float)
    not_displayed = list(request.slice_input.not_displayed)

    def thick_slice_at_level(level):
        data_slice = at_level(level)
        f = factors[level][not_displayed]
        if level == 0 or np.all(f == 1):
            return data_slice
        arr = data_slice.as_array()
        shapes = np.asarray(request.level_shapes[level])
        arr[0, not_displayed] -= (f - 1) / (2 * f)
        arr[0] = np.clip(arr[0], 0, shapes - 1)
        return type(data_slice).from_array(arr)

    object.__setattr__(request, '_thick_slice_at_level', thick_slice_at_level)
