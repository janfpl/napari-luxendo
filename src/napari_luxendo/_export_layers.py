"""Turn napari Luxendo layers into :class:`~napari_luxendo._export.ExportSource`.

The export ROI is a box in world coordinates (sample-space micrometres, or the
camera grid when the coordinates toggle is on). Each layer crops the bounding
box of that region in its own voxel grid, using the layer's current placement,
so the crop matches what the viewer shows. For layers placed with flips and
translations only (tile grids) that crop is exact; for a rotated view it is the
voxel bounding box of the rotated region.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Optional, Sequence

import numpy as np

from ._coordinates import luxendo_layers
from ._export import ExportSource, Roi

# (z_min, z_max, y_min, y_max, x_min, x_max) in world units.
WorldBox = Sequence[float]

# Tolerance for voxel centres that land on a box face after rounding errors.
_EPS = 1e-6


def exportable_layers(viewer: Any) -> list[Any]:
    """Luxendo image layers in *viewer* that can be exported."""
    return [layer for layer in luxendo_layers(viewer) if layer.ndim in (3, 4)]


def full_resolution(layer: Any) -> Any:
    """The layer's full-resolution data as ``(T, Z, Y, X)``."""
    data = layer.data[0] if layer.multiscale else layer.data
    return data if data.ndim == 4 else data[None]


def layer_timepoints(layer: Any) -> list[int]:
    n = full_resolution(layer).shape[0]
    tps = list(layer.metadata.get("timepoints") or [])
    return [int(t) for t in tps] if len(tps) == n else list(range(n))


def _spatial(affine: Any) -> Optional[np.ndarray]:
    if affine is None:
        return None
    a = np.asarray(affine, dtype=float)
    return a[1:, 1:] if a.shape == (5, 5) else a


def _to_world(layer: Any, zyx: Sequence[float]) -> np.ndarray:
    point = ([0.0] if layer.ndim == 4 else []) + [float(v) for v in zyx]
    return np.asarray(layer.data_to_world(point), dtype=float)[-3:]


def _to_data(layer: Any, zyx: Sequence[float]) -> np.ndarray:
    point = ([0.0] if layer.ndim == 4 else []) + [float(v) for v in zyx]
    return np.asarray(layer.world_to_data(point), dtype=float)[-3:]


def world_extent(layer: Any) -> tuple[np.ndarray, np.ndarray]:
    """World ``(z, y, x)`` min and max of the layer's voxel centres."""
    nz, ny, nx = full_resolution(layer).shape[-3:]
    corners = np.array([
        _to_world(layer, c)
        for c in itertools.product((0, nz - 1), (0, ny - 1), (0, nx - 1))
    ])
    return corners.min(axis=0), corners.max(axis=0)


def voxel_roi(layer: Any, box: WorldBox) -> Optional[Roi]:
    """Voxel crop of *layer* covering the world *box*; None if they don't overlap.

    Keeps every voxel whose centre lies inside the box (for a rotated layer,
    inside the voxel-space bounding box of the world box).
    """
    zr, yr, xr = (box[0], box[1]), (box[2], box[3]), (box[4], box[5])
    data = np.array([_to_data(layer, c) for c in itertools.product(zr, yr, xr)])
    lo, hi = data.min(axis=0), data.max(axis=0)
    roi: list[int] = []
    for size, a, b in zip(full_resolution(layer).shape[-3:], lo, hi):
        start = max(0, math.ceil(a - _EPS))
        stop = min(int(size), math.floor(b + _EPS) + 1)
        if stop <= start:
            return None
        roi += [start, stop]
    return tuple(roi)  # type: ignore[return-value]


def _color(layer: Any) -> Optional[tuple[float, float, float]]:
    try:
        rgb = tuple(float(v) for v in np.asarray(layer.colormap.colors)[-1][:3])
    except (AttributeError, IndexError, TypeError):
        return None
    return rgb if any(rgb) else None  # type: ignore[return-value]


def _contrast_limits(layer: Any) -> Optional[tuple[float, float]]:
    try:
        lo, hi = (float(v) for v in layer.contrast_limits)
    except (AttributeError, TypeError, ValueError):
        return None
    return (lo, hi)


def export_source(layer: Any, roi: Optional[Roi] = None) -> ExportSource:
    """An :class:`ExportSource` for a Luxendo *layer*, cropped to *roi* (voxels)."""
    md = layer.metadata
    placements = md.get("placements") or {}
    voxel = md.get("voxel_size_um")
    # Without affine_to_sample the layer is placed by voxel size (the camera
    # placement); writing that keeps a crop's offset.
    affine = placements.get("sample")
    if affine is None:
        affine = placements.get("camera")
    return ExportSource(
        name=layer.name,
        data=full_resolution(layer),
        timepoints=layer_timepoints(layer),
        roi=roi,
        pyramid_levels=[n for n in (md.get("pyramid_levels") or []) if n != "Data"],
        affine_zyx=_spatial(affine),
        voxel_size_um=tuple(float(v) for v in voxel) if voxel else None,
        metadata=md.get("luxendo") or {},
        color=_color(layer),
        contrast_limits=_contrast_limits(layer),
        fused_views=max(1, len(md.get("views") or [])),
        source_files=list(md.get("files") or []),
    )
