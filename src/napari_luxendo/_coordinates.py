"""Switching Luxendo layers between sample and camera coordinates.

*Sample* places each layer with ``affine_to_sample``, so views of one sample
overlap where they image the same spot. Raw views from angled objectives are
then rotated in the viewer, and napari can only show such data in 2D as
straight planes of the stack (with a "non-orthogonal slicing" warning).

*Camera* places each layer by voxel size only: the raw grid as the camera
recorded it. 2D slices are then the planes the camera took, but views no
longer line up with each other.

The reader stores both placements in ``layer.metadata["placements"]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SAMPLE = "sample"
CAMERA = "camera"


def luxendo_layers(viewer: Any) -> list[Any]:
    """Layers opened by this plugin that carry both placements."""
    return [layer for layer in viewer.layers if _placements(layer) is not None]


def set_coordinates(layer: Any, mode: str) -> None:
    """Place *layer* in ``"sample"`` or ``"camera"`` coordinates.

    Layers without a sample transform keep the camera placement either way.
    """
    if mode not in (SAMPLE, CAMERA):
        raise ValueError(f"mode must be {SAMPLE!r} or {CAMERA!r}, not {mode!r}")
    placements = _placements(layer)
    if placements is None:
        return
    target = placements.get(mode)
    if target is None:
        target = placements[CAMERA]
    ndim = layer.ndim
    # The placement lives in the affine alone, so the result does not depend
    # on how the reader first placed the layer (by affine or by scale).
    layer.scale = np.ones(ndim)
    layer.translate = np.zeros(ndim)
    layer.affine = np.asarray(target, dtype=float)
    layer.metadata["coordinates"] = mode


def _placements(layer: Any) -> dict[str, Any] | None:
    placements = getattr(layer, "metadata", {}).get("placements")
    if isinstance(placements, dict) and placements.get(CAMERA) is not None:
        return placements
    return None
