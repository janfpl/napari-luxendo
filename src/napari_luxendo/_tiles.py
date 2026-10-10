"""Unstitching mosaic layers into one layer per tile, and per-tile Z planes.

*Unstitch* replaces a mosaic layer with a layer per tile, read again from the
same files and placed the same way, with the mosaic's colormap and contrast.
*Stitch* puts the mosaic layer back in place of its tiles.

napari has one Z slider for the whole viewer, so layers cannot each sit at
their own Z position. A tile is given its own Z plane by shifting its voxel
grid along Z (``layer.translate``) by a whole number of planes: at the
viewer's current Z position the tile then shows another of its planes. The
shift is kept in ``layer.metadata["plane_offset"]`` (in planes). In 3D a
shifted tile is drawn shifted by the same amount.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Display settings a tile takes over from its mosaic, and gives back.
_SHARED = ("colormap", "contrast_limits", "gamma", "opacity", "blending", "visible")


def is_mosaic(layer: Any) -> bool:
    """True for a mosaic layer the reader built (it knows its tiles)."""
    return isinstance(getattr(layer, "metadata", {}).get("tiles"), dict)


def is_tile(layer: Any) -> bool:
    """True for a tile layer made by :func:`unstitch`."""
    return getattr(layer, "metadata", {}).get("mosaic_layer") is not None


def mosaic_layers(viewer: Any) -> list[Any]:
    return [layer for layer in viewer.layers if is_mosaic(layer)]


def tile_layers(viewer: Any) -> list[Any]:
    return [layer for layer in viewer.layers if is_tile(layer)]


def unstitch(viewer: Any, mosaic: Any) -> list[Any]:
    """Replace *mosaic* in *viewer* by one layer per tile; return the tile layers."""
    from napari.layers import Layer

    from ._coordinates import set_coordinates
    from ._reader import read_tiles

    tiles = []
    for data, kwargs, kind in read_tiles(mosaic.metadata):
        layer = Layer.create(data, kwargs, kind)
        for attr in _SHARED:
            setattr(layer, attr, getattr(mosaic, attr))
        layer.metadata["mosaic_layer"] = mosaic
        layer.metadata["plane_offset"] = 0
        if mosaic.metadata.get("coordinates"):
            set_coordinates(layer, mosaic.metadata["coordinates"])
        tiles.append(layer)
    index = viewer.layers.index(mosaic)
    viewer.layers.remove(mosaic)
    for k, layer in enumerate(tiles):
        viewer.layers.insert(index + k, layer)
    return tiles


def stitch(viewer: Any, mosaic: Any) -> Optional[Any]:
    """Put *mosaic* back in place of its tile layers; return it, or None if it has none."""
    from ._coordinates import set_coordinates

    tiles = [layer for layer in viewer.layers if layer.metadata.get("mosaic_layer") is mosaic]
    if not tiles:
        return None
    first = tiles[0]
    # Contrast and colour changed on the tiles carry over to the mosaic.
    for attr in ("colormap", "contrast_limits", "gamma", "opacity", "visible"):
        setattr(mosaic, attr, getattr(first, attr))
    if first.metadata.get("coordinates"):
        set_coordinates(mosaic, first.metadata["coordinates"])
    index = viewer.layers.index(first)
    for layer in tiles:
        viewer.layers.remove(layer)
    viewer.layers.insert(index, mosaic)
    return mosaic


def stitched_mosaics(viewer: Any) -> list[Any]:
    """The mosaics whose tiles are in *viewer*, in layer order."""
    out: list[Any] = []
    for layer in tile_layers(viewer):
        mosaic = layer.metadata["mosaic_layer"]
        if all(m is not mosaic for m in out):
            out.append(mosaic)
    return out


# --------------------------------------------------------------------------- #
# Per-tile Z planes
# --------------------------------------------------------------------------- #


def z_axis(layer: Any) -> int:
    return layer.ndim - 3


def plane_count(layer: Any) -> int:
    shape = layer.data[0].shape if layer.multiscale else layer.data.shape
    return int(shape[z_axis(layer)])


def _layer_point(layer: Any, world_point: Any) -> np.ndarray:
    """The world point restricted to *layer*'s dimensions (the last ones)."""
    return np.asarray(world_point, dtype=float)[-layer.ndim:]


def shown_plane(layer: Any, world_point: Any) -> int:
    """The Z plane of *layer* shown at the viewer's *world_point*."""
    data = np.asarray(layer.world_to_data(_layer_point(layer, world_point)), dtype=float)
    plane = int(np.round(data[z_axis(layer)]))
    return min(max(plane, 0), plane_count(layer) - 1)


def set_plane_offset(layer: Any, offset: int) -> None:
    """Shift *layer* so it shows its plane ``z + offset`` where it showed plane ``z``."""
    translate = np.array(layer.translate, dtype=float)
    translate[z_axis(layer)] = -float(offset) * float(layer.scale[z_axis(layer)])
    layer.translate = translate
    layer.metadata["plane_offset"] = int(offset)


def show_plane(layer: Any, plane: int, world_point: Any) -> None:
    """Make *layer* alone show *plane* at *world_point*, by changing its own offset."""
    data = np.asarray(layer.world_to_data(_layer_point(layer, world_point)), dtype=float)
    current = data[z_axis(layer)]
    offset = int(layer.metadata.get("plane_offset", 0))
    set_plane_offset(layer, offset + int(np.round(plane - current)))


def world_point_for_plane(layer: Any, plane: int, world_point: Any) -> np.ndarray:
    """The viewer world point at which *layer* shows *plane*, moving along Z only.

    Only the world axis that the layer's Z axis mostly maps to changes; for an
    angled view the other axes keep their positions.
    """
    point = np.asarray(world_point, dtype=float).copy()
    own = _layer_point(layer, point)
    data = np.asarray(layer.world_to_data(own), dtype=float)
    data[z_axis(layer)] = plane
    target = np.asarray(layer.data_to_world(data), dtype=float)
    step = np.asarray(layer.data_to_world(data + _unit(layer)), dtype=float) - target
    axis = int(np.argmax(np.abs(step)))
    offset = len(point) - layer.ndim
    point[offset + axis] = target[axis]
    return point


def _unit(layer: Any) -> np.ndarray:
    unit = np.zeros(layer.ndim)
    unit[z_axis(layer)] = 1.0
    return unit


def move_all_to_plane(viewer: Any, layer: Any, plane: int) -> None:
    """Move the viewer's Z so *layer* shows *plane*; every other tile moves with it."""
    point = world_point_for_plane(layer, plane, viewer.dims.point)
    current = np.asarray(viewer.dims.point, dtype=float)
    for axis in np.flatnonzero(~np.isclose(point, current)):
        viewer.dims.set_point(int(axis), float(point[axis]))
