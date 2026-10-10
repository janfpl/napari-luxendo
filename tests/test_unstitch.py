"""Unstitching a mosaic and per-tile Z planes on a viewer model (no Qt needed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari")

from napari.components import ViewerModel  # noqa: E402
from napari.layers import Layer  # noqa: E402

from napari_luxendo import _tiles, read_luxendo  # noqa: E402
from napari_luxendo._reader import read_tiles  # noqa: E402

from conftest import GRID, TILE, ground_truth  # noqa: E402

N_TILES = GRID[0] * GRID[1]


@pytest.fixture
def viewer(tiled_experiment):
    viewer = ViewerModel()
    for ld in read_luxendo(str(tiled_experiment / "main_raw.lux.h5")):
        viewer.add_layer(Layer.create(*ld))
    return viewer


def test_read_tiles_gives_one_layer_per_tile(tiled_experiment):
    mosaic = read_luxendo(str(tiled_experiment / "main_raw.lux.h5"))[1]
    tiles = read_tiles(mosaic[1]["metadata"])
    assert len(tiles) == N_TILES
    data, kwargs, kind = tiles[0]
    assert kind == "image" and data[0].shape == (2, *TILE)
    np.testing.assert_array_equal(np.asarray(data[0][1]), ground_truth(1, 1)[:, :TILE[1], :TILE[2]])
    assert "tiles" not in kwargs["metadata"]


def test_unstitch_then_stitch_restores_the_mosaic(viewer):
    mosaics = list(viewer.layers)
    for m in mosaics:
        _tiles.unstitch(viewer, m)
    assert len(viewer.layers) == 2 * N_TILES
    assert all(t.colormap.name == mosaics[0].colormap.name for t in list(viewer.layers)[:N_TILES])
    for m in _tiles.stitched_mosaics(viewer):
        _tiles.stitch(viewer, m)
    assert list(viewer.layers) == mosaics


def test_planes_move_together_or_alone(viewer):
    for m in list(viewer.layers):
        _tiles.unstitch(viewer, m)
    tiles = list(viewer.layers)
    _tiles.move_all_to_plane(viewer, tiles[0], 6)
    assert {_tiles.shown_plane(t, viewer.dims.point) for t in tiles} == {6}

    _tiles.show_plane(tiles[2], 1, viewer.dims.point)
    planes = [_tiles.shown_plane(t, viewer.dims.point) for t in tiles]
    assert planes[2] == 1 and set(planes[:2] + planes[3:]) == {6}
    assert tiles[2].metadata["plane_offset"] == -5

    # Moving the viewer's Z keeps the tile's own offset.
    _tiles.move_all_to_plane(viewer, tiles[0], 9)
    assert _tiles.shown_plane(tiles[2], viewer.dims.point) == 4
