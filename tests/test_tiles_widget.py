"""Unstitching mosaics and per-tile Z planes; skipped without napari and pytest-qt."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("pytestqt")
pytest.importorskip("qtpy.QtWidgets")

from napari_luxendo import _tiles  # noqa: E402
from napari_luxendo._coordinates import CAMERA, set_coordinates  # noqa: E402
from napari_luxendo._tiles_widget import TilesWidget  # noqa: E402

from conftest import GRID, STEP, TILE, ground_truth  # noqa: E402

N_TILES = GRID[0] * GRID[1]


@pytest.fixture
def viewer(make_napari_viewer, tiled_experiment):
    viewer = make_napari_viewer()
    viewer.open(str(tiled_experiment / "main_raw.lux.h5"), plugin="napari-luxendo")
    return viewer


def _z_world_axis(viewer):
    return viewer.dims.ndim - 3


def test_unstitch_and_stitch_back(viewer, qtbot):
    widget = TilesWidget(viewer)
    qtbot.addWidget(widget)
    mosaics = list(viewer.layers)
    assert len(mosaics) == 2 and all(_tiles.is_mosaic(m) for m in mosaics)
    mosaics[0].contrast_limits = (10, 2000)
    assert not widget.sync.isEnabled()

    widget.unstitch.setChecked(True)
    tiles = list(viewer.layers)
    assert len(tiles) == 2 * N_TILES and all(_tiles.is_tile(t) for t in tiles)
    assert widget.sync.isEnabled() and len(widget._rows) == 2 * N_TILES
    first = tiles[0]
    assert first.metadata["mosaic_layer"] is mosaics[0]
    assert tuple(first.contrast_limits) == (10, 2000)
    assert first.data[0].shape == (2, *TILE)  # two timepoints, one tile

    # A tile shows exactly its part of the mosaic, at the same world position.
    truth = ground_truth(0, 0)
    np.testing.assert_array_equal(np.asarray(first.data[0][0]), truth[:, :TILE[1], :TILE[2]])
    np.testing.assert_allclose(first.data_to_world((0, 0, 0, 0)), mosaics[0].data_to_world((0, 0, 0, 0)))
    last = tiles[N_TILES - 1]
    oy, ox = STEP[0] * (GRID[0] - 1), STEP[1] * (GRID[1] - 1)
    np.testing.assert_allclose(last.data_to_world((0, 0, 0, 0)), mosaics[0].data_to_world((0, 0, oy, ox)))

    first.contrast_limits = (5, 500)
    widget.unstitch.setChecked(False)
    assert list(viewer.layers) == mosaics
    assert tuple(mosaics[0].contrast_limits) == (5, 500)
    assert not widget._rows


def test_mosaics_opened_later_are_unstitched(viewer, qtbot, tiled_experiment):
    widget = TilesWidget(viewer)
    qtbot.addWidget(widget)
    widget.unstitch.setChecked(True)
    viewer.open(str(tiled_experiment / "main_raw.lux.h5"), plugin="napari-luxendo")
    qtbot.waitUntil(lambda: len(viewer.layers) == 4 * N_TILES)
    assert not _tiles.mosaic_layers(viewer)


def test_sync_planes_moves_every_tile(viewer, qtbot):
    widget = TilesWidget(viewer)
    qtbot.addWidget(widget)
    widget.unstitch.setChecked(True)
    assert widget.sync.isChecked()
    layer, slider, _ = widget._rows[0]
    slider.setValue(7)
    planes = [_tiles.shown_plane(t, viewer.dims.point) for t, _, _ in widget._rows]
    assert planes == [7] * len(planes)
    assert all(s.value() == 7 for _, s, _ in widget._rows)
    assert all(t.metadata["plane_offset"] == 0 for t, _, _ in widget._rows)


def test_unsynced_tile_moves_alone(viewer, qtbot):
    widget = TilesWidget(viewer)
    qtbot.addWidget(widget)
    widget.unstitch.setChecked(True)
    widget._rows[0][1].setValue(4)  # synced: all tiles at plane 4
    widget.sync.setChecked(False)
    layer, slider, spin = widget._rows[1]
    spin.setValue(9)
    planes = [_tiles.shown_plane(t, viewer.dims.point) for t, _, _ in widget._rows]
    assert planes[1] == 9 and planes[0] == 4 and planes[2:] == [4] * (len(planes) - 2)
    assert slider.value() == 9
    # napari's own Z slider moves every tile, keeping the offset.
    axis = _z_world_axis(viewer)
    viewer.dims.set_current_step(axis, viewer.dims.current_step[axis] + 1)
    moved = [_tiles.shown_plane(t, viewer.dims.point) for t, _, _ in widget._rows]
    assert moved[0] != 4 and moved[1] - moved[0] == 5
    assert [s.value() for _, s, _ in widget._rows[:2]] == moved[:2]

    # Switching coordinates keeps the tile's own plane.
    set_coordinates(layer, CAMERA)
    assert layer.metadata["plane_offset"] == 5
    assert float(layer.translate[layer.ndim - 3]) == -5.0
