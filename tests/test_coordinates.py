"""Switching layers between sample and camera coordinates."""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from napari_luxendo import read_luxendo
from napari_luxendo._coordinates import CAMERA, SAMPLE, set_coordinates
from napari_luxendo._reader import camera_affine

from conftest import TILE, VX, lux_metadata, make_volume

napari = pytest.importorskip("napari")

from napari.components import ViewerModel  # noqa: E402

S = 1 / np.sqrt(2)


def _rotated_view(tmp_path):
    """A raw view from an objective at 45 deg, like a dual-view LCS-SPIM stack."""
    path = tmp_path / "Cam_left_00000.lux.h5"
    affine = [{
        # (x, y, z) voxel -> sample: x and z both run diagonally in the x-z plane.
        "matrix": [[VX[2] * S, 0, VX[0] * S], [0, VX[1], 0], [-VX[2] * S, 0, VX[0] * S]],
        "translation": [100.0, 200.0, 300.0],
    }]
    with h5py.File(path, "w") as f:
        f.create_dataset("Data", data=make_volume(0, TILE))
        f.create_dataset("metadata", data=lux_metadata(0, 0, "1", affine))
    return path


def _open(viewer, path, **kwargs):
    for data, meta, kind in read_luxendo(str(path), **kwargs):
        viewer.add_layer(napari.layers.Layer.create(data, meta, kind))
    return viewer.layers[-1]


def test_camera_affine_keeps_only_voxel_size():
    flip = np.diag([-5.0, 2.925, 2.925, 1.0])
    flip[:3, 3] = [9500, -10, 20]
    np.testing.assert_allclose(camera_affine(flip, None), np.diag([5.0, 2.925, 2.925, 1.0]))
    np.testing.assert_allclose(camera_affine(None, (2.0, 0.5, 0.4)), np.diag([2.0, 0.5, 0.4, 1.0]))
    np.testing.assert_allclose(camera_affine(None, None), np.eye(4))


def test_toggle_rotated_view(tmp_path):
    viewer = ViewerModel()
    layer = _open(viewer, _rotated_view(tmp_path))
    sample = layer.affine.affine_matrix.copy()
    assert abs(sample[0, 2]) > 1  # placed rotated in sample space

    set_coordinates(layer, CAMERA)
    np.testing.assert_allclose(layer.affine.affine_matrix, np.diag([*VX, 1.0]), atol=1e-12)
    np.testing.assert_allclose(layer.scale, 1.0)
    assert layer.metadata["coordinates"] == CAMERA
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "non-orthogonal slicing" warning
        layer.refresh()

    set_coordinates(layer, SAMPLE)
    np.testing.assert_allclose(layer.affine.affine_matrix, sample)


def test_toggle_layer_read_in_voxel_mode(tmp_path):
    """A layer first placed by scale ends up in the same places as one placed by affine."""
    viewer = ViewerModel()
    path = _rotated_view(tmp_path)
    layer = _open(viewer, path, transform="voxel")
    np.testing.assert_allclose(layer.scale, VX)

    set_coordinates(layer, SAMPLE)
    reference = _open(viewer, path)
    np.testing.assert_allclose(layer.data_to_world((1, 2, 3)), reference.data_to_world((1, 2, 3)))

    set_coordinates(layer, CAMERA)
    np.testing.assert_allclose(layer.data_to_world((1, 2, 3)), np.multiply((1, 2, 3), VX))


def test_toggle_tiled_time_series(tiled_experiment):
    viewer = ViewerModel()
    mosaic = _open(viewer, tiled_experiment / "main_raw.lux.h5", views="raw")
    assert mosaic.ndim == 4
    set_coordinates(mosaic, CAMERA)
    np.testing.assert_allclose(mosaic.affine.affine_matrix, np.diag([1.0, *VX, 1.0]))


def test_layers_without_sample_transform_stay_on_camera(lux_dir):
    viewer = ViewerModel()
    layer = _open(viewer, lux_dir / "dataset.ims")
    before = layer.data_to_world((0, 1, 2, 3))
    set_coordinates(layer, SAMPLE)
    np.testing.assert_allclose(layer.data_to_world((0, 1, 2, 3)), before)


def test_other_layers_are_left_alone():
    viewer = ViewerModel()
    layer = viewer.add_image(np.zeros((4, 4, 4)), scale=(2, 1, 1))
    set_coordinates(layer, CAMERA)
    np.testing.assert_allclose(layer.scale, (2, 1, 1))
