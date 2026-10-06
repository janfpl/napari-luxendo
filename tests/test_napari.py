"""Checks against napari itself; skipped when napari is not installed."""

from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari.layers import Layer  # noqa: E402

from napari_luxendo import napari_get_reader  # noqa: E402

from conftest import SHAPE, make_volume, write_lux  # noqa: E402


def test_plugin_is_discovered_for_lux_files(tmp_path):
    import npe2

    path = str(write_lux(tmp_path / "a.lux.h5", make_volume(0)))
    pm = npe2.PluginManager.instance()
    pm.discover()
    readers = pm.iter_compatible_readers([path])
    assert "napari-luxendo" in {r.plugin_name for r in readers}


def test_layers_build_in_napari(lux_dir):
    path = str(lux_dir / "dataset.ims")
    layers = [Layer.create(*ld) for ld in napari_get_reader(path)(path)]

    assert [layer.name for layer in layers] == ["GFP", "mCherry"]
    gfp = layers[0]
    assert gfp.multiscale and gfp.ndim == 4
    np.testing.assert_allclose(gfp.scale, [1.0, 2.0, 0.5, 0.4])
    np.testing.assert_allclose(gfp.colormap.colors[-1], [0, 1, 0, 1])
    assert gfp.blending == "additive"
    assert gfp.data.shapes[0] == (2, *SHAPE)


def test_tiled_time_series_builds_in_napari(tiled_experiment):
    path = str(tiled_experiment / "main_raw.lux.h5")
    layers = [Layer.create(*ld) for ld in napari_get_reader(path)(path)]
    assert len(layers) == 2
    mosaic = layers[0]
    assert mosaic.ndim == 4 and mosaic.multiscale
    assert mosaic.data.shapes[0][0] == 2  # two timepoints
    np.testing.assert_allclose(np.diag(mosaic.affine.affine_matrix)[1:4], [-5.0, 2.925, 2.925])
    # A voxel maps to the same sample position through napari as through the metadata.
    np.testing.assert_allclose(
        mosaic.data_to_world((0, 0, 0, 0))[1:],
        mosaic.affine.affine_matrix[1:4, 4],
    )
