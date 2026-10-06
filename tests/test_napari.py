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
