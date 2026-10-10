"""Averaged preview cache and coarse-while-scrolling display."""
import h5py
import numpy as np
import pytest

from napari_luxendo import close_all, read_luxendo
from napari_luxendo._preview_cache import block_means, wait_for_builds
from conftest import make_volume, write_lux


def naive_means(a, f):
    out = np.empty(tuple(-(-n // f) for n in a.shape))
    for z in range(out.shape[0]):
        for y in range(out.shape[1]):
            for x in range(out.shape[2]):
                out[z, y, x] = a[z*f:(z+1)*f, y*f:(y+1)*f, x*f:(x+1)*f].mean()
    return np.clip(np.rint(out), 0, 65535).astype(a.dtype)


def test_block_means_match_naive_on_ragged_edges():
    a = make_volume(7, shape=(9, 13, 17))
    for got, f in zip(block_means(a, [2, 4, 8]), [2, 4, 8]):
        np.testing.assert_array_equal(got, naive_means(a, f))


@pytest.fixture
def cache_on(monkeypatch):
    import napari_luxendo._preview as preview
    monkeypatch.setenv("NAPARI_LUXENDO_PYRAMID_CACHE", "1")
    monkeypatch.setattr(preview, "preview_factors", lambda shape: [2, 4])


def test_single_volume_preview_cache_builds_and_is_reused(tmp_path, cache_on):
    truth = make_volume(3)
    path = write_lux(tmp_path / "single.lux.h5", truth, pyramids=())
    [(levels, _, _)] = read_luxendo(str(path))
    # Before the build, previews are exact samples; the first read queues the build.
    np.testing.assert_array_equal(levels[-1][0, :2, :2].compute(), truth[0, :8:4, :8:4])
    wait_for_builds(30)
    np.testing.assert_array_equal(levels[1].compute(), naive_means(truth, 2))
    np.testing.assert_array_equal(levels[2][1].compute(), naive_means(truth, 4)[1])
    close_all()
    # A new session finds the finished cache file without building again.
    [(levels, _, _)] = read_luxendo(str(path))
    from napari_luxendo._preview import level_is_cheap
    assert level_is_cheap(levels[2]) and level_is_cheap(levels[0])
    np.testing.assert_array_equal(levels[2].compute(), naive_means(truth, 4))


def test_changed_source_is_not_served_from_old_cache(tmp_path, cache_on):
    path = write_lux(tmp_path / "single.lux.h5", make_volume(3), pyramids=())
    [(levels, _, _)] = read_luxendo(str(path))
    levels[1][0].compute()
    wait_for_builds(30)
    close_all()
    other = make_volume(4)
    write_lux(path, other, pyramids=())
    [(levels, _, _)] = read_luxendo(str(path))
    # The new data gets its own cache (built in the background on first read),
    # never the averages of the old file.
    levels[1][0].compute()
    wait_for_builds(30)
    np.testing.assert_array_equal(levels[1].compute(), naive_means(other, 2))


def test_mosaic_preview_cache_averages_the_stitched_mosaic(tiled_experiment, cache_on):
    for p in (tiled_experiment / "raw").rglob("*.h5"):
        with h5py.File(p, "a") as f:
            del f["Data_2_2_2"]
    layers = read_luxendo(str(tiled_experiment / "main_raw.lux.h5"))
    levels = layers[0][0]
    full = levels[0].compute()
    levels[1][0, 0].compute()
    levels[1][1, 0].compute()
    wait_for_builds(30)
    for t in range(2):
        np.testing.assert_array_equal(levels[1][t].compute(), naive_means(full[t], 2))
        np.testing.assert_array_equal(levels[2][t].compute(), naive_means(full[t], 4))


def test_cache_disabled_by_env(tmp_path, monkeypatch):
    import napari_luxendo._preview as preview
    monkeypatch.setattr(preview, "preview_factors", lambda shape: [2])
    path = write_lux(tmp_path / "single.lux.h5", make_volume(3), pyramids=())
    [(levels, _, _)] = read_luxendo(str(path))
    from napari_luxendo._preview import level_is_cheap, preview_caches
    assert preview_caches(levels[1]) == [None]
    assert not level_is_cheap(levels[1])


def test_scroll_shows_coarse_level_then_refines(tmp_path):
    napari = pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.layers import Layer
    from napari_luxendo._scroll import install

    truth = make_volume(5, shape=(16, 256, 256))
    path = write_lux(tmp_path / "pyr.lux.h5", truth, pyramids=((2, 2, 2), (4, 4, 4)))
    viewer = ViewerModel()
    controller = install(viewer)
    layer = Layer.create(*read_luxendo(str(path))[0])
    viewer.add_layer(layer)
    controller.settle()  # adding the layer moved the sliders
    # Zoomed out: the canvas shows 1000 x 1000 full-resolution pixels on 512 screen pixels.
    corners = np.array([[0.0, 0.0], [1000 * 0.5, 1000 * 0.4]])
    layer._update_draw(scale_factor=1.0, corner_pixels_displayed=corners, shape_threshold=(512, 512))
    assert layer.data_level == 0

    viewer.dims.set_current_step(0, 5)
    assert controller.scrolling and layer.data_level == 2
    np.testing.assert_array_equal(np.asarray(layer._slice.image.raw), truth[::4, ::4, ::4][1])
    controller.settle()
    assert not controller.scrolling and layer.data_level == 0
    np.testing.assert_array_equal(np.asarray(layer._slice.image.raw), truth[5])


def test_scroll_keeps_full_detail_when_coarse_level_is_uncached(tmp_path, monkeypatch):
    pytest.importorskip("napari")
    import napari_luxendo._preview as preview
    from napari.components import ViewerModel
    from napari.layers import Layer
    from napari_luxendo._scroll import install

    monkeypatch.setattr(preview, "preview_factors", lambda shape: [2, 4])
    truth = make_volume(5, shape=(16, 256, 256))
    path = write_lux(tmp_path / "raw.lux.h5", truth, pyramids=())
    viewer = ViewerModel()
    controller = install(viewer)
    layer = Layer.create(*read_luxendo(str(path))[0])
    viewer.add_layer(layer)
    controller.settle()
    corners = np.array([[0.0, 0.0], [1000 * 0.5, 1000 * 0.4]])
    layer._update_draw(scale_factor=1.0, corner_pixels_displayed=corners, shape_threshold=(512, 512))
    viewer.dims.set_current_step(0, 5)
    assert controller.scrolling and layer.data_level == 0
    np.testing.assert_array_equal(np.asarray(layer._slice.image.raw), truth[5])
