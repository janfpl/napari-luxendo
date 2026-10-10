"""In-memory block cache, prefetch around the view and coarse-first zooming."""
import numpy as np
import pytest

from napari_luxendo import _tilecache, read_luxendo
from napari_luxendo._tilecache import CachedSource, TileStore
from conftest import make_volume, write_lux


class Counting:
    def __init__(self, data):
        self.data, self.shape, self.dtype = data, data.shape, data.dtype
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.data[key]


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    monkeypatch.setenv("NAPARI_LUXENDO_TILE_CACHE_MB", "64")
    _tilecache.clear()
    yield
    _tilecache.clear()


def test_store_evicts_least_recently_used_by_bytes():
    store = TileStore(capacity=3 * 800)
    for i in range(3):
        store.put(("t", i), np.zeros(100))  # 800 bytes each
    assert store.get(("t", 0)) is not None  # now most recent
    store.put(("t", 3), np.zeros(100))
    assert ("t", 1) not in store and ("t", 0) in store and store.nbytes == 2400


def test_crop_reads_are_not_kept_but_whole_blocks_are():
    truth = make_volume(1, shape=(3, 1100, 700))
    inner = Counting(truth)
    src = CachedSource(inner)
    crop = (slice(1, 2), slice(10, 300), slice(20, 400))
    np.testing.assert_array_equal(src[crop], truth[crop])
    assert not src.has_region(1, 10, 300, 20, 400)

    src.fill_block(1, 0, 0)
    src.fill_block(1, 2, 1)  # ragged corner block
    assert src.has_region(1, 10, 300, 20, 400) and src.has_region(1, 1024, 1100, 512, 700)
    reads = len(inner.reads)
    np.testing.assert_array_equal(src[crop], truth[crop])
    corner = (slice(1, 2), slice(1030, 1100), slice(600, 700))
    np.testing.assert_array_equal(src[corner], truth[corner])
    assert len(inner.reads) == reads  # both from memory
    # Other planes are separate blocks.
    assert not src.has_region(0, 10, 300, 20, 400)


def test_samples_of_an_unfinished_preview_are_not_kept():
    truth = make_volume(2, shape=(2, 600, 600))
    ready = [False]
    src = CachedSource(Counting(truth), lambda: ready[0])
    src.fill_block(0, 0, 0)
    assert not src.has_region(0, 0, 10, 0, 10)
    ready[0] = True
    src.fill_block(0, 0, 0)
    assert src.has_region(0, 0, 10, 0, 10)


def test_budget_zero_turns_the_cache_off(monkeypatch):
    monkeypatch.setenv("NAPARI_LUXENDO_TILE_CACHE_MB", "0")
    _tilecache.clear()
    truth = make_volume(3, shape=(2, 600, 600))
    src = CachedSource(Counting(truth))
    src.fill_block(0, 0, 0)
    assert not src.has_region(0, 0, 10, 0, 10)
    np.testing.assert_array_equal(src[(slice(0, 1), slice(0, 512), slice(0, 512))],
                                  truth[:1, :512, :512])


@pytest.fixture
def viewer_with_volume(tmp_path):
    pytest.importorskip("napari")
    from napari.components import ViewerModel
    from napari.layers import Layer
    from napari_luxendo._scroll import install

    truth = make_volume(5, shape=(8, 2048, 2048))
    path = write_lux(tmp_path / "big.lux.h5", truth, pyramids=((2, 2, 2), (4, 4, 4)))
    viewer = ViewerModel()
    controller = install(viewer)
    layer = Layer.create(*read_luxendo(str(path))[0])
    viewer.add_layer(layer)
    controller.settle()
    controller.wait_prefetch(30)
    return viewer, controller, layer, truth


def _shown(layer, truth, z):
    """Check the loaded full-resolution tile against *truth* wherever napari placed it."""
    raw = np.asarray(layer._slice.image.raw)
    ty, tx = (int(t) for t in layer._slice.tile_to_data.translate[-2:])
    np.testing.assert_array_equal(raw, truth[z, ty:ty + raw.shape[0], tx:tx + raw.shape[1]])
    return ty, tx, raw.shape


def _draw(layer, y0, x0, size, threshold=(256, 256)):
    """Show full-resolution pixels [y0:y0+size, x0:x0+size] on a 256 x 256 canvas."""
    # World units: Y 0.5 um and X 0.4 um per voxel (conftest VOXEL).
    corners = np.array([[y0 * 0.5, x0 * 0.4], [(y0 + size) * 0.5, (x0 + size) * 0.4]])
    layer._update_draw(scale_factor=1.0, corner_pixels_displayed=corners,
                       shape_threshold=threshold)


def test_prefetch_loads_the_surroundings_of_the_view(viewer_with_volume):
    from napari_luxendo._scroll import locate

    viewer, controller, layer, truth = viewer_with_volume
    _draw(layer, 600, 600, 200)
    assert layer.data_level == 0
    controller.settle()
    assert controller.wait_prefetch(30)
    source, z = locate(layer, 0)
    # One screen (200 px) around [600:800] is in memory; far away is not.
    assert source.has_region(z, 400, 1000, 400, 1000)
    assert not source.has_region(z, 1600, 2048, 1600, 2048)
    # Panning into it shows the right pixels.
    _draw(layer, 850, 450, 200)
    assert layer.data_level == 0
    ty, tx, (h, w) = _shown(layer, truth, z)
    assert ty <= 850 and tx <= 450 and ty + h >= 1051 and tx + w >= 651


def test_prefetch_can_be_turned_off(viewer_with_volume, monkeypatch):
    from napari_luxendo._scroll import locate

    monkeypatch.setenv("NAPARI_LUXENDO_PREFETCH", "0")
    viewer, controller, layer, truth = viewer_with_volume
    _draw(layer, 1600, 1600, 200)
    controller.settle()
    controller.wait_prefetch(30)
    source, z = locate(layer, 0)
    assert not source.has_region(z, 1400, 2000, 1400, 2000)


def test_zoom_shows_coarse_level_first_unless_detail_is_in_memory(viewer_with_volume):
    from napari_luxendo._scroll import locate

    viewer, controller, layer, truth = viewer_with_volume
    _draw(layer, 0, 0, 2048)  # whole plane: napari picks the coarsest level
    controller.settle()
    controller.wait_prefetch(30)
    _tilecache.clear()

    # Zoom into an area whose full detail is not in memory: coarse first.
    controller._on_zoom()
    _draw(layer, 1700, 1700, 200)
    assert controller.zooming and layer.data_level > 0
    controller.settle()
    assert layer.data_level == 0
    _shown(layer, truth, locate(layer, 0)[1])
    assert controller.wait_prefetch(30)

    # Zooming within what is now in memory goes straight to full detail.
    controller._on_zoom()
    _draw(layer, 1750, 1750, 150)
    assert controller.zooming and layer.data_level == 0
