"""Views whose ``Data`` and ``metadata`` live behind external links."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import numpy as np

from napari_luxendo import read_luxendo
from napari_luxendo._lux import open_lux_dataset, open_lux_volume

from conftest import TILE, VX


def _metadata(tx: float) -> str:
    nz, ny, nx = TILE
    proc = {
        "voxel_size_um": {"width": VX[2], "height": VX[1], "depth": VX[0]},
        "image_size_vx": {"width": nx, "height": ny, "depth": nz},
        "affine_to_sample": [{"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              "translation": [tx, 0.0, 0.0]}],
    }
    return json.dumps({"processingInformation": proc})


def _pixels(path: Path, tx: float | None = 0.0) -> np.ndarray:
    data = np.arange(np.prod(TILE), dtype=np.uint16).reshape(TILE)
    with h5py.File(path, "w") as f:
        f["Data"] = data
        f["Data_2_2_2"] = data[::2, ::2, ::2]
        if tx is not None:
            f["metadata"] = _metadata(tx)
    return data


def _x_translation(vol) -> float:
    assert vol.affine is not None
    return float(vol.affine[2, 3])


def test_header_link_keeps_wrapper_view_metadata(tmp_path):
    """A header pointing at a wrapper view uses the wrapper's metadata, not the pixel file's."""
    _pixels(tmp_path / "pixels.lux.h5", tx=0.0)
    with h5py.File(tmp_path / "wrapper.lux.h5", "w") as f:
        g = f.create_group("timepoint_0/channel_0/raw")
        g["Data"] = h5py.ExternalLink("pixels.lux.h5", "/Data")
        g["metadata"] = _metadata(100.0)

    vol = open_lux_dataset(tmp_path / "wrapper.lux.h5", "timepoint_0/channel_0/raw/Data")

    assert _x_translation(vol) == 100.0
    assert vol.view_name == "raw"


def test_external_view_group_resolves_links_next_to_its_own_file(tmp_path):
    """Links inside a view group in another file are relative to that file."""
    sub = tmp_path / "sub"
    sub.mkdir()
    truth = _pixels(sub / "pixels.lux.h5", tx=0.0)
    with h5py.File(sub / "views.h5", "w") as f:
        g = f.create_group("raw")
        g["Data"] = h5py.ExternalLink("pixels.lux.h5", "/Data")
        g["metadata"] = h5py.ExternalLink("pixels.lux.h5", "/metadata")
    with h5py.File(tmp_path / "main_raw.lux.h5", "w") as m:
        ch = m.create_group("timepoint_0/channel_0")
        ch["raw"] = h5py.ExternalLink("sub/views.h5", "/raw")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        [(data, kw, _)] = read_luxendo(str(tmp_path / "main_raw.lux.h5"), tiles="separate")

    level0 = data[0] if kw["multiscale"] else data
    np.testing.assert_array_equal(np.asarray(level0).reshape(TILE), truth)


def test_metadata_fallback_with_explicit_pyramid_link(tmp_path):
    """A wrapper linking Data and its pyramid levels still inherits the pixel file's metadata."""
    _pixels(tmp_path / "pixels.lux.h5", tx=100.0)
    with h5py.File(tmp_path / "wrapper.lux.h5", "w") as f:
        f["Data"] = h5py.ExternalLink("pixels.lux.h5", "/Data")
        f["Data_2_2_2"] = h5py.ExternalLink("pixels.lux.h5", "/Data_2_2_2")

    vol = open_lux_volume(tmp_path / "wrapper.lux.h5")

    assert vol.level_names == ["Data", "Data_2_2_2"]
    assert vol.voxel_size_um is not None
    assert _x_translation(vol) == 100.0
