from __future__ import annotations

from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import pytest

from napari_luxendo import napari_get_reader, open_lux_volume
from napari_luxendo._reader import colormap_from_description

from conftest import SHAPE, make_volume, write_lux


def _read(path):
    reader = napari_get_reader(str(path) if isinstance(path, Path) else path)
    assert callable(reader)
    return reader(str(path) if isinstance(path, Path) else path)


def test_single_lux_file(tmp_path):
    data = make_volume(0)
    path = write_lux(tmp_path / "a.lux.h5", data)

    [(levels, kwargs, layer_type)] = _read(path)

    assert layer_type == "image"
    assert kwargs["multiscale"] is True
    assert [lv.shape for lv in levels] == [SHAPE, (12, 20, 24), (6, 10, 12)]
    assert all(isinstance(lv, da.Array) for lv in levels)
    np.testing.assert_array_equal(np.asarray(levels[0]), data)
    assert kwargs["scale"] == [2.0, 0.5, 0.4]
    assert kwargs["name"] == "Green-488"
    assert kwargs["colormap"] == "green"
    assert kwargs["metadata"]["pyramid_levels"] == ["Data", "Data_2_2_2", "Data_4_4_4"]
    assert kwargs["metadata"]["luxendo"]["processingInformation"]["channel_id"] == 1
    lo, hi = kwargs["contrast_limits"]
    assert 100 <= lo < hi <= 5000


def test_single_level_without_metadata(tmp_path):
    path = write_lux(tmp_path / "plain.lux.h5", make_volume(1), description=None,
                     pyramids=(), voxel=None)
    [(data, kwargs, _)] = _read(path)
    assert isinstance(data, da.Array) and data.shape == SHAPE
    assert kwargs["multiscale"] is False
    assert kwargs["scale"] == [1.0, 1.0, 1.0]
    assert kwargs["name"] == "plain"
    assert kwargs["colormap"] == "gray"


def test_dask_chunks_follow_hdf5_z_chunks(tmp_path):
    vol = open_lux_volume(write_lux(tmp_path / "a.lux.h5", make_volume(0)))
    assert vol.data.chunks[0][0] == 8
    assert vol.data.chunks[1:] == ((SHAPE[1],), (SHAPE[2],))


def test_non_shrinking_pyramid_level_is_dropped(tmp_path):
    path = write_lux(tmp_path / "a.lux.h5", make_volume(0), pyramids=((1, 1, 1), (2, 2, 2)))
    vol = open_lux_volume(path)
    assert vol.level_names == ["Data", "Data_2_2_2"]


def test_ims_header(lux_dir):
    layers = _read(lux_dir / "dataset.ims")
    assert [kw["name"] for _, kw, _ in layers] == ["GFP", "mCherry"]

    levels, kwargs, _ = layers[1]
    # Two timepoints stacked into T, Z, Y, X; pyramids come from the lux files.
    assert [lv.shape for lv in levels] == [(2, *SHAPE), (2, 12, 20, 24), (2, 6, 10, 12)]
    np.testing.assert_array_equal(np.asarray(levels[0][1]), make_volume(11))
    assert kwargs["scale"] == [1.0, 2.0, 0.5, 0.4]
    assert kwargs["blending"] == "additive"
    assert kwargs["colormap"]["colors"][1] == [1.0, 0.0, 1.0, 1.0]
    assert kwargs["metadata"]["source"].endswith("dataset.ims")
    assert len(kwargs["metadata"]["files"]) == 2


def test_bdv_header(lux_dir):
    layers = _read(lux_dir / "dataset_bdv.h5")
    assert [kw["name"] for _, kw, _ in layers] == ["ch-488", "ch-561"]
    assert [kw["colormap"] for _, kw, _ in layers] == ["green", "red"]
    assert layers[0][0][0].shape == (2, *SHAPE)


def test_header_falls_back_to_header_voxel_size(lux_dir):
    for p in lux_dir.glob("*.lux.h5"):
        with h5py.File(p, "a") as f:
            del f["metadata"]
    [(_, kwargs, _), _] = _read(lux_dir / "dataset.ims")
    assert kwargs["scale"] == pytest.approx([1.0, 2.0, 0.5, 0.4])


def test_header_with_missing_channel_file(lux_dir):
    (lux_dir / "uni_tp-1_ch-0.lux.h5").unlink()
    reader = napari_get_reader(str(lux_dir / "dataset.ims"))
    with pytest.warns(UserWarning, match="uni_tp-1_ch-0"):
        layers = reader(str(lux_dir / "dataset.ims"))
    levels = layers[0][0]
    assert levels[0].shape == (2, *SHAPE)
    assert not np.asarray(levels[0][1]).any()  # missing timepoint left empty
    np.testing.assert_array_equal(np.asarray(levels[0][0]), make_volume(0))


def test_mismatched_timepoints_are_left_empty(lux_dir):
    write_lux(lux_dir / "uni_tp-1_ch-0.lux.h5", make_volume(5, shape=(10, 40, 48)))
    with pytest.warns(UserWarning, match="differ in shape"):
        levels, _, _ = _read(lux_dir / "dataset.ims")[0]
    assert levels[0].shape == (2, *SHAPE)
    assert not np.asarray(levels[0][1]).any()


def test_list_of_files_gets_distinct_colormaps(tmp_path):
    paths = [
        str(write_lux(tmp_path / f"c{i}.lux.h5", make_volume(i), description=f"ch{i}"))
        for i in range(3)
    ]
    layers = _read(paths)
    assert [kw["colormap"] for _, kw, _ in layers] == ["green", "magenta", "cyan"]
    assert all(kw["blending"] == "additive" for _, kw, _ in layers)


@pytest.mark.parametrize("name", ["other.h5", "notes.txt", "native.ims"])
def test_rejects_files_that_are_not_luxendo(tmp_path, name):
    path = tmp_path / name
    if name == "other.h5":
        with h5py.File(path, "w") as f:
            f.create_dataset("images", data=np.zeros((2, 2)))
    elif name == "native.ims":
        with h5py.File(path, "w") as f:
            f.create_dataset("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
                             data=np.zeros((2, 2, 2)))
    else:
        path.write_text("hi")
    assert napari_get_reader(str(path)) is None
    assert napari_get_reader(str(tmp_path / "missing.lux.h5")) is None


def test_mixed_list_is_rejected(tmp_path):
    good = write_lux(tmp_path / "a.lux.h5", make_volume(0))
    bad = tmp_path / "b.txt"
    bad.write_text("x")
    assert napari_get_reader([str(good), str(bad)]) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Green-488", "green"),
        ("Red-561", "red"),
        ("405 nm", "blue"),
        ("640", "magenta"),
        ("Laser 515", "yellow"),
        ("Cyan channel", "cyan"),
        ("Green-22", "green"),
        ("ch-1", None),
        ("", None),
    ],
)
def test_colormap_from_description(text, expected):
    assert colormap_from_description(text) == expected
