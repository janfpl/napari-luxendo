"""Regression tests for the four issues found in external review (October 2026)."""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from napari_luxendo import napari_get_reader, read_luxendo

from conftest import TILE, lux_metadata, tile_affine_to_sample, write_lux, make_volume


def _tile(path: Path, t: int, j: int, *, levels=((2, 2, 2),), value=None) -> np.ndarray:
    data = (np.full(TILE, value, np.uint16) if value is not None
            else np.random.default_rng(10 * t + j).integers(1, 999, TILE, dtype=np.uint16))
    with h5py.File(path, "w") as f:
        f["Data"] = data
        for fw, fh, fd in levels:
            f[f"Data_{fw}_{fh}_{fd}"] = data[::fd, ::fh, ::fw]
        f["metadata"] = lux_metadata(t, 0, f"1-x00-y{j:02d}", tile_affine_to_sample(0, 37 * j))
    return data


def _main(root: Path, layout: dict, name="main_raw.lux.h5") -> Path:
    """layout: {(t, view): relative file or None (link to a missing file)}."""
    with h5py.File(root / name, "w") as m:
        for (t, view), rel in layout.items():
            g = m.create_group(f"timepoint_{t}/channel_0/{view}")
            g["Data"] = h5py.ExternalLink(rel, "Data")
            g["metadata"] = h5py.ExternalLink(rel, "metadata")
    return root / name


def _quiet_read(path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_luxendo(str(path), **kw)


# 1. Pyramid levels must exist at every timepoint ------------------------------


@pytest.mark.parametrize("tiles", ["mosaic", "separate"])
def test_later_timepoint_without_pyramid_still_reads(tmp_path, tiles):
    truth = {}
    for t in range(2):
        for j in range(2):
            truth[t, j] = _tile(tmp_path / f"t{t}_y{j}.lux.h5", t, j,
                                levels=((2, 2, 2),) if t == 0 else ())
    main = _main(tmp_path, {(t, f"raw_y{j}"): f"t{t}_y{j}.lux.h5" for t in range(2) for j in range(2)})
    with pytest.warns(UserWarning, match="pyramid levels differ"):
        layers = read_luxendo(str(main), tiles=tiles)
    data, kw, _ = layers[0]
    assert kw["multiscale"] is False and kw["metadata"]["pyramid_levels"] == ["Data"]
    frame = np.asarray(data[1])  # timepoint 1, actually computed
    np.testing.assert_array_equal(frame[:, :, :TILE[2]][:, :, :30], truth[1, 0][:, :, :30])


def test_same_level_index_with_different_factors_is_dropped(tmp_path):
    _tile(tmp_path / "t0.lux.h5", 0, 0, levels=((2, 2, 2),))
    _tile(tmp_path / "t1.lux.h5", 1, 0, levels=((3, 3, 3),))
    main = _main(tmp_path, {(0, "raw_a"): "t0.lux.h5", (1, "raw_a"): "t1.lux.h5"})
    [(data, kw, _)] = _quiet_read(main)
    assert kw["metadata"]["pyramid_levels"] == ["Data"]
    assert np.asarray(data[1]).shape == TILE


def test_pyramids_kept_when_every_timepoint_has_them(tmp_path):
    for t in range(2):
        _tile(tmp_path / f"t{t}.lux.h5", t, 0)
    main = _main(tmp_path, {(t, "raw_a"): f"t{t}.lux.h5" for t in range(2)})
    [(levels, kw, _)] = read_luxendo(str(main))
    assert kw["metadata"]["pyramid_levels"] == ["Data", "Data_2_2_2"]
    assert np.asarray(levels[1][1]).shape == tuple(n // 2 for n in TILE)


# 2. Headers must open the dataset their link points at ------------------------


def _nested_target(path: Path, values=(7, 9)) -> None:
    with h5py.File(path, "w") as f:
        f["Data"] = np.zeros(TILE, np.uint16)  # decoy at the root
        for j, value in enumerate(values):
            g = f.create_group(f"timepoint_0/channel_0/raw_tile{j}")
            g["Data"] = np.full(TILE, value, np.uint16)
            g["Data_2_2_2"] = np.full(tuple(n // 2 for n in TILE), value, np.uint16)
            g["metadata"] = lux_metadata(0, 0, f"1-x00-y{j:02d}", tile_affine_to_sample(0, 37 * j),
                                         description=f"view{j}")


@pytest.mark.parametrize("kind", ["ims", "bdv"])
def test_header_follows_dataset_path_inside_nested_file(tmp_path, kind):
    _nested_target(tmp_path / "nested.lux.h5")
    links = [h5py.ExternalLink("nested.lux.h5", f"/timepoint_0/channel_0/raw_tile{j}/Data")
             for j in range(2)]
    if kind == "ims":
        header = tmp_path / "h.ims"
        with h5py.File(header, "w") as f:
            for j, link in enumerate(links):
                f.create_group(f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {j}")["Data"] = link
    else:
        header = tmp_path / "h_bdv.h5"
        with h5py.File(header, "w") as f:
            for j, link in enumerate(links):
                f.create_group(f"t00000/s{j:02d}/0")["cells"] = link
    layers = read_luxendo(str(header), tiles="separate")
    assert len(layers) == 2
    for (levels, kw, _), value, name in zip(layers, (7, 9), ("view0", "view1")):
        assert kw["metadata"]["luxendo"]["processingInformation"]["channel_description"] == name
        assert kw["metadata"]["pyramid_levels"] == ["Data", "Data_2_2_2"]  # the view's own levels
        assert (np.asarray(levels[0]) == value).all()
        assert "affine" in kw  # the view's own metadata, not the root's


def test_header_link_to_non_data_dataset_is_a_clear_error(tmp_path):
    write_lux(tmp_path / "a.lux.h5", make_volume(0))
    with h5py.File(tmp_path / "h.ims", "w") as f:
        f.create_group("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0")["Data"] = \
            h5py.ExternalLink("a.lux.h5", "/Data_2_2_2")
    with pytest.raises(ValueError, match="expected a link to a Luxendo 'Data'"):
        read_luxendo(str(tmp_path / "h.ims"))


# 3. Timepoints listed by a main file or header are kept -----------------------


@pytest.mark.parametrize("gone", [0, 1, 2])
def test_wholly_missing_timepoint_retains_empty_frame(tmp_path, gone):
    for t in range(3):
        if t != gone:
            for j in range(2):
                _tile(tmp_path / f"t{t}_y{j}.lux.h5", t, j, value=t + 1)
    main = _main(tmp_path, {(t, f"raw_y{j}"): f"t{t}_y{j}.lux.h5" for t in range(3) for j in range(2)})
    with pytest.warns(UserWarning, match="missing"):
        [(levels, kw, _)] = read_luxendo(str(main))
    assert kw["metadata"]["timepoints"] == [0, 1, 2]
    for t in range(3):
        frame = np.asarray(levels[0][t])
        assert (frame == 0).all() if t == gone else (frame == t + 1).all()


def test_wholly_missing_timepoint_in_header(lux_dir):
    for c in range(2):
        (lux_dir / f"uni_tp-1_ch-{c}.lux.h5").unlink()
    with pytest.warns(UserWarning, match="missing"):
        layers = read_luxendo(str(lux_dir / "dataset.ims"))
    levels = layers[0][0]
    assert levels[0].shape[0] == 2 and not np.asarray(levels[0][1]).any()


def test_sparse_timepoints_are_not_filled_in(tmp_path):
    for t in (0, 2):
        _tile(tmp_path / f"t{t}.lux.h5", t, 0)
    main = _main(tmp_path, {(t, "raw_a"): f"t{t}.lux.h5" for t in (0, 2)})
    [(levels, kw, _)] = read_luxendo(str(main))
    assert kw["metadata"]["timepoints"] == [0, 2] and levels[0].shape[0] == 2


def test_unselected_views_do_not_add_timepoints(tmp_path):
    for t in range(2):
        _tile(tmp_path / f"t{t}.lux.h5", t, 0)
    # raw views at t0, t1; a processed view only at t5
    main = _main(tmp_path, {(0, "raw_a"): "t0.lux.h5", (1, "raw_a"): "t1.lux.h5",
                            (5, "proc_a"): "t0.lux.h5"})
    [(_, kw, _)] = read_luxendo(str(main), views="raw")
    assert kw["metadata"]["timepoints"] == [0, 1]


# 4. Generic .h5 files are not claimed -----------------------------------------


def test_generic_h5_data_is_not_claimed(tmp_path):
    with h5py.File(tmp_path / "unrelated.h5", "w") as f:
        f["Data"] = np.zeros((4, 4, 4))
    assert napari_get_reader(str(tmp_path / "unrelated.h5")) is None
    with h5py.File(tmp_path / "group.h5", "w") as f:
        f.create_group("Data")
    assert napari_get_reader(str(tmp_path / "group.h5")) is None
    with h5py.File(tmp_path / "scalar.lux.h5", "w") as f:
        f["Data"] = 3
    assert napari_get_reader(str(tmp_path / "scalar.lux.h5")) is None


def test_plain_h5_with_luxendo_metadata_is_claimed(tmp_path):
    _tile(tmp_path / "exported.h5", 0, 0)
    assert napari_get_reader(str(tmp_path / "exported.h5")) is not None
    with h5py.File(tmp_path / "other_meta.h5", "w") as f:
        f["Data"] = np.zeros((4, 4, 4), np.uint16)
        f["metadata"] = '{"something": "else"}'
    assert napari_get_reader(str(tmp_path / "other_meta.h5")) is None
