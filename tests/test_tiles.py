"""Tiles, time series and multiview: main files, mosaics and affine_to_sample."""

from __future__ import annotations

import json
import warnings

import h5py
import numpy as np
import pytest

from napari_luxendo import napari_get_reader, read_luxendo
from napari_luxendo._lux import compose_affine_to_sample

from conftest import GRID, STEP, TILE, VX, ground_truth, lux_metadata, tile_affine_to_sample


def _read(path, **kw):
    assert napari_get_reader(str(path)) is not None
    return read_luxendo(str(path), **kw)


def _world(affine: np.ndarray, zyx) -> np.ndarray:
    return (affine @ np.r_[np.asarray(zyx, float), 1.0])[:3]


def test_affine_matches_real_bdv_registration():
    """Real LCS-SPIM tile x00-y00: composed affine equals its BigDataViewer ViewRegistration."""
    chain = [
        {"matrix": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "translation": [-1023.5, -1023.5, 0.0]},
        {"matrix": [[2.925, 0, -0.0], [0, 2.925, -0.0], [0, 0, -5.0]],
         "translation": [12212.0, -11283.6, 9500.0]},
        {"matrix": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], "translation": [0.0, 0.0, 0.0]},
    ]
    zyx = compose_affine_to_sample(chain)
    # BDV (x, y, z): "2.925 0 0 9218.263  0 2.925 0 -14277.337  0 0 -5 9500"
    bdv_xyz = np.array([[2.925, 0, 0, 9218.263], [0, 2.925, 0, -14277.337], [0, 0, -5, 9500]])
    expected = np.eye(4)
    expected[:3, :3] = bdv_xyz[::-1, :3][:, ::-1]
    expected[:3, 3] = bdv_xyz[::-1, 3]
    np.testing.assert_allclose(zyx, expected, atol=1e-3)


def test_affine_composes_in_listed_order():
    chain = [
        {"matrix": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [1, 0, 0]},
        {"matrix": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "translation": [0, 0, 7]},
    ]
    a = compose_affine_to_sample(chain)
    # voxel (x=1, y=0, z=0): scale+shift -> (3, 0, 0); rotate -> (0, 3, 0); +z 7
    np.testing.assert_allclose(_world(a, (0, 0, 1)), (7, 3, 0))
    assert compose_affine_to_sample([{"matrix": [[0, 0, 0]] * 3}]) is None
    assert compose_affine_to_sample(None) is None


def test_main_file_builds_one_mosaic_per_channel(tiled_experiment):
    layers = _read(tiled_experiment / "main_raw.lux.h5")
    assert len(layers) == 2
    names = [kw["name"] for _, kw, _ in layers]
    assert names == ["channel_0_cam_long (raw) mosaic (6 tiles)",
                     "channel_1_cam_long (raw) mosaic (6 tiles)"]

    for c, (levels, kw, _) in enumerate(layers):
        assert kw["multiscale"] and kw["blending"] == "additive"
        assert "scale" not in kw and kw["affine"].shape == (5, 5)  # T + ZYX
        full = levels[0]
        truth0, truth1 = ground_truth(0, c), ground_truth(1, c)
        assert full.shape == (2, *truth0.shape)
        np.testing.assert_array_equal(np.asarray(full[0]), truth0)
        np.testing.assert_array_equal(np.asarray(full[1]), truth1)
        # Pyramid levels come from the raw files (the main file links only Data).
        assert levels[1].shape == (2, *(n // 2 for n in truth0.shape))
        assert kw["metadata"]["timepoints"] == [0, 1]
        assert len(kw["metadata"]["views"]) == 6
    assert layers[0][1]["colormap"] != layers[1][1]["colormap"]


def test_mosaic_is_placed_where_the_tiles_are(tiled_experiment):
    [(_, kw, _), _] = _read(tiled_experiment / "main_raw.lux.h5")
    mosaic = kw["affine"][1:, 1:]
    # Every tile voxel lands at the same sample position via its own affine
    # and via the mosaic (tile offsets here are whole voxels).
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            oy, ox = STEP[0] * i, STEP[1] * j
            tile = compose_affine_to_sample(tile_affine_to_sample(oy, ox))
            for u in [(0, 0, 0), (TILE[0] - 1, TILE[1] - 1, TILE[2] - 1), (5, 7, 9)]:
                np.testing.assert_allclose(
                    _world(mosaic, np.add(u, (0, oy, ox))), _world(tile, u), atol=1e-6)
    # z is mirrored, exactly as on the microscope.
    assert mosaic[0, 0] == pytest.approx(-VX[0])


def test_seams_fall_in_the_middle_of_overlaps(tiled_experiment):
    """Give every tile a constant value and check who owns each overlap column."""
    raw = tiled_experiment / "raw"
    values = {}
    for n, f in enumerate(sorted(raw.glob("*channel_0*/Cam_long_00000.lux.h5"))):
        with h5py.File(f, "a") as h:
            h["Data"][...] = n + 1
        values[f.parent.name] = n + 1
    [(levels, _, _), _] = _read(tiled_experiment / "main_raw.lux.h5")
    plane = np.asarray(levels[0][0, 0])
    v = lambda i, j: values[f"stack_1-x{i:02d}-y{j:02d}_channel_0_obj_bottom"]  # noqa: E731
    # X overlap between tile y00 [0, 48) and y01 [37, 85): midpoint ~ 42.
    assert plane[5, 36] == v(0, 0) and plane[5, 48] == v(0, 1)
    assert plane[5, 40] == v(0, 0) and plane[5, 44] == v(0, 1)
    # Y overlap between x00 [0, 40) and x01 [30, 70).
    assert plane[33, 5] == v(0, 0) and plane[36, 5] == v(1, 0)


def test_separate_tiles_and_voxel_mode(tiled_experiment):
    main = tiled_experiment / "main_raw.lux.h5"
    separate = _read(main, tiles="separate")
    assert len(separate) == 12
    assert separate[0][1]["name"] == "channel_0_cam_long raw_stack_1-x00-y00_obj_bottom"
    assert separate[0][1]["affine"].shape == (5, 5)

    voxel = _read(main, transform="voxel")
    assert len(voxel) == 12 and all("affine" not in kw for _, kw, _ in voxel)
    assert voxel[0][1]["scale"] == [1.0, *VX]


def test_environment_variable_opt_out(tiled_experiment, monkeypatch):
    monkeypatch.setenv("NAPARI_LUXENDO_TRANSFORM", "voxel")
    layers = _read(tiled_experiment / "main_raw.lux.h5")
    assert all(kw["metadata"]["placement"] == "voxel_size" for _, kw, _ in layers)
    monkeypatch.setenv("NAPARI_LUXENDO_TRANSFORM", "bogus")
    with pytest.raises(ValueError):
        _read(tiled_experiment / "main_raw.lux.h5")


def test_flat_tile_files_selected_together_make_a_mosaic(tiled_experiment):
    files = sorted(str(p) for p in (tiled_experiment / "raw").glob("*channel_0*/Cam_long_*.lux.h5"))
    assert len(files) == 12  # 6 tiles x 2 timepoints
    [(levels, kw, _)] = read_luxendo(files)
    assert kw["name"] == "ch 0 bottom/long mosaic (6 tiles)"
    np.testing.assert_array_equal(np.asarray(levels[0][1]), ground_truth(1, 0))


def test_single_tile_file_is_placed_in_sample_space(tiled_experiment):
    f = next((tiled_experiment / "raw").glob("stack_1-x01-y02_channel_1*/Cam_long_00000.lux.h5"))
    [(levels, kw, _)] = _read(f)
    assert kw["name"] == "ch 1 st:1-x01-y02 bottom/long"
    np.testing.assert_allclose(kw["affine"], compose_affine_to_sample(
        tile_affine_to_sample(STEP[0], 2 * STEP[1])))


def test_missing_tile_files_warn_and_leave_gaps(tiled_experiment):
    gone = tiled_experiment / "raw/stack_1-x01-y01_channel_0_obj_bottom/Cam_long_00001.lux.h5"
    gone.unlink()
    with pytest.warns(UserWarning, match="missing"):
        layers = _read(tiled_experiment / "main_raw.lux.h5")
    levels = layers[0][0]
    t1 = np.asarray(levels[0][1])
    truth = ground_truth(1, 0)
    # Region owned by the missing tile at t=1 is empty; elsewhere data is intact.
    assert (t1[:, 45:60, 50:70] == 0).all()
    np.testing.assert_array_equal(t1[:, :20, :20], truth[:, :20, :20])
    np.testing.assert_array_equal(np.asarray(levels[0][0]), ground_truth(0, 0))


def test_drift_over_time_warns_and_uses_first_timepoint(tiled_experiment):
    f = tiled_experiment / "raw/stack_1-x00-y00_channel_0_obj_bottom/Cam_long_00001.lux.h5"
    moved = tile_affine_to_sample(0, 0)
    moved[1]["translation"][0] += 50.0
    with h5py.File(f, "a") as h:
        del h["metadata"]
        h["metadata"] = lux_metadata(1, 0, "1-x00-y00", moved)
    with pytest.warns(UserWarning, match="changes over time"):
        layers = _read(tiled_experiment / "main_raw.lux.h5")
    np.testing.assert_array_equal(np.asarray(layers[0][0][0][1]), ground_truth(1, 0))


def _add_processed_views(root):
    with h5py.File(root / "main_raw.lux.h5", "a") as main:
        for t in range(2):
            src = f"raw/stack_1-x00-y00_channel_0_obj_bottom/Cam_long_{t:05d}.lux.h5"
            g = main.create_group(f"timepoint_{t}/channel_0_cam_long/proc_fused")
            g["Data"] = h5py.ExternalLink(src, "Data")
            g["metadata"] = h5py.ExternalLink(src, "metadata")


def test_raw_or_processed_choice(tiled_experiment):
    _add_processed_views(tiled_experiment)
    main = tiled_experiment / "main_raw.lux.h5"
    proc = read_luxendo(str(main), views="proc")
    assert [kw["name"] for _, kw, _ in proc] == ["channel_0_cam_long proc_fused"]
    raw = read_luxendo(str(main), views="raw")
    assert len(raw) == 2 and all("(raw)" in kw["name"] for _, kw, _ in raw)


def test_raw_or_processed_asks_when_both_exist(tiled_experiment, monkeypatch):
    import napari_luxendo._reader as reader

    _add_processed_views(tiled_experiment)
    asked = []
    monkeypatch.setattr(reader, "_choose_raw_or_proc", lambda p: asked.append(p) or "raw")
    layers = read_luxendo(str(tiled_experiment / "main_raw.lux.h5"))
    assert asked and len(layers) == 2


def test_rotated_views_get_their_own_layers(tiled_experiment):
    """Views at another rotation angle cannot share a voxel grid with the rest."""
    f = tiled_experiment / "raw/stack_1-x01-y02_channel_0_obj_bottom/Cam_long_00000.lux.h5"
    chain = tile_affine_to_sample(STEP[0], 2 * STEP[1]) + [
        {"matrix": [[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]], "translation": [0, 0, 0]}]
    with h5py.File(f, "a") as h:
        del h["metadata"]
        h["metadata"] = lux_metadata(0, 0, "1-x01-y02", chain)
    layers = _read(tiled_experiment / "main_raw.lux.h5")
    names = [kw["name"] for _, kw, _ in layers]
    assert names[0] == "channel_0_cam_long (raw) mosaic (5 tiles)"
    assert "channel_0_cam_long raw_stack_1-x01-y02_obj_bottom" in names


def test_contrast_limits_shared_within_a_channel(tiled_experiment):
    sep = _read(tiled_experiment / "main_raw.lux.h5", tiles="separate")
    ch0 = [kw["contrast_limits"] for _, kw, _ in sep[:6]]
    assert all(c == ch0[0] for c in ch0)
    assert 0 < ch0[0][0] < ch0[0][1] <= 60000


def test_real_main_file_layout_is_recognised(tmp_path):
    """Mirror the link layout of a real main_raw.lux.h5 (links to Data + metadata only)."""
    with h5py.File(tmp_path / "main_raw.lux.h5", "w") as f:
        g = f.create_group("timepoint_0/channel_0_cam_long/raw_stack_1-x00-y00_obj_bottom")
        g["Data"] = h5py.ExternalLink(
            "raw/stack_1-x00-y00_channel_0_obj_bottom/Cam_long_00000.lux.h5", "Data")
        g["metadata"] = h5py.ExternalLink(
            "raw/stack_1-x00-y00_channel_0_obj_bottom/Cam_long_00000.lux.h5", "metadata")
    path = str(tmp_path / "main_raw.lux.h5")
    assert napari_get_reader(path) is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="No readable"):
            read_luxendo(path)


def test_tiles_at_different_depths(tiled_experiment):
    """A tile acquired 2 planes deeper extends the mosaic in Z and owns its own planes."""
    f = tiled_experiment / "raw/stack_1-x01-y02_channel_0_obj_bottom/Cam_long_00000.lux.h5"
    chain = tile_affine_to_sample(STEP[0], 2 * STEP[1])
    chain[1]["translation"][2] -= 2 * VX[0]  # z is mirrored: 2 planes further along the mosaic Z
    with h5py.File(f, "a") as h:
        h["Data"][...] = 7
        del h["metadata"]
        h["metadata"] = lux_metadata(0, 0, "1-x01-y02", chain)
    layers = read_luxendo(str(tiled_experiment / "main_raw.lux.h5"), views="raw")
    full = layers[0][0][0]
    assert full.shape[1] == TILE[0] + 2
    vol = np.asarray(full[0])
    cy, cx = STEP[0] + TILE[1] // 2, 2 * STEP[1] + TILE[2] // 2  # centre of that tile
    assert (vol[2:, cy, cx] == 7).all()  # its planes
    assert vol[0, cy, cx] == 0  # above it nothing else covers this column
    np.testing.assert_array_equal(vol[:TILE[0], 5, 5], ground_truth(0, 0)[:, 5, 5])
