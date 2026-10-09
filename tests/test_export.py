"""Exporting Luxendo layers to .lux.h5 / BigTIFF (engine and layer conversion)."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import tifffile

from napari_luxendo import open_lux_volume, read_luxendo
from napari_luxendo import _export
from napari_luxendo._export import (
    FORMAT_TIFF,
    IMS_FILENAME,
    SUMMARY_FILENAME,
    ExportSource,
    affine_to_luxendo,
    plan_export,
    run_export,
)
from napari_luxendo._lux import compose_affine_to_sample

from conftest import VX, ground_truth

ROI = (2, 10, 5, 27, 3, 40)  # z0, z1, y0, y1, x0, x1


def block_mean(data, fw, fh, fd):
    """Reference pyramid level: floor of the (fd, fh, fw) block mean."""
    nz, ny, nx = data.shape
    oz, oy, ox = nz // fd, ny // fh, nx // fw
    trimmed = data[: oz * fd, : oy * fh, : ox * fw].astype(np.uint64)
    sums = trimmed.reshape(oz, fd, oy, fh, ox, fw).sum(axis=(1, 3, 5))
    return (sums // (fw * fh * fd)).astype(np.uint16)


def crop(data, roi):
    z0, z1, y0, y1, x0, x1 = roi
    return data[z0:z1, y0:y1, x0:x1]


@pytest.fixture
def small_slabs(monkeypatch):
    """Force 3-plane slabs so pyramid Z-groups straddle slab boundaries."""
    monkeypatch.setattr(_export, "compute_chunk_size", lambda *a, **k: 3)


def _source(name="Green-488", roi=None, timepoints=(0, 1), seed=0, shape=(13, 30, 44)):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 60000, size=(len(timepoints), *shape), dtype=np.uint16)
    affine = np.diag([-5.0, 2.925, 2.925, 1.0])
    affine[:3, 3] = [9500.0, -2000.0, 1000.0]
    return ExportSource(
        name=name,
        data=data,
        timepoints=list(timepoints),
        roi=roi,
        pyramid_levels=["Data_2_2_2", "Data_4_4_4"],
        affine_zyx=affine,
        voxel_size_um=(5.0, 2.925, 2.925),
        metadata={"processingInformation": {
            "channel": "0", "stack": "1-x00-y00", "time_point": "0",
            "image_size_vx": {"width": 1, "height": 1, "depth": 1},
        }},
        color=(0.0, 1.0, 0.0),
        fused_views=6,
    )


def test_affine_to_luxendo_round_trips():
    rng = np.random.default_rng(3)
    a = np.eye(4)
    a[:3, :3] = rng.normal(size=(3, 3))
    a[:3, 3] = rng.normal(size=3) * 100
    np.testing.assert_allclose(compose_affine_to_sample([affine_to_luxendo(a)]), a)


@pytest.mark.parametrize("roi", [None, ROI])
def test_lux_h5_export(tmp_path, small_slabs, roi):
    src = _source(roi=roi)
    progress = []
    plan = plan_export([src], tmp_path / "out")
    summary = run_export(plan, progress_callback=lambda d, t: progress.append((d, t)))

    for frame, t in enumerate(src.timepoints):
        expected = src.data[frame] if roi is None else crop(src.data[frame], roi)
        path = tmp_path / "out" / f"Green-488_tp-{t}.lux.h5"
        with h5py.File(path, "r") as f:
            np.testing.assert_array_equal(f["Data"][()], expected)
            for name, factor in (("Data_2_2_2", 2), ("Data_4_4_4", 4)):
                np.testing.assert_array_equal(
                    f[name][()], block_mean(expected, factor, factor, factor)
                )
        # Rewritten metadata: size, timepoint, fused stack id dropped, placement.
        vol = open_lux_volume(path)
        proc = vol.metadata["raw"]["processingInformation"]
        nz, ny, nx = expected.shape
        assert proc["image_size_vx"] == {"width": nx, "height": ny, "depth": nz}
        assert proc["time_point"] == str(t) and "stack" not in proc
        assert vol.voxel_size_um == (5.0, 2.925, 2.925)
        origin = (0, 0, 0) if roi is None else (roi[0], roi[2], roi[4])
        np.testing.assert_allclose(
            vol.affine @ np.r_[0, 0, 0, 1.0], src.affine_zyx @ np.r_[origin, 1.0]
        )
        assert vol.level_names == ["Data", "Data_2_2_2", "Data_4_4_4"]

    assert progress[-1] == (plan.total_bytes, plan.total_bytes)
    meta = json.loads(summary.read_text())
    assert [o["timepoint"] for o in meta["layers"][0]["outputs"]] == [0, 1]
    assert meta["ims_header"] == IMS_FILENAME


@pytest.mark.parametrize("roi", [None, ROI])
def test_bigtiff_export(tmp_path, small_slabs, roi):
    src = _source(roi=roi)
    plan = plan_export([src], tmp_path / "out", FORMAT_TIFF)
    assert plan.header is None and not plan.write_pyramids
    run_export(plan)
    for frame, t in enumerate(src.timepoints):
        with tifffile.TiffFile(tmp_path / "out" / f"Green-488_tp-{t}.tif") as tf:
            assert tf.is_bigtiff
            got = tf.asarray()
        expected = src.data[frame] if roi is None else crop(src.data[frame], roi)
        np.testing.assert_array_equal(got, expected)


def test_no_pyramids(tmp_path):
    run_export(plan_export([_source()], tmp_path, write_pyramids=False))
    with h5py.File(tmp_path / "Green-488_tp-0.lux.h5", "r") as f:
        assert sorted(f.keys()) == ["Data", "metadata"]


def test_timepoint_range(tmp_path):
    plan = plan_export([_source(timepoints=(3, 4, 5))], tmp_path, timepoint_range=(4, 5))
    assert [job.timepoint for job in plan.jobs] == [4, 5]
    assert [job.frame for job in plan.jobs] == [1, 2]
    with pytest.raises(ValueError, match="No timepoint"):
        plan_export([_source()], tmp_path, timepoint_range=(7, 9))


def test_ims_header_links_channels_and_timepoints(tmp_path):
    sources = [_source("Green-488", seed=0), _source("Red-561", seed=1)]
    sources[1].color = (1.0, 0.0, 1.0)
    run_export(plan_export(sources, tmp_path))

    layers = read_luxendo(str(tmp_path / IMS_FILENAME))
    assert [kw["name"] for _, kw, _ in layers] == ["Green-488", "Red-561"]
    for src, (levels, kw, _) in zip(sources, layers):
        assert kw["metadata"]["timepoints"] == [0, 1]
        np.testing.assert_array_equal(np.asarray(levels[0]), src.data)
        assert len(levels) == 3
    np.testing.assert_allclose(layers[1][1]["colormap"]["colors"][-1], [1, 0, 1, 1])
    with h5py.File(tmp_path / IMS_FILENAME, "r") as f:
        g = f["DataSet/ResolutionLevel 1/TimePoint 1/Channel 0"]
        assert g.attrs["ImageSizeX"].tobytes() == b"22"


def test_ims_header_skipped_for_different_sizes(tmp_path):
    plan = plan_export([_source(), _source("Red", roi=ROI)], tmp_path)
    assert plan.header is None and "differ in size" in plan.header_note


def test_duplicate_layer_names_get_distinct_files(tmp_path):
    plan = plan_export([_source("a/b"), _source("a/b")], tmp_path, timepoint_range=(0, 0))
    assert [j.output.name for j in plan.jobs] == ["a_b_tp-0.lux.h5", "a_b_2_tp-0.lux.h5"]


def test_refuses_to_write_next_to_sources(tmp_path):
    src = _source()
    src.source_files = [str(tmp_path / "raw.lux.h5")]
    with pytest.raises(ValueError, match="does not hold the source files"):
        plan_export([src], tmp_path)


@pytest.mark.parametrize("roi", [(0, 14, 0, 30, 0, 44), (5, 5, 0, 30, 0, 44), (0, 13, -1, 3, 0, 4)])
def test_invalid_roi(tmp_path, roi):
    with pytest.raises(ValueError, match="ROI"):
        plan_export([_source(roi=roi)], tmp_path)


def test_cancel_keeps_finished_files_and_removes_partial(tmp_path, small_slabs):
    plan = plan_export([_source()], tmp_path)
    slabs_per_file = -(-13 // 3)
    calls = iter([False] * (slabs_per_file + 1))
    assert run_export(plan, cancel_check=lambda: next(calls, True)) is None
    assert (tmp_path / "Green-488_tp-0.lux.h5").exists()
    assert not (tmp_path / "Green-488_tp-1.lux.h5").exists()
    assert not (tmp_path / SUMMARY_FILENAME).exists()
    assert not (tmp_path / IMS_FILENAME).exists()


# --------------------------------------------------------------------------- #
# From napari layers (a tiled, time-lapse experiment)
# --------------------------------------------------------------------------- #

napari = pytest.importorskip("napari")

from napari.components import ViewerModel  # noqa: E402

from napari_luxendo._coordinates import CAMERA, set_coordinates  # noqa: E402
from napari_luxendo._export_layers import (  # noqa: E402
    export_source,
    exportable_layers,
    voxel_roi,
    world_extent,
)


def _viewer(path):
    viewer = ViewerModel()
    for data, meta, kind in read_luxendo(str(path)):
        viewer.add_layer(napari.layers.Layer.create(data, meta, kind))
    return viewer


def _world_box(layer, roi):
    """World box whose voxel centres are exactly *roi* of *layer*."""
    z0, z1, y0, y1, x0, x1 = roi
    corners = np.array([
        layer.data_to_world([0, z, y, x])[1:]
        for z in (z0, z1 - 1) for y in (y0, y1 - 1) for x in (x0, x1 - 1)
    ])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def test_world_box_maps_to_voxels_through_flipped_z(tiled_experiment):
    viewer = _viewer(tiled_experiment / "main_raw.lux.h5")
    layer = exportable_layers(viewer)[0]
    assert layer.affine.affine_matrix[1, 1] < 0  # Z runs backwards in sample space
    shape = ground_truth(0, 0).shape

    lo, hi = world_extent(layer)
    assert voxel_roi(layer, (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])) == (
        0, shape[0], 0, shape[1], 0, shape[2]
    )
    assert voxel_roi(layer, _world_box(layer, ROI)) == ROI
    assert voxel_roi(layer, (0, 1, 0, 1, 0, 1)) is None  # far from the sample

    # The camera-coordinates toggle changes the world box, not the voxels it means.
    box = _world_box(layer, ROI)
    set_coordinates(layer, CAMERA)
    assert voxel_roi(layer, box) != ROI
    assert voxel_roi(layer, _world_box(layer, ROI)) == ROI


def test_fused_mosaic_export_reopens_in_place(tiled_experiment, tmp_path):
    viewer = _viewer(tiled_experiment / "main_raw.lux.h5")
    layers = exportable_layers(viewer)
    box = _world_box(layers[0], ROI)
    sources = [export_source(layer, voxel_roi(layer, box)) for layer in layers]
    assert sources[0].fused_views == 6 and sources[0].pyramid_levels == ["Data_2_2_2"]

    out = tmp_path / "export"
    run_export(plan_export(sources, out))
    files = sorted(p.name for p in out.glob("*.lux.h5"))
    assert len(files) == 4  # 2 channels x 2 timepoints

    for c, layer in enumerate(layers):
        for t in (0, 1):
            vol = open_lux_volume(next(out.glob(f"channel_{c}*_tp-{t}.lux.h5")))
            np.testing.assert_array_equal(np.asarray(vol.data), crop(ground_truth(t, c), ROI))
            # Voxel (0, 0, 0) of the export sits where ROI's origin was in the mosaic.
            np.testing.assert_allclose(
                vol.affine @ np.r_[0, 0, 0, 1.0],
                [*layer.data_to_world([t, ROI[0], ROI[2], ROI[4]])[1:], 1.0],
                atol=1e-9,
            )
            assert np.allclose(np.abs(np.diag(vol.affine))[:3], VX)

    # The .ims header reopens as one layer per channel, with both timepoints.
    reopened = read_luxendo(str(out / IMS_FILENAME))
    assert len(reopened) == 2
    np.testing.assert_array_equal(
        np.asarray(reopened[1][0][0][1]), crop(ground_truth(1, 1), ROI)
    )
