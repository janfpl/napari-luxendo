"""Each exported frame carries the metadata of its own source files."""

import copy
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from test_spec import RAW_TILE

from napari_luxendo import _export, read_luxendo
from napari_luxendo._export import export_metadata, plan_export, run_export
from napari_luxendo._export_layers import export_source
from napari_luxendo._lux import parse_metadata
from napari_luxendo._spec import to_spec


def _sources(layers):
    return [export_source(SimpleNamespace(
        data=data, ndim=4 if len(kw["metadata"]["timepoints"]) > 1 else 3,
        **{k: kw[k] for k in ("name", "metadata", "multiscale")}))
        for data, kw, _kind in layers]


def sources_for(folder):
    return _sources(read_luxendo([str(p) for p in sorted(folder.glob("*.lux.h5"))], tiles="separate"))


@pytest.fixture
def three_channels(tmp_path):
    """Three channels at two timepoints, every file with its own ids and time stamps."""
    root = tmp_path / "input"
    root.mkdir()
    for t in (0, 1):
        for c in (0, 1, 2):
            md = copy.deepcopy(RAW_TILE)
            p = md["processingInformation"]
            p.update(channel=str(c), channel_description=f"Channel {c}", time_point=f"{t:05d}",
                     image_id=f"image-{c}-{t}", image_size_vx={"width": 6, "height": 5, "depth": 4})
            a = p["acquisition"][0]
            a.update(channel=str(c), time_point=t, image_id=f"acq-{c}-{t}",
                     stack_scan_id=f"oc:default_st:0_ch:{c}_tp:{t}_pm:none",
                     time_stamps=[f"2026-08-18T09:17:{t:02d}.123-04:00"])
            p["stack_scan_ids"] = [a["stack_scan_id"]]
            # A processed file lists both cameras' acquisitions; both must survive.
            p["acquisition"].append({**copy.deepcopy(a), "camera": "right", "image_id": f"right-{c}-{t}"})
            with h5py.File(root / f"uni_tp-{t}_ch-{c}.lux.h5", "w") as f:
                f["Data"] = np.full((4, 5, 6), c + 10 * t, dtype=np.uint16)
                f["metadata"] = json.dumps(md)
    return root


@pytest.mark.parametrize("fmt", ["json", "txt", "csv"])
def test_three_channels_two_timepoints(three_channels, tmp_path, fmt):
    sources = sources_for(three_channels)
    assert len(sources) == 3
    plan = plan_export(sources, tmp_path / fmt, metadata_format=fmt, write_header=False)
    assert len(plan.jobs) == len(plan.sidecars) == 6
    summary = json.loads(run_export(plan).read_text())
    assert all(not layer["metadata_missing_fields"] for layer in summary["layers"])
    for job in plan.jobs:
        with h5py.File(job.output, "r") as f:
            embedded = json.loads(f["metadata"][()])
            p = embedded["processingInformation"]
            c, t = int(p["channel"]), job.timepoint
            assert np.all(f["Data"][()] == c + 10 * t)
        assert embedded == json.loads(job.sidecar.read_text(encoding="utf-8"))
        assert job.sidecar.name == job.output.name.removesuffix(".lux.h5") + ".json"
        assert p["time_point"] == f"{t:05d}"  # leading zeros kept
        assert p["image_id"] == f"image-{c}-{t}"
        assert [a["image_id"] for a in p["acquisition"]] == [f"acq-{c}-{t}", f"right-{c}-{t}"]
        assert all(a["time_point"] == str(t) for a in p["acquisition"])
        assert p["acquisition"][0]["time_stamps"] == [f"2026-08-18T13:17:{t:02d}.123000Z"]
        assert "imagingBranch" not in embedded  # null in the source
        if fmt != "json":
            extra = job.sidecar.with_suffix("." + fmt)
            assert extra in plan.output_paths and extra.is_file()
            assert "processingInformation.acquisition.1.image_id" in extra.read_text(encoding="utf-8")


def test_metadata_is_fixed_when_planning(three_channels, tmp_path):
    sources = sources_for(three_channels)
    plan = plan_export(sources, tmp_path / "out", timepoint_range=(1, 1), write_header=False)
    assert len(plan.jobs) == 3 and all(j.timepoint == 1 for j in plan.jobs)
    for src in sources:
        src.metadata_by_timepoint[1][0]["luxendo"]["processingInformation"]["image_id"] = "changed"
    run_export(plan)
    for job in plan.jobs:
        with h5py.File(job.output, "r") as f:
            assert json.loads(f["metadata"][()]) == json.loads(job.sidecar.read_text())
        assert job.metadata["processingInformation"]["image_id"] != "changed"


def test_frame_without_a_file_is_not_exported(three_channels, tmp_path):
    (three_channels / "uni_tp-1_ch-2.lux.h5").unlink()
    sources = sources_for(three_channels)
    assert all(s.data.shape[0] == 2 for s in sources)  # napari still shows it, as zeros
    plan = plan_export(sources, tmp_path / "out")
    assert len(plan.jobs) == len(plan.sidecars) == 5
    assert plan.header is None and "different timepoints" in plan.header_note


def test_duplicate_timepoints_rejected(three_channels, tmp_path):
    source = sources_for(three_channels)[0]
    source.timepoints = [0, 0]
    with pytest.raises(ValueError, match="duplicate timepoint"):
        plan_export([source], tmp_path / "out")


def test_mosaic_uses_each_frames_own_tiles(tiled_experiment, tmp_path):
    sources = _sources(read_luxendo(str(tiled_experiment / "main_raw.lux.h5"), tiles="mosaic"))
    for source in sources:
        assert len(source.metadata_by_timepoint[1]) == 6
        for t, entries in source.metadata_by_timepoint.items():
            for entry in entries:
                p = entry["luxendo"]["processingInformation"]
                p["acquisition"] = [{"image_id": f'{t}:{p["channel"]}:{p["stack"]}'}]
    plan = plan_export(sources, tmp_path / "out", timepoint_range=(1, 1))
    assert len(plan.jobs) == 2
    for job in plan.jobs:
        p = job.metadata["processingInformation"]
        assert len(p["acquisition"]) == 6
        assert all(a["image_id"].startswith("1:") for a in p["acquisition"])
        assert p["stack"].startswith("Fused-")
        assert len(job.sidecars) == 1 + 6  # the output's own, then one per tile
        tile_ids = [md["processingInformation"]["acquisition"][0]["image_id"]
                    for md, _path in job.sidecars[1:]]
        assert tile_ids == [a["image_id"] for a in p["acquisition"]]


@pytest.mark.parametrize("encoding", ["scalar", "uint8", "int8", "chars", "strings"])
def test_hdf5_json_encodings(tmp_path, encoding):
    raw = json.dumps(RAW_TILE).encode("utf-8")
    variants = {"scalar": raw, "uint8": np.frombuffer(raw + b"\0", dtype="u1"),
                "int8": np.frombuffer(raw, dtype="i1"), "chars": np.frombuffer(raw, dtype="S1"),
                "strings": np.array([raw], dtype=h5py.string_dtype())}
    with h5py.File(tmp_path / "metadata.h5", "w") as f:
        ds = f.create_dataset("metadata", data=variants[encoding])
        assert parse_metadata(ds)["raw"] == RAW_TILE


def test_cancel_keeps_finished_outputs_sidecars(three_channels, tmp_path):
    plan = plan_export(sources_for(three_channels), tmp_path / "out", metadata_format="csv")
    assert run_export(plan, cancel_check=lambda: plan.jobs[0].sidecar.exists()) is None
    assert plan.jobs[0].output.exists() and plan.jobs[0].sidecar.exists()
    assert plan.jobs[0].sidecar.with_suffix(".csv").exists()
    assert not plan.jobs[1].output.exists()
    assert not list(plan.output_dir.glob(".*"))


def test_failed_sidecar_keeps_previous_export(three_channels, tmp_path, monkeypatch):
    source = sources_for(three_channels)[0]
    plan = plan_export([source], tmp_path / "out", timepoint_range=(0, 0), write_header=False)
    run_export(plan)
    image, sidecar = plan.jobs[0].output.read_bytes(), plan.jobs[0].sidecar.read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(_export, "write_metadata_sidecar", fail)
    with pytest.raises(OSError, match="disk full"):
        run_export(plan)
    assert plan.jobs[0].output.read_bytes() == image
    assert plan.jobs[0].sidecar.read_bytes() == sidecar
    assert not list(plan.output_dir.glob(".*"))


def test_integer_timepoints_become_strings():
    raw = {"processingInformation": {"time_point": 3, "acquisition": [{"time_point": 3}]}}
    out = to_spec(raw)["processingInformation"]
    assert out["time_point"] == out["acquisition"][0]["time_point"] == "3"
    assert raw["processingInformation"]["time_point"] == 3


@pytest.mark.parametrize("pyramids", [True, False])
def test_lux_h5_requires_uint16(three_channels, tmp_path, pyramids):
    source = sources_for(three_channels)[0]
    source.data = source.data.astype(np.float32)
    with pytest.raises(ValueError, match="requires uint16"):
        plan_export([source], tmp_path / "out", write_pyramids=pyramids)
    assert plan_export([source], tmp_path / "out", fmt="tiff").jobs


def test_ims_uses_acquisition_times(three_channels, tmp_path):
    plan = plan_export(sources_for(three_channels), tmp_path / "out")
    run_export(plan)
    with h5py.File(plan.header, "r") as f:
        times = f["DataSetInfo/TimeInfo"].attrs
        for t in (0, 1):
            assert times[f"TimePoint{t + 1}"].tobytes().decode() == f"2026-08-18 13:17:{t:02d}.123"


def test_time_series_without_per_timepoint_metadata_refused(three_channels, tmp_path):
    source = sources_for(three_channels)[0]
    source.metadata_by_timepoint = None
    with pytest.raises(ValueError, match="supply metadata_by_timepoint"):
        plan_export([source], tmp_path / "out")


def test_source_transform_kept_without_layer_affine(three_channels):
    source = sources_for(three_channels)[0]
    source.affine_zyx = None
    original = source.metadata_by_timepoint[0][0]["luxendo"]["processingInformation"]["affine_to_sample"]
    assert export_metadata(source, 0)["processingInformation"]["affine_to_sample"] == original
