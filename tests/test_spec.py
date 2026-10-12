"""Exported metadata follows the Luxendo Image spec layout."""

import json

import numpy as np

from napari_luxendo._export import ExportSource, export_metadata
from napari_luxendo._lux import compose_affine_to_sample
from napari_luxendo._spec import (
    ACQUISITION_FIELDS,
    DERIVED_FIELDS,
    missing_fields,
    to_spec,
    utc_time_stamp,
)

# Trimmed from a raw LCS-SPIM tile: fields out of spec order, local-offset time
# stamps, no repetition / pm / channel_description, extra top-level blocks.
RAW_TILE = {
    "processingInformation": {
        "version": "1.0.0",
        "image_id": "2025-05-15T14:32:58.953Z-05a0176d",
        "sources": ["Luxendo LCS-SPIM, Embedded v, serial-nr: 50169"],
        "contains_beads": False,
        "time_point": "0",
        "channel": "3",
        "stack": "2-x00-y00",
        "objective": "bottom",
        "camera": "long",
        "stack_scan_ids": ["oc:default_st:2-x00-y00_ch:3_tp:0_pm:none"],
        "voxel_size_um": {"depth": 5.0, "width": 1.4625, "height": 1.4625},
        "image_size_vx": {"width": 2048, "height": 2048, "depth": 201},
        "affine_to_sample": [
            {"translation": [-1023.5, -1023.5, 0.0],
             "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            {"matrix": [[1.4625, 0.0, 0.0], [0.0, 1.4625, 0.0], [0.0, 0.0, -5.0]],
             "translation": [6480.6, -5451.0, 0.0]},
        ],
        "detection_directions": [[0.0, 0.0, 1.0]],
        "acquisition": [{
            "image_id": "2025-05-15T14:32:58.953Z-05a0176d",
            "microscope_type": "LCS-SPIM", "serial_number": "50169", "embedded_version": "",
            "edits": [], "contains_beads": False, "time_point": "0", "channel": "3",
            "stack": "2-x00-y00", "objective": "bottom", "camera": "long",
            "stack_scan_id": "oc:default_st:2-x00-y00_ch:3_tp:0_pm:none",
            "number_planes": 201,
            "stage_positions": [{"type": "linear", "name": "x", "start_um": -5451.0,
                                 "end_um": -5451.0,
                                 "movement": {"offset_um": [0, 0, 0], "direction": [0, -1, 0]}}],
            "image_plane_vectors": {"cam_top_to_bottom": [0, 1, 0], "cam_left_to_right": [1, 0, 0]},
            "refractive_index": 0.0,
            "detection": {"direction": [0, 0, 1], "wavelength_nm": 0.0, "numerical_aperture": 0.28,
                          "magnification": 4.44, "cam_pixel_size_um": {"height": 6.5, "width": 6.5},
                          "sensor_size_px": {"width": 2048, "height": 2048},
                          "crop_offset_px": {"top": 0, "left": 0},
                          "crop_size_px": {"width": 2048, "height": 2048}},
            "illuminations": [],
            "time_stamps": ["2025-05-15T10:32:58.953-04:00", "2025-05-15T10:32:59.021-04:00"],
        }],
    },
    "metaData": {"lasers": [{"name": "561 nm", "on": True}]},
    "cameraData": {"slit_size": 50},
    "imagingBranch": None,
}


def _subsequence(keys, order):
    return [k for k in order if k in keys] == [k for k in keys if k in order]


def test_raw_tile_to_spec_layout():
    out = to_spec(RAW_TILE)
    assert list(out) == ["processingInformation", "metaData", "cameraData"]  # null block dropped
    proc = out["processingInformation"]
    assert _subsequence(list(proc), DERIVED_FIELDS)
    assert list(proc["voxel_size_um"]) == ["width", "height", "depth"]
    assert all(list(t) == ["matrix", "translation"] for t in proc["affine_to_sample"])
    assert proc["repetition"] == 0 and "pm" not in proc  # pm:none means no photo-manipulation
    assert "channel_description" not in proc and "stack_description" not in proc  # not made up

    acq = proc["acquisition"][0]
    assert _subsequence(list(acq), ACQUISITION_FIELDS) and acq["repetition"] == 0
    assert list(acq["stage_positions"][0]) == ["name", "start_um", "end_um", "movement", "type"]
    assert list(acq["detection"])[-1] == "direction"
    assert acq["time_stamps"] == ["2025-05-15T14:32:58.953000Z", "2025-05-15T14:32:59.021000Z"]
    assert missing_fields(out) == []
    # The original values are untouched.
    assert RAW_TILE["processingInformation"]["acquisition"][0]["time_stamps"][0].endswith("-04:00")


def test_pm_and_gaps():
    md = {"processingInformation": {"stack_scan_ids": ["oc:default_st:0_ch:1_tp:3_pm:5"],
                                    "acquisition": [{"stack_scan_id": "oc:default_st:0_ch:1_tp:3_pm:5"}]}}
    proc = to_spec(md)["processingInformation"]
    assert proc["pm"] == "5" and proc["acquisition"][0]["pm"] == "5"
    gaps = missing_fields(to_spec(md))
    assert "processingInformation.stack" in gaps
    assert "processingInformation.acquisition.0.time_stamps" in gaps


def test_utc_time_stamp():
    assert utc_time_stamp("2021-04-19T14:45:38.073876Z") == "2021-04-19T14:45:38.073876Z"
    assert utc_time_stamp("2025-02-24T16:15:12.919-05:00") == "2025-02-24T21:15:12.919000Z"
    assert utc_time_stamp("not a time") == "not a time"
    assert utc_time_stamp("2025-02-24T16:15:12") == "2025-02-24T16:15:12"


def _tile_source(roi=None, fused=1, tiles=()):
    affine = compose_affine_to_sample(RAW_TILE["processingInformation"]["affine_to_sample"])
    return ExportSource(
        name="tile", data=np.zeros((1, 201, 64, 64), np.uint16), timepoints=[0], roi=roi,
        affine_zyx=affine, voxel_size_um=(5.0, 1.4625, 1.4625), metadata=RAW_TILE,
        fused_views=fused, tile_metadata=list(tiles),
    )


def test_export_keeps_source_transform_chain():
    proc = export_metadata(_tile_source(), 0)["processingInformation"]
    assert proc["affine_to_sample"] == to_spec(RAW_TILE)["processingInformation"]["affine_to_sample"]
    assert proc["image_size_vx"] == {"width": 64, "height": 64, "depth": 201}
    assert proc["sources"][0].startswith("Luxendo") and proc["sources"][-1].startswith("napari-luxendo")


def test_export_crop_is_applied_first():
    roi = (10, 20, 5, 30, 7, 40)
    src = _tile_source(roi=roi)
    chain = export_metadata(src, 0)["processingInformation"]["affine_to_sample"]
    assert chain[0]["type"] == "crop" and chain[0]["translation"] == [7.0, 5.0, 10.0]
    composed = compose_affine_to_sample(chain)
    np.testing.assert_allclose(composed @ [0, 0, 0, 1], src.affine_zyx @ [10, 5, 7, 1])


def test_export_mosaic_lists_every_tile_and_starts_with_voxel_size():
    other = json.loads(json.dumps(RAW_TILE))
    other["processingInformation"]["acquisition"][0]["stack"] = "2-x01-y00"
    other["processingInformation"]["stack_scan_ids"] = ["oc:default_st:2-x01-y00_ch:3_tp:0_pm:none"]
    src = _tile_source(fused=2, tiles=[("a", RAW_TILE), ("b", other)])
    proc = export_metadata(src, 0)["processingInformation"]
    assert [a["stack"] for a in proc["acquisition"]] == ["2-x00-y00", "2-x01-y00"]
    assert len(proc["stack_scan_ids"]) == 2 and "stack" not in proc
    chain = proc["affine_to_sample"]
    np.testing.assert_allclose(chain[0]["matrix"], np.diag([1.4625, 1.4625, 5.0]))
    np.testing.assert_allclose(compose_affine_to_sample(chain), src.affine_zyx, atol=1e-9)
    assert "processingInformation.stack" in missing_fields(export_metadata(src, 0))
