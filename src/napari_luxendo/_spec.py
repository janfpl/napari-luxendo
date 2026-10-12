"""Luxendo Image metadata in the layout of the published spec.

https://github.com/Luxendo/luxendo-image defines the ``processingInformation``
block of a ``.lux.h5`` file's JSON ``metadata``: which fields exist, their
order, and which are optional. Files in the wild differ from it (fields missing,
time stamps with a local UTC offset, extra top-level blocks), so metadata the
plugin writes is passed through :func:`to_spec` first.

Only what the spec itself defines or what follows from other fields is filled
in. Everything else is kept as found, and :func:`missing_fields` lists the
required fields that are still absent so they can be flagged rather than made
up. This module does not import napari or Qt.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Any

SPEC_VERSION = "1.0.0"

# Field order of the spec's example. Unknown fields are kept after these.
DERIVED_FIELDS = (
    "version", "image_id", "sources", "contains_beads", "time_point", "channel",
    "channel_description", "pm", "stack", "stack_description", "objective", "camera",
    "repetition", "stack_scan_ids", "voxel_size_um", "image_size_vx", "affine_to_sample",
    "detection_directions", "acquisition",
)
ACQUISITION_FIELDS = (
    "image_id", "microscope_type", "serial_number", "embedded_version", "edits",
    "contains_beads", "time_point", "channel", "channel_description", "pm", "stack",
    "stack_description", "objective", "camera", "repetition", "stack_scan_id",
    "number_planes", "stage_positions", "image_plane_vectors", "refractive_index",
    "detection", "illuminations", "time_stamps",
)
DETECTION_FIELDS = (
    "wavelength_nm", "numerical_aperture", "magnification", "cam_pixel_size_um",
    "sensor_size_px", "crop_offset_px", "crop_size_px", "direction",
)
STAGE_FIELDS = ("name", "start_um", "end_um", "start_deg", "end_deg", "movement", "type")
MOVEMENT_FIELDS = ("direction", "offset_um")
TRANSFORM_FIELDS = ("matrix", "translation", "type")
ILLUMINATION_FIELDS = ("objective", "direction")
SIZE_FIELDS = ("width", "height", "depth")

# Fields the spec does not mark optional.
REQUIRED_DERIVED = (
    "version", "sources", "contains_beads", "time_point", "channel", "stack", "objective",
    "camera", "voxel_size_um", "image_size_vx", "affine_to_sample", "detection_directions",
    "acquisition",
)
REQUIRED_ACQUISITION = (
    "microscope_type", "serial_number", "embedded_version", "edits", "contains_beads",
    "time_point", "channel", "stack", "objective", "camera", "number_planes",
    "stage_positions", "image_plane_vectors", "refractive_index", "detection",
    "illuminations", "time_stamps",
)

_PM_RE = re.compile(r"(?:^|_)pm:([^_]*)")


def _ordered(d: Any, order: tuple[str, ...]) -> Any:
    """*d* with the keys in *order* first, then any others as they were."""
    if not isinstance(d, dict):
        return d
    out = {k: d[k] for k in order if k in d}
    out.update((k, v) for k, v in d.items() if k not in out)
    return out


def _pm_from_scan_id(scan_id: Any) -> str | None:
    """The photo-manipulation id in a ``stack_scan_id`` (``..._pm:5``), if one was set."""
    m = _PM_RE.search(scan_id) if isinstance(scan_id, str) else None
    if m is None or m.group(1).lower() in ("", "none", "null"):
        return None
    return m.group(1)


def utc_time_stamp(value: Any) -> Any:
    """An ISO-8601 time stamp as the spec writes it: UTC, microseconds, ``Z``.

    ``2025-02-24T16:15:12.919-05:00`` becomes ``2025-02-24T21:15:12.919000Z``
    (the same instant). Stamps without an offset, or that don't parse, are kept.
    """
    if not isinstance(value, str):
        return value
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if t.tzinfo is None:
        return value
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _acquisition(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry
    e = dict(entry)
    if "pm" not in e:
        pm = _pm_from_scan_id(e.get("stack_scan_id"))
        if pm is not None:
            e["pm"] = pm
    e.setdefault("repetition", 0)  # the spec's default when missing
    if isinstance(e.get("stage_positions"), list):
        e["stage_positions"] = [
            _ordered({**s, "movement": _ordered(s["movement"], MOVEMENT_FIELDS)}
                     if isinstance(s, dict) and isinstance(s.get("movement"), dict) else s,
                     STAGE_FIELDS)
            for s in e["stage_positions"]
        ]
    if isinstance(e.get("detection"), dict):
        det = dict(e["detection"])
        for key in ("cam_pixel_size_um", "sensor_size_px", "crop_size_px"):
            if key in det:
                det[key] = _ordered(det[key], SIZE_FIELDS)
        if "crop_offset_px" in det:
            det["crop_offset_px"] = _ordered(det["crop_offset_px"], ("left", "top"))
        e["detection"] = _ordered(det, DETECTION_FIELDS)
    if isinstance(e.get("illuminations"), list):
        e["illuminations"] = [_ordered(i, ILLUMINATION_FIELDS) for i in e["illuminations"]]
    if isinstance(e.get("image_plane_vectors"), dict):
        e["image_plane_vectors"] = _ordered(
            e["image_plane_vectors"], ("cam_left_to_right", "cam_top_to_bottom"))
    if isinstance(e.get("time_stamps"), list):
        e["time_stamps"] = [utc_time_stamp(t) for t in e["time_stamps"]]
    return _ordered(e, ACQUISITION_FIELDS)


def processing_information(proc: dict[str, Any]) -> dict[str, Any]:
    """*proc* (a ``processingInformation`` block) in the spec's layout."""
    p = copy.deepcopy(proc)
    p.setdefault("version", SPEC_VERSION)
    acq = p.get("acquisition")
    if isinstance(acq, dict):
        acq = p["acquisition"] = [acq]
    if isinstance(acq, list):
        acq = p["acquisition"] = [_acquisition(a) for a in acq]
        entries = [a for a in acq if isinstance(a, dict)]
        if "stack_scan_ids" not in p:
            ids = list(dict.fromkeys(a["stack_scan_id"] for a in entries
                                     if isinstance(a.get("stack_scan_id"), str)))
            if ids:
                p["stack_scan_ids"] = ids
        if "repetition" not in p:
            reps = [a.get("repetition") for a in entries]
            p["repetition"] = max((r for r in reps if isinstance(r, int)), default=0)
    else:
        p.setdefault("repetition", 0)
    if "pm" not in p:
        pms = {_pm_from_scan_id(s) for s in p.get("stack_scan_ids") or []} - {None}
        if len(pms) == 1:
            p["pm"] = pms.pop()
    for key in ("voxel_size_um", "image_size_vx"):
        if isinstance(p.get(key), dict):
            p[key] = _ordered(p[key], SIZE_FIELDS)
    tr = p.get("affine_to_sample")
    if isinstance(tr, dict):
        tr = p["affine_to_sample"] = [tr]
    if isinstance(tr, list):
        p["affine_to_sample"] = [_ordered(t, TRANSFORM_FIELDS) for t in tr]
    return _ordered(p, DERIVED_FIELDS)


def to_spec(metadata: dict[str, Any], keep_extra: bool = True) -> dict[str, Any]:
    """A Luxendo ``metadata`` document in the spec's layout.

    ``processingInformation`` comes first, in the spec's field order. Other
    top-level blocks some files carry (``metaData``, ``cameraData``,
    ``imagingBranch``) are not part of the spec: they follow it unchanged when
    *keep_extra* and hold data, and are dropped otherwise.
    """
    proc = metadata.get("processingInformation") if isinstance(metadata, dict) else None
    out: dict[str, Any] = {
        "processingInformation": processing_information(proc if isinstance(proc, dict) else {})
    }
    if keep_extra and isinstance(metadata, dict):
        out.update((k, copy.deepcopy(v)) for k, v in metadata.items()
                   if k != "processingInformation" and v is not None)
    return out


def missing_fields(metadata: dict[str, Any]) -> list[str]:
    """Required spec fields absent from *metadata*, as dotted paths."""
    proc = metadata.get("processingInformation") if isinstance(metadata, dict) else None
    if not isinstance(proc, dict):
        return ["processingInformation"]
    missing = [f"processingInformation.{k}" for k in REQUIRED_DERIVED if k not in proc]
    for i, a in enumerate(proc.get("acquisition") or []):
        if isinstance(a, dict):
            missing += [f"processingInformation.acquisition.{i}.{k}"
                        for k in REQUIRED_ACQUISITION if k not in a]
    return missing
