"""Export Luxendo layers to ``.lux.h5``, BigTIFF, OME-TIFF or JPEG.

Ported from Shifter's exporter. Each source is the full-resolution lazy array
of a layer (a single view, or a stitched mosaic), so what is written is what
napari shows. For every source and timepoint the volume is streamed in Z-slabs,
optionally cropped to a voxel ROI, and written as either

* a flat Luxendo ``.lux.h5`` file: ``Data``, regenerated ``Data_W_H_D`` pyramid
  levels (optional), and the source's JSON ``metadata`` with
  ``image_size_vx``, ``time_point`` and ``affine_to_sample`` rewritten for the
  exported grid, so the file reopens in the right place in sample space; or
* a BigTIFF with one page per Z-plane;
* an OME-TIFF (BigTIFF with OME-XML) with one page per Z-plane, carrying the
  voxel size, layer name and colour; or
* a folder of 8-bit JPEGs, one per Z-plane, scaled with the layer's contrast
  limits.

For ``.lux.h5`` an Imaris ``.ims`` header linking every exported channel and
timepoint can be written alongside.

Every exported volume gets a ``<layer>_tp-<id>.json`` sidecar next to it,
optionally with a TXT or CSV copy: the same metadata a ``.lux.h5`` export
embeds, built from the JSON of that timepoint's source view(s). A stitched
mosaic also gets ``<layer>_tp-<id>_<tile>.json`` per tile, with that tile's own
metadata. All exported metadata follows the Luxendo Image spec layout
(:mod:`._spec`).

This module does not import napari or Qt.
"""

from __future__ import annotations

import copy
import os
import json
import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ._spec import missing_fields, to_spec
from ._pyramids import (
    StreamingPyramidWriter,
    compute_pyramid_level_shape,
    level_factors,
    pyramid_backend_status,
)

logger = logging.getLogger(__name__)

FORMAT_LUX_H5 = "lux.h5"
FORMAT_TIFF = "tiff"
FORMAT_OME_TIFF = "ome.tiff"
FORMAT_JPEG = "jpeg"
FORMATS = (FORMAT_LUX_H5, FORMAT_TIFF, FORMAT_OME_TIFF, FORMAT_JPEG)
_EXTENSIONS = {
    FORMAT_LUX_H5: ".lux.h5",
    FORMAT_TIFF: ".tif",
    FORMAT_OME_TIFF: ".ome.tif",
    FORMAT_JPEG: "",  # a folder of planes
}

SUMMARY_FILENAME = "luxendo_export.json"
SIDECAR_SUFFIX = ".json"  # next to <name>.lux.h5, as Luxendo writes <name>.json

# Extra copies of the JSON summary and metadata sidecars, flattened to one ``key, value`` per line.
METADATA_JSON = "json"  # the JSON summary only
METADATA_TXT = "txt"
METADATA_CSV = "csv"
METADATA_FORMATS = (METADATA_JSON, METADATA_TXT, METADATA_CSV)
IMS_FILENAME = "luxendo_export.ims"

# (z_start, z_stop, y_start, y_stop, x_start, x_stop) in voxels, stops exclusive.
Roi = Tuple[int, int, int, int, int, int]
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]

# Default chunk edge of exported HDF5 datasets (Luxendo writes 64^3).
_H5_CHUNK = 64

JPEG_QUALITY = 95
# Largest width or height a baseline JPEG can hold.
_JPEG_MAX_EDGE = 65500

# --- Chunk-size policy ----------------------------------------------------- #
# Reading a slab and writing it (plus the pyramid working buffers) transiently
# holds a few full-size copies of it in RAM; budget for that many copies.
_SLAB_PEAK_COPIES = 3

# Hard cap on a single slab's bytes, independent of installed RAM. Export is
# disk-bound, so larger slabs are not faster, only a bigger memory footprint.
_MAX_SLAB_BYTES = 4 * 1024**3


def compute_chunk_size(xy_shape: tuple[int, int], ram_percent: int = 90,
                       bytes_per_voxel: int = 2) -> int:
    """Number of Z-planes to process per slab.

    Bounded by *ram_percent* of the currently *available* system memory and by
    an absolute per-slab cap (:data:`_MAX_SLAB_BYTES`).
    """
    import psutil

    budget = int(psutil.virtual_memory().available * ram_percent / 100)
    plane_bytes = max(1, xy_shape[0] * xy_shape[1] * bytes_per_voxel)
    by_ram = budget // (plane_bytes * _SLAB_PEAK_COPIES)
    by_cap = _MAX_SLAB_BYTES // plane_bytes
    return max(1, int(min(by_ram, by_cap)))


# --------------------------------------------------------------------------- #
# Sources and plans
# --------------------------------------------------------------------------- #


@dataclass
class ExportSource:
    """One layer to export (built from a napari layer by :mod:`._export_layers`)."""

    name: str
    data: Any  # full-resolution lazy array, (T, Z, Y, X)
    timepoints: list[int]  # timepoint id of each T frame
    roi: Optional[Roi] = None  # voxel crop of the full-resolution grid
    pyramid_levels: list[str] = field(default_factory=list)  # Data_W_H_D to regenerate
    affine_zyx: Optional[np.ndarray] = None  # 4x4 voxel -> sample transform
    voxel_size_um: Optional[tuple[float, float, float]] = None  # (z, y, x)
    metadata: dict[str, Any] = field(default_factory=dict)  # Luxendo JSON of the reference view
    # (tile name, Luxendo JSON) of every tile stitched into a mosaic; empty for one view.
    tile_metadata: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    color: Optional[tuple[float, float, float]] = None
    contrast_limits: Optional[tuple[float, float]] = None  # 8-bit scaling for JPEG
    fused_views: int = 1  # number of tiles stitched into the data
    source_files: list[str] = field(default_factory=list)
    # The reader's per-timepoint views: {t: [{"view", "file", "luxendo"}]}. None for
    # sources built without it; an empty list marks a frame with no source file.
    metadata_by_timepoint: Optional[dict[int, list[dict[str, Any]]]] = None

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape[-3:])  # type: ignore[return-value]

    @property
    def output_shape(self) -> tuple[int, int, int]:
        if self.roi is None:
            return self.shape_zyx
        z0, z1, y0, y1, x0, x1 = self.roi
        return (z1 - z0, y1 - y0, x1 - x0)


@dataclass
class ExportJob:
    """One output file: one source at one timepoint."""

    source: ExportSource
    frame: int  # index along the source's T axis
    timepoint: int
    output: Path
    levels: list[tuple[str, int, int, int]]  # (name, factor_w, factor_h, factor_d)
    output_bytes: int
    metadata: Optional[dict[str, Any]] = None  # embedded in .lux.h5, built once when planning
    # (metadata, path) of the JSON sidecars published together with this output.
    sidecars: list[tuple[dict[str, Any], Path]] = field(default_factory=list)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.source.output_shape

    @property
    def sidecar(self) -> Optional[Path]:
        """The output's own JSON sidecar (per-tile ones follow it in :attr:`sidecars`)."""
        return self.sidecars[0][1] if self.sidecars else None


@dataclass
class ExportPlan:
    """Validated description of an export, computed before anything is written."""

    fmt: str
    output_dir: Path
    write_pyramids: bool
    jobs: list[ExportJob]
    header: Optional[Path] = None  # .ims to write, if any
    header_note: str = ""  # why no header is written, when one was asked for
    metadata_format: str = METADATA_JSON  # extra TXT / CSV copy of the summary and sidecars
    # Source Luxendo metadata to write as JSON sidecars: (source, metadata, sidecar path).
    sidecars: list[tuple[ExportSource, dict[str, Any], Path]] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(job.output_bytes for job in self.jobs)

    @property
    def output_paths(self) -> list[Path]:
        paths = [job.output for job in self.jobs]
        if self.header is not None:
            paths.append(self.header)
        for json_path in [self.output_dir / SUMMARY_FILENAME, *(p for _s, _m, p in self.sidecars)]:
            paths.append(json_path)
            if self.metadata_format != METADATA_JSON:
                paths.append(json_path.with_suffix(f".{self.metadata_format}"))
        return paths


def safe_filename(name: str) -> str:
    """*name* reduced to characters that are safe in a filename on every OS."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "layer"


def validate_roi(roi: Roi, shape: tuple[int, int, int]) -> None:
    """Raise ValueError unless *roi* is a non-empty box inside *shape*."""
    bounds = ((roi[0], roi[1], shape[0]), (roi[2], roi[3], shape[1]), (roi[4], roi[5], shape[2]))
    for axis, (start, stop, size) in zip("ZYX", bounds):
        if not 0 <= start < stop <= size:
            raise ValueError(
                f"ROI {axis} range [{start}, {stop}) is empty or outside the volume (size {size})."
            )


def _job_levels(source: ExportSource, out_shape: tuple[int, int, int]) -> list:
    levels = []
    for name in source.pyramid_levels:
        factors = level_factors(name)
        if factors is None:
            continue
        if all(n > 0 for n in compute_pyramid_level_shape(out_shape, *factors)):
            levels.append((name, *factors))
    return levels


def _output_bytes(shape: tuple[int, int, int], itemsize: int, levels: list) -> int:
    total = int(np.prod(shape)) * itemsize
    for _name, fw, fh, fd in levels:
        total += int(np.prod(compute_pyramid_level_shape(shape, fw, fh, fd))) * 2
    return total


def plan_export(
    sources: list[ExportSource],
    output_dir: Path | str,
    fmt: str = FORMAT_LUX_H5,
    timepoint_range: Optional[tuple[int, int]] = None,
    write_pyramids: bool = True,
    write_header: bool = True,
    metadata_format: str = METADATA_JSON,
) -> ExportPlan:
    """Validate an export request and list the files it will write.

    Parameters
    ----------
    timepoint_range : (first, last), optional
        Inclusive range of timepoint ids to export; all when None.

    Raises
    ------
    ValueError
        For an unknown format, an ROI outside a volume, no timepoint in range,
        duplicate timepoint ids, non-uint16 data for ``.lux.h5``, or an output
        directory holding a source.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown export format: {fmt!r}")
    if metadata_format not in METADATA_FORMATS:
        raise ValueError(f"Unknown metadata format: {metadata_format!r}")
    if not sources:
        raise ValueError("No layers selected for export.")

    output_dir = Path(output_dir)
    if output_dir.exists():
        out = output_dir.resolve()
        for src in sources:
            for f in src.source_files:
                if Path(f).resolve().parent == out:
                    raise ValueError(
                        "Choose an output directory that does not hold the source files "
                        f"({Path(f).name} is in {out})."
                    )

    pyramids = write_pyramids and fmt == FORMAT_LUX_H5
    ext = _EXTENSIONS[fmt]
    jobs: list[ExportJob] = []
    sidecars: list[tuple[ExportSource, dict[str, Any], Path]] = []
    used_stems: set[str] = set()
    for src in sources:
        if src.data.ndim != 4 or len(src.timepoints) != src.data.shape[0]:
            raise ValueError(f"{src.name}: expected (T, Z, Y, X) data with one id per timepoint")
        if len(set(src.timepoints)) != len(src.timepoints):
            raise ValueError(f"{src.name}: duplicate timepoint ids would overwrite an output")
        if src.roi is not None:
            validate_roi(src.roi, src.shape_zyx)
        out_shape = src.output_shape
        levels = _job_levels(src, out_shape) if pyramids else []
        # The Luxendo Image spec stores Data as uint16.
        if fmt == FORMAT_LUX_H5 and np.dtype(src.data.dtype) != np.uint16:
            raise ValueError(
                f"{src.name}: Luxendo Image (.lux.h5) requires uint16 data, not "
                f"{src.data.dtype}. Export it as TIFF to keep its data type."
            )

        if fmt == FORMAT_JPEG and max(out_shape[1:]) > _JPEG_MAX_EDGE:
            raise ValueError(
                f"{src.name}: JPEG images can be at most {_JPEG_MAX_EDGE} pixels wide or high, "
                f"but this export is {out_shape[2]} x {out_shape[1]}. Export an ROI or "
                "choose another format."
            )

        stem = safe_filename(src.name)
        base, n = stem, 2
        # Case-insensitive: Windows and macOS would write "A" and "a" to one file.
        while stem.casefold() in used_stems:
            stem, n = f"{base}_{n}", n + 1
        used_stems.add(stem.casefold())

        itemsize = 1 if fmt == FORMAT_JPEG else np.dtype(src.data.dtype).itemsize
        for frame, t in enumerate(src.timepoints):
            if timepoint_range is not None and not timepoint_range[0] <= t <= timepoint_range[1]:
                continue
            if src.metadata_by_timepoint is not None and not src.metadata_by_timepoint.get(t):
                # The reader shows a frame with no source file as zeros; there
                # is no acquired image to export for it.
                continue
            metadata = export_metadata(src, t)
            job = ExportJob(
                source=src,
                frame=frame,
                timepoint=t,
                output=output_dir / f"{stem}_tp-{t}{ext}",
                levels=levels,
                output_bytes=_output_bytes(out_shape, itemsize, levels),
                metadata=metadata,
                sidecars=_sidecars(src, t, output_dir / f"{stem}_tp-{t}", metadata),
            )
            jobs.append(job)
            sidecars += [(src, md, path) for md, path in job.sidecars]
    if not jobs:
        raise ValueError("No timepoint of the selected layers is in the chosen range.")

    plan = ExportPlan(fmt=fmt, output_dir=output_dir, write_pyramids=pyramids, jobs=jobs,
                      metadata_format=metadata_format, sidecars=sidecars)
    if fmt == FORMAT_LUX_H5 and write_header:
        plan.header_note = _ims_header_problem(plan)
        if not plan.header_note:
            plan.header = output_dir / IMS_FILENAME
    return plan


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


def _sidecars(src: ExportSource, timepoint: int, base: Path,
              metadata: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    """``(metadata, path)`` of the JSON sidecars for one output, named after *base*.

    ``<base>.json`` holds the output's own metadata (as embedded in a
    ``.lux.h5``). A stitched mosaic adds ``<base>_<tile>.json`` per tile,
    holding that tile's own metadata in the spec layout.
    """
    out = [(metadata, base.with_name(base.name + SIDECAR_SUFFIX))]
    tiles = _tile_documents(src, timepoint)
    if src.fused_views > 1 and len(tiles) > 1:
        used: set[str] = set()
        for name, md in tiles:
            tile = stem = safe_filename(re.sub(r"(\.lux)?\.h5$", "", name, flags=re.IGNORECASE))
            n = 2
            while tile.casefold() in used:
                tile, n = f"{stem}_{n}", n + 1
            used.add(tile.casefold())
            out.append((md, base.with_name(f"{base.name}_{tile}{SIDECAR_SUFFIX}")))
    return out


def _tile_documents(source: ExportSource, timepoint: int) -> list[tuple[str, dict[str, Any]]]:
    """``(view name, spec metadata)`` of every source view of one exported frame."""
    if source.metadata_by_timepoint is not None:
        return [(e.get("view") or Path(e.get("file", "")).name or "view",
                 to_spec(e.get("luxendo") or {}))
                for e in source.metadata_by_timepoint.get(timepoint) or []]
    if source.fused_views > 1 and source.tile_metadata:
        return [(name, to_spec(md if isinstance(md, dict) else {}))
                for name, md in source.tile_metadata]
    return [(source.name, to_spec(source.metadata if isinstance(source.metadata, dict) else {}))]


def _frame_metadata(source: ExportSource, timepoint: int) -> list[dict[str, Any]]:
    """Spec metadata of the source views that make up one exported frame.

    Without per-timepoint metadata, a time series whose reference view lists
    acquisitions is refused: those would describe the first frame only.
    """
    if source.metadata_by_timepoint is not None and not source.metadata_by_timepoint.get(timepoint):
        raise ValueError(f"{source.name}: no source file for timepoint {timepoint}")
    documents = [md for _name, md in _tile_documents(source, timepoint)]
    if (source.metadata_by_timepoint is None and len(source.timepoints) > 1
            and any(d["processingInformation"].get("acquisition") for d in documents)):
        raise ValueError(
            f"{source.name}: supply metadata_by_timepoint for a time series; the "
            "first frame's acquisition metadata cannot describe every timepoint."
        )
    return documents


def affine_to_luxendo(affine_zyx: np.ndarray) -> dict[str, list]:
    """One ``affine_to_sample`` entry (x, y, z order) for a napari 4x4 (z, y, x) affine.

    The inverse of :func:`napari_luxendo._lux.compose_affine_to_sample`.
    """
    a = np.asarray(affine_zyx, dtype=float)
    rev = np.eye(3)[::-1]
    return {
        "matrix": (rev @ a[:3, :3] @ rev).tolist(),
        "translation": a[:3, 3][::-1].tolist(),
    }


def export_metadata(source: ExportSource, timepoint: int) -> dict[str, Any]:
    """The source's Luxendo metadata, rewritten for the exported volume.

    The result follows the Luxendo Image spec (see :mod:`._spec`).
    ``image_size_vx``, ``voxel_size_um`` and ``time_point`` describe the
    exported grid. ``affine_to_sample`` keeps the source's transform chain when
    the layer sits where that chain puts it, with a crop offset applied first
    for an ROI; otherwise it is the layer's placement split into the voxel-size
    scaling the spec asks for first, and the rest. A stitched mosaic lists every
    tile's ``acquisition`` entries, as the spec does for fused images.
    """
    from . import __version__
    from ._lux import compose_affine_to_sample

    frame = _frame_metadata(source, timepoint)
    meta = copy.deepcopy(frame[0])
    proc = meta.get("processingInformation")
    if not isinstance(proc, dict):
        proc = meta["processingInformation"] = {}

    nz, ny, nx = source.output_shape
    proc["image_size_vx"] = {"width": nx, "height": ny, "depth": nz}
    # Timepoint names are strings and may keep leading zeros ("00003").
    original = str(proc.get("time_point", ""))
    proc["time_point"] = (original if original.isascii() and original.isdigit()
                          and int(original) == timepoint else str(timepoint))
    if source.voxel_size_um:
        z, y, x = source.voxel_size_um
        proc["voxel_size_um"] = {"width": x, "height": y, "depth": z}
    software = f"napari-luxendo {__version__}"
    sources = proc.get("sources") if isinstance(proc.get("sources"), list) else []
    proc["sources"] = sources + ([software] if software not in sources else [])

    chain = proc.get("affine_to_sample")
    chain = [chain] if isinstance(chain, dict) else chain
    composed = compose_affine_to_sample(chain)
    affine = source.affine_zyx if source.affine_zyx is not None else composed
    if affine is None:
        proc.pop("affine_to_sample", None)
    elif (source.fused_views == 1 and composed is not None
          and np.allclose(composed, affine)):
        if source.roi is not None:
            crop = {"matrix": np.eye(3).tolist(),
                    "translation": [float(source.roi[4]), float(source.roi[2]), float(source.roi[0])],
                    "type": "crop"}
            chain = [crop, *chain]
        proc["affine_to_sample"] = chain
    else:
        affine = np.array(affine, dtype=float)
        if source.roi is not None:
            origin = np.array([source.roi[0], source.roi[2], source.roi[4]], dtype=float)
            affine[:3, 3] += affine[:3, :3] @ origin
        proc["affine_to_sample"] = _split_voxel_scaling(affine, source.voxel_size_um)

    if source.fused_views > 1:
        tiles = [md["processingInformation"] for md in frame]
        if tiles:
            acq = [a for t in tiles for a in (t.get("acquisition") or [])]
            if acq:
                proc["acquisition"] = acq
            ids = [i for t in tiles for i in (t.get("stack_scan_ids") or [])]
            if ids:
                proc["stack_scan_ids"] = list(dict.fromkeys(ids))
            dirs = [d for t in tiles for d in (t.get("detection_directions") or [])]
            if dirs:
                proc["detection_directions"] = [list(d) for d in dict.fromkeys(tuple(d) for d in dirs)]
            proc["contains_beads"] = any(bool(t.get("contains_beads")) for t in tiles)
            # A fused image still has a stack, objective and camera; the spec
            # labels mixed ones "Fused-..." (e.g. "Fused-left-right"). Each
            # tile's own identity stays in its acquisition entries.
            for key in ("stack", "objective", "camera"):
                values = list(dict.fromkeys(str(t[key]) for t in tiles if key in t))
                if values:
                    proc[key] = values[0] if len(values) == 1 else "Fused-" + "-".join(values)
            proc.pop("stack_description", None)
            merged = list(dict.fromkeys(x for t in tiles for x in (t.get("sources") or [])))
            proc["sources"] = merged + ([software] if software not in merged else [])
            proc["repetition"] = max((t.get("repetition", 0) for t in tiles), default=0)
            pms = {t.get("pm") for t in tiles}
            if len(pms) != 1 or None in pms:
                proc.pop("pm", None)
            else:
                proc["pm"] = pms.pop()
    return to_spec(meta)


def _split_voxel_scaling(affine_zyx: np.ndarray, voxel_zyx: Optional[tuple]) -> list[dict]:
    """``affine_to_sample`` for a napari affine: voxel-size scaling first, then the rest.

    The spec asks writers to start the chain with a diagonal matrix of
    ``voxel_size_um``, so viewers that only read the voxel size agree with
    those that apply the whole chain.
    """
    whole = affine_to_luxendo(affine_zyx)
    if not voxel_zyx or not all(v > 0 for v in voxel_zyx):
        return [whole]
    scale = np.diag([float(v) for v in voxel_zyx[::-1]])  # x, y, z
    rest = np.asarray(whole["matrix"], dtype=float) @ np.linalg.inv(scale)
    return [
        {"matrix": scale.tolist(), "translation": [0.0, 0.0, 0.0], "type": "voxel-size"},
        {"matrix": rest.tolist(), "translation": whole["translation"]},
    ]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


class _Cancelled(Exception):
    """Raised inside a writer when the user cancels."""


def _iter_slabs(job: ExportJob, chunk_z: int, cancel_check: Optional[CancelCheck]):
    """Yield ``(out_z_start, slab)`` for *job*; raise :class:`_Cancelled` on cancel."""
    src = job.source
    nz, ny, nx = src.shape_zyx
    z0, _, y0, y1, x0, x1 = src.roi if src.roi is not None else (0, nz, 0, ny, 0, nx)
    out_nz = job.output_shape[0]
    for start in range(0, out_nz, chunk_z):
        if cancel_check and cancel_check():
            raise _Cancelled
        stop = min(start + chunk_z, out_nz)
        block = src.data[job.frame, z0 + start:z0 + stop, y0:y1, x0:x1]
        if hasattr(block, "compute"):
            block = block.compute()
        yield start, np.ascontiguousarray(block)


def _chunk_z(job: ExportJob, ram_percent: int) -> int:
    """Slab depth for *job*, rounded down to whole source Z-chunks when possible."""
    _, ny, nx = job.output_shape
    chunk = compute_chunk_size((ny, nx), ram_percent, np.dtype(job.source.data.dtype).itemsize)
    chunks = getattr(job.source.data, "chunks", None)
    if chunks and len(chunks) == 4 and chunks[1]:
        zc = int(chunks[1][0])
        if chunk >= zc:
            chunk -= chunk % zc
    return max(1, chunk)


def _write_tiff(job: ExportJob, chunk_z: int, progress: ProgressCallback,
                cancel_check: Optional[CancelCheck]) -> None:
    import tifffile

    done = 0
    with tifffile.TiffWriter(str(job.output), bigtiff=True) as tw:
        for _start, slab in _iter_slabs(job, chunk_z, cancel_check):
            for plane in slab:
                tw.write(plane, photometric="minisblack", contiguous=True)
            done += slab.nbytes
            progress(done, job.output_bytes)


def _ome_color(rgb: tuple[float, float, float]) -> int:
    """OME ``Channel/@Color``: RGBA packed into a signed 32-bit integer."""
    r, g, b = (int(round(min(max(v, 0.0), 1.0) * 255)) for v in rgb)
    return int(np.array((r << 24) | (g << 16) | (b << 8) | 255, dtype=np.uint32).view(np.int32))


def ome_metadata(source: ExportSource) -> dict[str, Any]:
    """tifffile OME metadata for *source*: axes, name, voxel size and colour."""
    meta: dict[str, Any] = {"axes": "ZYX", "Name": source.name}
    if source.voxel_size_um:
        for axis, size in zip("ZYX", source.voxel_size_um):
            meta[f"PhysicalSize{axis}"] = float(size)
            meta[f"PhysicalSize{axis}Unit"] = "µm"
    channel: dict[str, Any] = {"Name": source.name}
    if source.color is not None:
        channel["Color"] = _ome_color(source.color)
    meta["Channel"] = channel
    return meta


def _write_ome_tiff(job: ExportJob, chunk_z: int, progress: ProgressCallback,
                    cancel_check: Optional[CancelCheck]) -> None:
    import tifffile

    def planes():
        done = 0
        for _start, slab in _iter_slabs(job, chunk_z, cancel_check):
            yield from slab
            done += slab.nbytes
            progress(done, job.output_bytes)

    with tifffile.TiffWriter(str(job.output), bigtiff=True, ome=True) as tw:
        tw.write(
            planes(), shape=job.output_shape, dtype=job.source.data.dtype,
            photometric="minisblack", metadata=ome_metadata(job.source),
        )


def jpeg_contrast_limits(source: ExportSource) -> tuple[float, float]:
    """Intensities mapped to 0 and 255 in a JPEG export of *source*."""
    if source.contrast_limits is not None:
        lo, hi = (float(v) for v in source.contrast_limits)
    else:
        dtype = np.dtype(source.data.dtype)
        lo, hi = (float(np.iinfo(dtype).min), float(np.iinfo(dtype).max)) \
            if dtype.kind in "ui" else (0.0, 1.0)
    return lo, hi if hi > lo else lo + 1.0


def to_uint8(plane: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """*plane* linearly scaled so *limits* map to 0 and 255, clipped."""
    lo, hi = limits
    scaled = (plane.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


_JPEG_PLANE = re.compile(r"z\d+\.jpg")


def jpeg_plane_paths(job: ExportJob) -> list[Path]:
    """Files a JPEG *job* writes: one per Z-plane in its output folder."""
    nz = job.output_shape[0]
    width = max(4, len(str(nz - 1)))
    return [job.output / f"z{z:0{width}d}.jpg" for z in range(nz)]


def _remove_jpeg_planes(folder: Path) -> None:
    """Delete the plane files of an earlier or partial JPEG export from *folder*."""
    if not folder.is_dir():
        return
    for f in folder.iterdir():
        if _JPEG_PLANE.fullmatch(f.name):
            f.unlink()
    if not any(folder.iterdir()):
        folder.rmdir()


def _write_jpeg(job: ExportJob, chunk_z: int, progress: ProgressCallback,
                cancel_check: Optional[CancelCheck]) -> None:
    from PIL import Image

    job.output.mkdir(parents=True, exist_ok=True)
    paths = jpeg_plane_paths(job)
    limits = jpeg_contrast_limits(job.source)
    done = 0
    for start, slab in _iter_slabs(job, chunk_z, cancel_check):
        for z, plane in enumerate(slab, start):
            Image.fromarray(to_uint8(plane, limits)).save(paths[z], quality=JPEG_QUALITY)
            done += plane.size
        progress(done, job.output_bytes)


def _remove_output(job: ExportJob) -> None:
    """Delete *job*'s (partial) output."""
    if job.output.is_dir():
        _remove_jpeg_planes(job.output)
    else:
        job.output.unlink(missing_ok=True)


def _publish(staged: Path, output: Path) -> None:
    """Move a finished *staged* file or JPEG folder to *output*, replacing an earlier export."""
    if not staged.is_dir():
        os.replace(staged, output)
        return
    # A folder can't replace a non-empty folder: swap its plane files instead,
    # keeping unrelated files and dropping planes of a deeper earlier export.
    _remove_jpeg_planes(output)
    output.mkdir(parents=True, exist_ok=True)
    for f in staged.iterdir():
        os.replace(f, output / f.name)
    staged.rmdir()


def _write_lux_h5(job: ExportJob, chunk_z: int, progress: ProgressCallback,
                  cancel_check: Optional[CancelCheck]) -> None:
    import h5py

    shape = job.output_shape
    with h5py.File(job.output, "w") as out:
        ds = out.create_dataset(
            "Data", shape=shape, dtype=job.source.data.dtype,
            chunks=tuple(min(_H5_CHUNK, n) for n in shape),
        )
        pyramid = StreamingPyramidWriter(out, job.levels, shape) if job.levels else None
        done = 0
        for start, slab in _iter_slabs(job, chunk_z, cancel_check):
            ds[start:start + slab.shape[0]] = slab
            done += slab.nbytes
            if pyramid is not None:
                done += pyramid.consume(slab, start)
            del slab  # release before the next (possibly multi-GiB) read
            progress(done, job.output_bytes)
        metadata = job.metadata if job.metadata is not None else export_metadata(job.source, job.timepoint)
        out.create_dataset("metadata", data=json.dumps(metadata))
        if pyramid is not None:
            summary = pyramid.finish()
            logger.info(
                "Pyramids for %s: %s (compute %.1fs, write %.1fs)", job.output.name,
                ",".join(summary["levels"]) or "-", summary["compute_s"], summary["write_s"],
            )


_WRITERS = {
    FORMAT_LUX_H5: _write_lux_h5,
    FORMAT_TIFF: _write_tiff,
    FORMAT_OME_TIFF: _write_ome_tiff,
    FORMAT_JPEG: _write_jpeg,
}


def run_export(
    plan: ExportPlan,
    ram_percent: int = 90,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> Optional[Path]:
    """Write every file of *plan*.

    Parameters
    ----------
    ram_percent : int
        Share of currently available RAM a Z-slab may use.
    progress_callback : callable, optional
        Called with ``(bytes_written, total_bytes)`` across all files.
    cancel_check : callable, optional
        Polled before each slab; return True to stop. The partially written file
        is deleted; files already finished are kept, and so is an earlier export
        the cancelled file would have replaced.

    Returns
    -------
    Path or None
        The JSON export summary, or None if cancelled.
    """
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    if plan.write_pyramids:
        logger.info("Pyramid reduction backend: %s", pyramid_backend_status())

    total = plan.total_bytes
    finished = 0
    writer = _WRITERS[plan.fmt]
    for job in plan.jobs:
        def _progress(done: int, _total: int, base: int = finished) -> None:
            if progress_callback:
                progress_callback(base + done, total)

        # Write next to the destination and swap it in only once complete, so a
        # cancelled or failed overwrite keeps the previous export.
        staged = replace(job, output=job.output.with_name(f".{uuid.uuid4().hex}.{job.output.name}"))
        # Its sidecars are staged too and published right after it, so a
        # finished output never lacks them and a failed one never gets them.
        staged_sidecars = [
            (md, path, path.with_name(f".{uuid.uuid4().hex}.{path.name}"))
            for md, path in job.sidecars
        ]
        suffixes = [SIDECAR_SUFFIX] + ([f".{plan.metadata_format}"]
                                       if plan.metadata_format != METADATA_JSON else [])
        try:
            writer(staged, _chunk_z(job, ram_percent), _progress, cancel_check)
            for md, _path, tmp in staged_sidecars:
                write_metadata_sidecar(md, tmp, plan.metadata_format)
            _publish(staged.output, job.output)
            for _md, path, tmp in staged_sidecars:
                for suffix in suffixes:
                    os.replace(tmp.with_suffix(suffix), path.with_suffix(suffix))
        except _Cancelled:
            _remove_output(staged)  # the writer has closed it by now
            logger.info("Export cancelled during %s", job.output.name)
            return None
        except BaseException:
            _remove_output(staged)  # never leave a truncated file behind
            raise
        finally:
            for _md, _path, tmp in staged_sidecars:
                for suffix in suffixes:
                    tmp.with_suffix(suffix).unlink(missing_ok=True)
        finished += job.output_bytes
        if progress_callback:
            progress_callback(finished, total)

    if plan.header is not None:
        write_ims_header(plan, plan.header)
    return _write_summary(plan)


def _write_summary(plan: ExportPlan) -> Path:
    from . import __version__

    layers: dict[int, dict[str, Any]] = {}
    for job in plan.jobs:
        src = job.source
        entry = layers.setdefault(id(src), {
            "layer": src.name,
            "source_files": src.source_files,
            "fused_views": src.fused_views,
            "roi_voxels": None if src.roi is None else dict(zip(
                ("z_start", "z_stop", "y_start", "y_stop", "x_start", "x_stop"),
                (int(v) for v in src.roi),
            )),
            "shape_zyx": list(src.output_shape),
            "pyramid_levels": [lvl[0] for lvl in job.levels],
            "metadata_sidecars": [p.name for s, _m, p in plan.sidecars if s is src],
            # Required Luxendo-spec fields the source metadata doesn't provide.
            "metadata_missing_fields": {p.name: gaps for s, m, p in plan.sidecars
                                        if s is src and (gaps := missing_fields(m))},
            "outputs": [],
        })
        if plan.fmt == FORMAT_JPEG:
            entry["contrast_limits"] = list(jpeg_contrast_limits(src))
        entry["outputs"].append({
            "timepoint": job.timepoint,
            "file": job.output.name,
            "metadata_sidecars": [path.name for _md, path in job.sidecars],
            "source_files": ([e["file"] for e in src.metadata_by_timepoint.get(job.timepoint, [])
                              if "file" in e] if src.metadata_by_timepoint is not None
                             else src.source_files),
        })
    summary = {
        "format": plan.fmt,
        "layers": list(layers.values()),
        "ims_header": plan.header.name if plan.header is not None else None,
        "bytes_written": plan.total_bytes,
        "export_date": datetime.now().isoformat(timespec="seconds"),
        "software": f"napari-luxendo {__version__}",
    }
    path = plan.output_dir / SUMMARY_FILENAME
    path.write_text(json.dumps(summary, indent=2))
    if plan.metadata_format != METADATA_JSON:
        convert_json_metadata(path, plan.metadata_format)
    return path


# --------------------------------------------------------------------------- #
# Metadata as TXT / CSV
# --------------------------------------------------------------------------- #


def flatten_json(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    """``(key, value)`` rows for every leaf of a JSON *value*.

    Keys are dotted paths, with list indices as numbers
    (``layers.0.outputs.1.file``). Empty lists and objects are kept as ``[]`` /
    ``{}``; ``null`` becomes an empty value.
    """
    if isinstance(value, dict) and value:
        items = value.items()
    elif isinstance(value, list) and value:
        items = enumerate(value)
    else:
        if isinstance(value, (dict, list)):
            text = json.dumps(value)
        elif value is None:
            text = ""
        elif isinstance(value, bool):
            text = json.dumps(value)
        else:
            text = str(value)
        return [(prefix, text)]
    rows: list[tuple[str, str]] = []
    for key, child in items:
        rows += flatten_json(child, f"{prefix}.{key}" if prefix else str(key))
    return rows


def write_metadata_sidecar(metadata: dict[str, Any], path: Path | str,
                           fmt: str = METADATA_JSON) -> Path:
    """Write Luxendo *metadata* (already in the spec layout) as the JSON sidecar *path*.

    With *fmt* ``txt`` or ``csv`` a flattened copy is written next to it (see
    :func:`convert_json_metadata`).
    """
    path = Path(path)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if fmt != METADATA_JSON:
        convert_json_metadata(path, fmt)
    return path


def convert_json_metadata(path: Path | str, fmt: str) -> Path:
    """Write the JSON file *path* as ``.txt`` or ``.csv`` next to it.

    Both hold one row per leaf of the JSON, as :func:`flatten_json` gives:
    ``key: value`` lines for TXT, ``key,value`` with a header row for CSV.
    """
    import csv

    if fmt not in (METADATA_TXT, METADATA_CSV):
        raise ValueError(f"Can't convert metadata to {fmt!r}")
    path = Path(path)
    rows = flatten_json(json.loads(path.read_text(encoding="utf-8")))
    out = path.with_suffix(f".{fmt}")
    with open(out, "w", encoding="utf-8", newline="") as f:
        if fmt == METADATA_CSV:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerows(rows)
        else:
            f.writelines(f"{key}: {value}\n" for key, value in rows)
    return out


# --------------------------------------------------------------------------- #
# Imaris header
# --------------------------------------------------------------------------- #


def _ims_header_problem(plan: ExportPlan) -> str:
    """Why *plan*'s outputs can't share one .ims header ("" if they can).

    Imaris needs every channel to have the same size, resolution levels and
    timepoints.
    """
    sources = list({id(j.source): j.source for j in plan.jobs}.values())
    if len({s.output_shape for s in sources}) > 1:
        return "the exported layers differ in size"
    by_source: dict[int, list[ExportJob]] = {}
    for job in plan.jobs:
        by_source.setdefault(id(job.source), []).append(job)
    if len({tuple(j.timepoint for j in jobs) for jobs in by_source.values()}) > 1:
        return "the exported layers have different timepoints"
    if len({tuple(lvl[0] for lvl in jobs[0].levels) for jobs in by_source.values()}) > 1:
        return "the exported layers have different pyramid levels"
    return ""


def _acquisition_times(job: ExportJob) -> list[datetime]:
    """The time-zone-aware acquisition time stamps in *job*'s metadata, in UTC."""
    metadata = job.metadata if job.metadata is not None else export_metadata(job.source, job.timepoint)
    times = []
    for acq in metadata.get("processingInformation", {}).get("acquisition") or []:
        for value in (acq.get("time_stamps") or []) if isinstance(acq, dict) else []:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is not None:
                times.append(parsed.astimezone(timezone.utc))
    return times


def _ims_text(value: Any) -> np.ndarray:
    """Imaris text attribute: an array of single characters."""
    return np.frombuffer(str(value).encode("ascii", "replace"), dtype="S1").copy()


def write_ims_header(plan: ExportPlan, path: Path) -> Path:
    """Write an Imaris header linking every exported channel and timepoint.

    Channels are the exported layers, in plan order; timepoints are numbered
    0..N-1 in the order exported. Holds no pixels: each level is an HDF5
    external link to ``Data`` / ``Data_W_H_D`` of an exported file.
    """
    import h5py

    by_source: dict[int, list[ExportJob]] = {}
    for job in plan.jobs:
        by_source.setdefault(id(job.source), []).append(job)
    channels = list(by_source.values())
    first = channels[0][0]
    nz, ny, nx = first.output_shape
    levels = [("Data", 1, 1, 1)] + list(first.levels)
    dtype_max = int(np.iinfo(first.source.data.dtype).max) \
        if np.dtype(first.source.data.dtype).kind in "ui" else 65535

    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        for key, value in (
            ("DataSetDirectoryName", "DataSet"),
            ("DataSetInfoDirectoryName", "DataSetInfo"),
            ("ImarisDataSet", "ImarisDataSet"),
            ("ImarisVersion", "5.5.0"),
            ("NumberOfDataSets", 1),
            ("ThumbnailDirectoryName", "Thumbnail"),
        ):
            f.attrs[key] = _ims_text(value)

        for r, (name, fw, fh, fd) in enumerate(levels):
            for t in range(len(channels[0])):
                for c, jobs in enumerate(channels):
                    g = f.create_group(f"DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}")
                    g["Data"] = h5py.ExternalLink(jobs[t].output.name, f"/{name}")
                    g.attrs["ImageSizeX"] = _ims_text(nx // fw)
                    g.attrs["ImageSizeY"] = _ims_text(ny // fh)
                    g.attrs["ImageSizeZ"] = _ims_text(nz // fd)
                    g.attrs["HistogramMin"] = _ims_text(0)
                    g.attrs["HistogramMax"] = _ims_text(dtype_max)

        img = f.create_group("DataSetInfo/Image")
        vz, vy, vx = first.source.voxel_size_um or (1.0, 1.0, 1.0)
        for axis, (dim, n, vox) in enumerate((("X", nx, vx), ("Y", ny, vy), ("Z", nz, vz))):
            img.attrs[dim] = _ims_text(n)
            img.attrs[f"ExtMin{axis}"] = _ims_text(0)
            img.attrs[f"ExtMax{axis}"] = _ims_text(f"{n * vox:.6g}")
        img.attrs["Unit"] = _ims_text("um")

        for c, jobs in enumerate(channels):
            src = jobs[0].source
            g = f.create_group(f"DataSetInfo/Channel {c}")
            g.attrs["Name"] = _ims_text(src.name)
            color = src.color or (1.0, 1.0, 1.0)
            g.attrs["Color"] = _ims_text(" ".join(f"{v:.3g}" for v in color))

        info = f.create_group("DataSetInfo/TimeInfo")
        n_t = len(channels[0])
        info.attrs["DatasetTimePoints"] = _ims_text(n_t)
        info.attrs["FileTimePoints"] = _ims_text(n_t)
        # The earliest acquisition time stamp of each exported frame (UTC);
        # frames without one fall back to increasing times from 2000-01-01.
        start = datetime(2000, 1, 1)
        for t in range(n_t):
            times = [stamp for jobs in channels for stamp in _acquisition_times(jobs[t])]
            first_time = min(times) if times else start + timedelta(seconds=t)
            stamp = first_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            info.attrs[f"TimePoint{t + 1}"] = _ims_text(stamp)
    tmp.replace(path)
    return path
