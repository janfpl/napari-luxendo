"""Export Luxendo layers to ``.lux.h5`` or BigTIFF.

Ported from Shifter's exporter. Each source is the full-resolution lazy array
of a layer (a single view, or a stitched mosaic), so what is written is what
napari shows. For every source and timepoint the volume is streamed in Z-slabs,
optionally cropped to a voxel ROI, and written as either

* a flat Luxendo ``.lux.h5`` file: ``Data``, regenerated ``Data_W_H_D`` pyramid
  levels (optional), and the source's JSON ``metadata`` with
  ``image_size_vx``, ``time_point`` and ``affine_to_sample`` rewritten for the
  exported grid, so the file reopens in the right place in sample space; or
* a BigTIFF with one page per Z-plane.

For ``.lux.h5`` an Imaris ``.ims`` header linking every exported channel and
timepoint can be written alongside.

This module does not import napari or Qt.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ._pyramids import (
    StreamingPyramidWriter,
    compute_pyramid_level_shape,
    level_factors,
    pyramid_backend_status,
)

logger = logging.getLogger(__name__)

FORMAT_LUX_H5 = "lux.h5"
FORMAT_TIFF = "tiff"

SUMMARY_FILENAME = "luxendo_export.json"
IMS_FILENAME = "luxendo_export.ims"

# (z_start, z_stop, y_start, y_stop, x_start, x_stop) in voxels, stops exclusive.
Roi = Tuple[int, int, int, int, int, int]
ProgressCallback = Callable[[int, int], None]
CancelCheck = Callable[[], bool]

# Default chunk edge of exported HDF5 datasets (Luxendo writes 64^3).
_H5_CHUNK = 64

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
    color: Optional[tuple[float, float, float]] = None
    fused_views: int = 1  # number of tiles stitched into the data
    source_files: list[str] = field(default_factory=list)

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

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.source.output_shape


@dataclass
class ExportPlan:
    """Validated description of an export, computed before anything is written."""

    fmt: str
    output_dir: Path
    write_pyramids: bool
    jobs: list[ExportJob]
    header: Optional[Path] = None  # .ims to write, if any
    header_note: str = ""  # why no header is written, when one was asked for

    @property
    def total_bytes(self) -> int:
        return sum(job.output_bytes for job in self.jobs)

    @property
    def output_paths(self) -> list[Path]:
        paths = [job.output for job in self.jobs]
        if self.header is not None:
            paths.append(self.header)
        return paths + [self.output_dir / SUMMARY_FILENAME]


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
        non-uint16 data with pyramids, or an output directory holding a source.
    """
    if fmt not in (FORMAT_LUX_H5, FORMAT_TIFF):
        raise ValueError(f"Unknown export format: {fmt!r}")
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
    ext = ".lux.h5" if fmt == FORMAT_LUX_H5 else ".tif"
    jobs: list[ExportJob] = []
    used_stems: set[str] = set()
    for src in sources:
        if src.data.ndim != 4 or len(src.timepoints) != src.data.shape[0]:
            raise ValueError(f"{src.name}: expected (T, Z, Y, X) data with one id per timepoint")
        if src.roi is not None:
            validate_roi(src.roi, src.shape_zyx)
        out_shape = src.output_shape
        levels = _job_levels(src, out_shape) if pyramids else []
        if levels and np.dtype(src.data.dtype) != np.uint16:
            raise ValueError(
                f"{src.name}: pyramid regeneration supports uint16 data only, not "
                f"{src.data.dtype}. Disable pyramid layers to export it."
            )

        stem = safe_filename(src.name)
        base, n = stem, 2
        while stem in used_stems:
            stem, n = f"{base}_{n}", n + 1
        used_stems.add(stem)

        itemsize = np.dtype(src.data.dtype).itemsize
        for frame, t in enumerate(src.timepoints):
            if timepoint_range is not None and not timepoint_range[0] <= t <= timepoint_range[1]:
                continue
            jobs.append(ExportJob(
                source=src,
                frame=frame,
                timepoint=t,
                output=output_dir / f"{stem}_tp-{t}{ext}",
                levels=levels,
                output_bytes=_output_bytes(out_shape, itemsize, levels),
            ))
    if not jobs:
        raise ValueError("No timepoint of the selected layers is in the chosen range.")

    plan = ExportPlan(fmt=fmt, output_dir=output_dir, write_pyramids=pyramids, jobs=jobs)
    if fmt == FORMAT_LUX_H5 and write_header:
        plan.header_note = _ims_header_problem(plan)
        if not plan.header_note:
            plan.header = output_dir / IMS_FILENAME
    return plan


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


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
    """The source's Luxendo metadata, rewritten for the exported volume."""
    meta = copy.deepcopy(source.metadata) if isinstance(source.metadata, dict) else {}
    proc = meta.get("processingInformation")
    if not isinstance(proc, dict):
        proc = meta["processingInformation"] = {}

    nz, ny, nx = source.output_shape
    proc["image_size_vx"] = {"width": nx, "height": ny, "depth": nz}
    proc["time_point"] = timepoint if isinstance(proc.get("time_point"), int) else str(timepoint)
    if source.voxel_size_um:
        z, y, x = source.voxel_size_um
        proc["voxel_size_um"] = {"width": x, "height": y, "depth": z}
    if source.affine_zyx is not None:
        affine = np.array(source.affine_zyx, dtype=float)
        if source.roi is not None:
            origin = np.array([source.roi[0], source.roi[2], source.roi[4]], dtype=float)
            affine[:3, 3] += affine[:3, :3] @ origin
        proc["affine_to_sample"] = [affine_to_luxendo(affine)]
    else:
        proc.pop("affine_to_sample", None)
    if source.fused_views > 1:
        # The reference tile's stack id no longer describes a fused mosaic.
        proc.pop("stack", None)
    return meta


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
        out.create_dataset("metadata", data=json.dumps(export_metadata(job.source, job.timepoint)))
        if pyramid is not None:
            summary = pyramid.finish()
            logger.info(
                "Pyramids for %s: %s (compute %.1fs, write %.1fs)", job.output.name,
                ",".join(summary["levels"]) or "-", summary["compute_s"], summary["write_s"],
            )


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
        is deleted; files already finished are kept.

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
    writer = _write_lux_h5 if plan.fmt == FORMAT_LUX_H5 else _write_tiff
    for job in plan.jobs:
        def _progress(done: int, _total: int, base: int = finished) -> None:
            if progress_callback:
                progress_callback(base + done, total)

        try:
            writer(job, _chunk_z(job, ram_percent), _progress, cancel_check)
        except _Cancelled:
            job.output.unlink(missing_ok=True)  # the writer has closed it by now
            logger.info("Export cancelled during %s", job.output.name)
            return None
        except BaseException:
            job.output.unlink(missing_ok=True)  # never leave a truncated file behind
            raise
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
            "outputs": [],
        })
        entry["outputs"].append({"timepoint": job.timepoint, "file": job.output.name})
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
    return path


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
        # Luxendo files carry no acquisition times; Imaris wants increasing ones.
        start = datetime(2000, 1, 1)
        for t in range(n_t):
            stamp = (start + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.000")
            info.attrs[f"TimePoint{t + 1}"] = _ims_text(stamp)
    tmp.replace(path)
    return path
