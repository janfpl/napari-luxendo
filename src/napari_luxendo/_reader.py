"""napari reader for Luxendo Image data.

Accepts flat ``.lux.h5`` files (one view each), nested / "main" files
(``main_raw.lux.h5`` ...) linking a whole experiment, and Imaris ``.ims`` /
BigDataViewer ``*.h5`` headers that link to ``.lux.h5`` files.

Every source is reduced to *views* (one tile / camera / channel at one
timepoint). Views are grouped into time series, series of the same channel are
stitched into one mosaic layer, and each layer is placed in sample space with
``affine_to_sample``.

Behaviour can be changed with keyword arguments to :func:`read_luxendo` or,
from the napari GUI, with environment variables:

``NAPARI_LUXENDO_TRANSFORM``  ``sample`` (default) or ``voxel``: place layers
    with ``affine_to_sample``, or only scale them by the voxel size.
``NAPARI_LUXENDO_TILES``  ``mosaic`` (default) or ``separate``: stitch the
    tiles of a channel into one layer, or give every tile its own layer.
``NAPARI_LUXENDO_VIEWS``  ``ask`` (default), ``raw`` or ``proc``: which views
    to load when a main file has both raw and processed ones.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import dask.array as da
import numpy as np

from ._headers import HeaderInfo, read_bdv_header, read_ims_header
from ._lux import (
    LuxVolume, is_lux_view, open_h5, open_lux_dataset, open_lux_group, open_lux_volume,
    parse_metadata,
)
from ._main import NestedView, is_nested_file, iter_nested_views, timepoint_index
from ._mosaic import MosaicLayout, build_mosaic_levels, plan_layout, same_linear
from ._preview import VolumeSource, preview_levels

logger = logging.getLogger(__name__)

LayerData = tuple[Any, dict[str, Any], str]

# Colormaps cycled through for channels with no color hint.
_CHANNEL_COLORMAPS = ("green", "magenta", "cyan", "yellow", "red", "blue")

# Contrast limits are estimated from at most this many voxels per sampled view.
_CONTRAST_SAMPLE_VOXELS = 4_000_000
_CONTRAST_MAX_VIEWS = 6


def napari_get_reader(path: str | list[str]) -> Callable[..., list[LayerData]] | None:
    """Return a reader if every path is a Luxendo file or header we can open."""
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if not paths or not all(_classify(Path(p)) for p in paths):
        return None
    return read_luxendo


def _classify(path: Path) -> str | None:
    """Return ``"lux"``, ``"nested"``, ``"ims"``, ``"bdv"`` or None for *path*."""
    name = path.name.lower()
    if not path.is_file() or not name.endswith((".h5", ".ims")):
        return None
    try:
        import h5py

        with h5py.File(str(path), "r") as f:
            if name.endswith(".ims"):
                return "ims" if read_ims_header(path, f) else None
            if is_lux_view(f):
                # "Data" alone is too common a name: a plain .h5 must also
                # carry Luxendo metadata. Fall through to the other checks.
                if _usable_data(f) and (name.endswith(".lux.h5") or _has_luxendo_metadata(f)):
                    return "lux"
                return None
            if is_nested_file(f):
                return "nested"
            if read_bdv_header(path, f):
                return "bdv"
    except Exception as exc:
        logger.debug("Not a Luxendo file %s: %s", path, exc)
    return None


def _usable_data(group: Any) -> bool:
    import h5py

    data = group.get("Data")
    return isinstance(data, h5py.Dataset) and data.ndim in (2, 3)


def _has_luxendo_metadata(group: Any) -> bool:
    """True if *group* has a ``metadata`` dataset with a ``processingInformation`` block."""
    import h5py

    meta = group.get("metadata")
    if not isinstance(meta, h5py.Dataset):
        return False
    raw = parse_metadata(meta).get("raw")
    return isinstance(raw, dict) and isinstance(raw.get("processingInformation"), dict)


# --------------------------------------------------------------------------- #
# Collecting views
# --------------------------------------------------------------------------- #


@dataclass
class _View:
    volume: LuxVolume
    timepoint: int
    series: tuple  # identifies one tile/camera/channel across time
    group: tuple  # tiles of one channel (candidates for one mosaic)
    color_key: str  # views sharing this share a colormap and contrast limits
    label: str  # layer name if the series gets its own layer
    group_label: str  # layer name for a mosaic of the group
    source: str  # the file that was opened (lux file, main file or header)
    color: Optional[tuple[float, float, float]] = None


def _option(value: str | None, env: str, default: str, allowed: tuple[str, ...]) -> str:
    value = value or os.environ.get(env, "") or default
    value = value.strip().lower()
    if value not in allowed:
        raise ValueError(f"{env}/{value!r}: expected one of {allowed}")
    return value


def read_luxendo(
    path: str | list[str],
    *,
    transform: str | None = None,
    tiles: str | None = None,
    views: str | None = None,
) -> list[LayerData]:
    """Read Luxendo files / headers into napari layer data.

    Parameters
    ----------
    transform : {"sample", "voxel"}
        Place layers with ``affine_to_sample`` (default) or by voxel size only.
    tiles : {"mosaic", "separate"}
        Stitch the tiles of each channel into one layer (default), or one layer
        per tile. Mosaics need ``transform="sample"``.
    views : {"ask", "raw", "proc"}
        Which views to load from a main file holding both raw and processed
        views. ``"ask"`` (default) shows a dialog inside napari and falls back
        to processed views elsewhere.
    """
    transform = _option(transform, "NAPARI_LUXENDO_TRANSFORM", "sample", ("sample", "voxel"))
    tiles = _option(tiles, "NAPARI_LUXENDO_TILES", "mosaic", ("mosaic", "separate"))
    views = _option(views, "NAPARI_LUXENDO_VIEWS", "ask", ("ask", "raw", "proc"))

    paths = [path] if isinstance(path, (str, Path)) else list(path)
    collected: list[_View] = []
    flat: list[LuxVolume] = []
    # Timepoints a main file or header lists, even if all their files are
    # missing, so the time axis keeps its frames in the right places.
    declared: set[int] = set()
    for p in map(Path, paths):
        kind = _classify(p)
        if kind == "lux":
            flat.append(open_lux_volume(p))
        elif kind == "nested":
            collected.extend(_views_from_nested(p, views, declared))
        elif kind in ("ims", "bdv"):
            collected.extend(_views_from_header(_read_header(p, kind), declared))
        else:
            raise ValueError(f"{p.name}: not a Luxendo .lux.h5 file or header")
    collected.extend(_views_from_flat(flat))
    if not collected:
        raise ValueError("No readable Luxendo views found.")
    return _build_layers(
        collected, transform=transform, mosaic=tiles == "mosaic", declared_timepoints=declared
    )


def _warn(message: str) -> None:
    """Surface a problem as a napari notification (and in the log)."""
    logger.warning(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def _views_from_flat(volumes: list[LuxVolume]) -> list[_View]:
    """Flat files: group by the view identity in their metadata."""
    out: list[_View] = []
    seen: dict[tuple, set[int]] = {}
    for i, vol in enumerate(volumes):
        md = vol.metadata
        identity = tuple(md.get(k) for k in ("channel", "stack", "objective", "camera"))
        t = _int_or(md.get("time_point"), 0)
        series = ("flat",) + identity if any(identity) else ("flat-file", str(vol.path))
        if t in seen.setdefault(series, set()):
            series = series + (i,)  # same view and timepoint twice: keep both
        seen.setdefault(series, set()).add(t)
        channel_label = md.get("channel_description") or (
            f"ch {md['channel']}" if "channel" in md else None
        )
        parts = [channel_label or vol.name]
        if md.get("stack"):
            parts.append(f"st:{md['stack']}")
        cam = "/".join(x for x in (md.get("objective"), md.get("camera")) if x)
        if cam and md.get("stack"):
            parts.append(cam)
        out.append(
            _View(
                volume=vol,
                timepoint=t,
                series=series,
                group=("flat", md.get("channel") or channel_label or str(vol.path),
                       md.get("objective"), md.get("camera")),
                color_key=md.get("channel") or channel_label or str(vol.path),
                label=" ".join(parts),
                group_label=" ".join([channel_label or vol.name] + ([cam] if cam else [])),
                source=str(vol.path),
            )
        )
    return out


def _views_from_nested(path: Path, choice: str, declared: set[int]) -> list[_View]:
    f = open_h5(path)
    nested = list(iter_nested_views(f))
    kinds = {v.kind for v in nested}
    if {"raw", "proc"} <= kinds:
        keep = _choose_raw_or_proc(path) if choice == "ask" else choice
        nested = [v for v in nested if v.kind in (keep, "other")]

    tp_names = sorted({v.timepoint for v in nested}, key=lambda n: timepoint_index(n, 0))
    tp_index = {name: timepoint_index(name, i) for i, name in enumerate(tp_names)}
    declared.update(tp_index.values())  # after the raw/proc choice, before skipping missing files
    out: list[_View] = []
    missing: list[str] = []
    for nv in nested:
        try:
            vol = open_lux_group(nv.group, path, view_name=nv.view)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        except ValueError as exc:
            logger.info("Skipping %s: %s", nv.group.name, exc)
            continue
        md = vol.metadata
        channel_label = md.get("channel_description") or f"channel_{nv.channel}"
        out.append(
            _View(
                volume=vol,
                timepoint=tp_index[nv.timepoint],
                series=("nested", str(path), nv.channel, nv.view),
                group=("nested", str(path), nv.channel, nv.kind),
                color_key=md.get("channel") or nv.channel,
                label=f"{channel_label} {nv.view}",
                group_label=f"{channel_label}" + (f" ({nv.kind})" if nv.kind != "other" else ""),
                source=str(path),
            )
        )
    if missing:
        _warn(f"{path.name}: {len(missing)} linked file(s) missing, e.g. {missing[0]}")
    return out


def _choose_raw_or_proc(path: Path) -> str:
    """Ask in napari whether to load raw or processed views."""
    try:
        from qtpy.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is not None:
            roles = getattr(QMessageBox, "ButtonRole", QMessageBox)  # Qt6 scoped enums
            box = QMessageBox()
            box.setWindowTitle("Luxendo: raw or processed?")
            box.setText(f"{path.name} contains both raw and processed views.\nWhich should be loaded?")
            proc = box.addButton("Processed", roles.AcceptRole)
            box.addButton("Raw", roles.RejectRole)
            box.setDefaultButton(proc)
            (box.exec() if hasattr(box, "exec") else box.exec_())
            return "proc" if box.clickedButton() is proc else "raw"
    except Exception as exc:  # no Qt available
        logger.debug("No Qt dialog: %s", exc)
    logger.info("%s has raw and processed views; loading processed.", path.name)
    return "proc"


def _read_header(path: Path, kind: str) -> HeaderInfo:
    import h5py

    with h5py.File(str(path), "r") as f:
        info = read_ims_header(path, f) if kind == "ims" else read_bdv_header(path, f)
    assert info is not None  # _classify already checked
    return info


def _views_from_header(header: HeaderInfo, declared: set[int]) -> list[_View]:
    out: list[_View] = []
    missing: list[str] = []
    for idx, ch in enumerate(header.channels):
        tps = ch.timepoints or list(range(len(ch.files)))
        dataset_paths = ch.dataset_paths or ["/Data"] * len(ch.files)
        declared.update(tps)
        for t, file, dataset_path in zip(tps, ch.files, dataset_paths):
            if not file.is_file():
                missing.append(str(file))
                continue
            vol = open_lux_dataset(file, dataset_path)
            if vol.voxel_size_um is None and header.voxel_size_um:
                vol.metadata["voxel_size_um"] = header.voxel_size_um
            channel = ch.channel if ch.channel is not None else str(idx)
            channel_label = ch.channel_name or vol.metadata.get("channel_description") or f"ch {channel}"
            out.append(
                _View(
                    volume=vol,
                    timepoint=t,
                    series=("header", str(header.path), idx),
                    group=("header", str(header.path), channel),
                    color_key=vol.metadata.get("channel") or channel,
                    label=ch.name or vol.name,
                    group_label=channel_label,
                    source=str(header.path),
                    color=ch.color,
                )
            )
    if missing:
        _warn(f"{header.path.name}: {len(missing)} linked file(s) missing, e.g. {missing[0]}")
    return out


def _int_or(value: Any, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Building layers
# --------------------------------------------------------------------------- #


@dataclass
class _Series:
    key: tuple
    by_time: dict[int, LuxVolume]
    first: _View

    @property
    def reference(self) -> LuxVolume:
        return self.by_time[min(self.by_time)]


def _build_layers(
    views: list[_View],
    *,
    transform: str,
    mosaic: bool,
    declared_timepoints: set[int] | None = None,
) -> list[LayerData]:
    series: OrderedDict[tuple, _Series] = OrderedDict()
    for v in views:
        s = series.setdefault(v.series, _Series(v.series, {}, v))
        s.by_time[v.timepoint] = v.volume
    timepoints = sorted({v.timepoint for v in views} | set(declared_timepoints or ()))
    use_affine = transform == "sample"

    # Group series into layers: one mosaic per channel group (when the tiles
    # share orientation and voxel size), otherwise one layer per series.
    groups: OrderedDict[tuple, list[_Series]] = OrderedDict()
    for s in series.values():
        groups.setdefault(s.first.group, []).append(s)

    plans: list[list[_Series]] = []
    for members in groups.values():
        if not (use_affine and mosaic and len(members) > 1):
            plans.extend([m] for m in members)
            continue
        buckets: list[list[_Series]] = []
        for m in members:
            ref = m.reference
            if ref.affine is None:
                buckets.append([m])
                continue
            for b in buckets:
                r0 = b[0].reference
                if r0.affine is not None and same_linear(r0.affine, ref.affine) \
                        and r0.dtype == ref.dtype and r0.data.ndim == ref.data.ndim:
                    b.append(m)
                    break
            else:
                buckets.append([m])
        plans.extend(buckets)

    color_keys = list(OrderedDict.fromkeys(s.first.color_key for s in series.values()))
    limits = {k: _contrast_for([s.reference for s in series.values() if s.first.color_key == k])
              for k in color_keys}
    multi = len(plans) > 1

    layers = []
    for members in plans:
        first = members[0].first
        if len(members) == 1:
            levels, native_count = _single_levels(members[0], timepoints)
            affine = members[0].reference.affine
            name = first.label
        else:
            levels, affine, native_count = _mosaic_levels(members, timepoints)
            name = f"{first.group_label} mosaic ({len(members)} tiles)"
        _check_drift(members)

        ref = members[0].reference
        has_time = len(timepoints) > 1
        kwargs: dict[str, Any] = {
            "name": name,
            "colormap": _pick_colormap(first, color_keys.index(first.color_key), len(color_keys) > 1),
            "blending": "additive" if multi else "translucent",
            "multiscale": len(levels) > 1,
            "metadata": {
                "luxendo": ref.metadata.get("raw", {}),
                "source": first.source,
                "path": str(ref.path),
                "files": sorted({str(v.path) for s in members for v in s.by_time.values()}),
                "views": [s.reference.view_name or s.reference.path.name for s in members],
                "timepoints": timepoints,
                "pyramid_levels": ref.level_names[:native_count],
                "voxel_size_um": ref.voxel_size_um,
                "placement": "affine_to_sample" if use_affine and affine is not None else "voxel_size",
                # Both placements, so the coordinates widget can switch between them.
                "placements": {
                    "sample": _with_time(affine, has_time) if affine is not None else None,
                    "camera": _with_time(camera_affine(affine, ref.voxel_size_um), has_time),
                },
            },
        }
        if use_affine and affine is not None:
            kwargs["affine"] = _with_time(affine, has_time)
        else:
            voxel = ref.voxel_size_um
            scale = list(voxel) if voxel else [1.0, 1.0, 1.0]
            kwargs["scale"] = [1.0, *scale] if has_time else scale
        if limits.get(first.color_key) is not None:
            kwargs["contrast_limits"] = limits[first.color_key]

        data = levels if len(levels) > 1 else levels[0]
        native_levels = len(kwargs['metadata']['pyramid_levels'])
        if len(levels) > native_levels:
            kwargs['metadata']['pyramid_levels'].extend(
                f'preview_nearest_{k}' for k in range(native_levels, len(levels)))
            kwargs['metadata']['display_pyramid'] = 'lazy nearest-neighbour preview; level 0 is full resolution'
        layers.append((data, kwargs, "image"))
    return layers


def camera_affine(affine: np.ndarray | None, voxel: Any) -> np.ndarray:
    """4x4 placement of the raw voxel grid as the camera recorded it.

    Only the voxel size is kept: no rotation, flip, shear or offset. The voxel
    size comes from the column lengths of *affine* when there is one (so it
    matches the sample placement), else from *voxel*, else 1 um.
    """
    if affine is not None:
        scale = np.linalg.norm(np.asarray(affine, dtype=float)[:3, :3], axis=0)
    elif voxel:
        scale = np.asarray(voxel, dtype=float)
    else:
        scale = np.ones(3)
    return np.diag([*scale, 1.0])


def _with_time(affine: np.ndarray, has_time: bool) -> np.ndarray:
    if not has_time:
        return affine
    out = np.eye(5)
    out[1:, 1:] = affine
    return out


def _compatible(v: LuxVolume, ref: LuxVolume) -> bool:
    """True if *v* can stand in for *ref* at another timepoint (same full-res grid)."""
    return v.shape == ref.shape and v.dtype == ref.dtype


def _common_level_count(members: list[_Series]) -> int:
    """Number of leading pyramid levels every usable timepoint of every series has.

    A level only counts if it has the reference's downsampling factors and
    shape: equal list positions alone do not mean equal resolution.
    Timepoints with an incompatible full-resolution grid are shown empty
    anyway, so they do not restrict the levels.
    """
    count = min(len(m.reference.levels) for m in members)
    for m in members:
        ref = m.reference
        for v in m.by_time.values():
            if not _compatible(v, ref):
                continue
            k = 0
            while (
                k < count
                and k < len(v.levels)
                and tuple(v.factors[k]) == tuple(ref.factors[k])
                and v.levels[k].shape == ref.levels[k].shape
            ):
                k += 1
            count = k
    return max(count, 1)


def _warn_dropped_levels(label: str, kept: int, available: int) -> None:
    if kept < available:
        _warn(
            f"{label}: pyramid levels differ between timepoints; using only the "
            f"{kept} level(s) all timepoints share."
        )


def _single_levels(s: _Series, timepoints: list[int]) -> tuple[list[da.Array], int]:
    """Display levels and native level count, stacked across timepoints."""
    ref = s.reference
    if len(timepoints) == 1:
        return (ref.levels + (preview_levels(VolumeSource(ref)) if len(ref.levels) == 1 else []),
                len(ref.levels))
    n_levels = _common_level_count([s])
    _warn_dropped_levels(s.first.label, n_levels, len(ref.levels))
    bad = [t for t, v in s.by_time.items() if not _compatible(v, ref)]
    if bad:
        _warn(f"{s.first.label}: timepoint(s) {bad} differ in shape or dtype and are left empty.")
    out = []
    for k in range(n_levels):
        frames = []
        for t in timepoints:
            v = s.by_time.get(t)
            if v is None or t in bad or v.levels[k].shape != ref.levels[k].shape:
                frames.append(da.zeros_like(ref.levels[k]))
            else:
                frames.append(v.levels[k])
        out.append(da.stack(frames))
    if n_levels == 1:
        reference_previews = preview_levels(VolumeSource(ref))
        per_t = []
        for t in timepoints:
            v = s.by_time.get(t)
            per_t.append([da.zeros_like(a) for a in reference_previews]
                         if v is None or t in bad else preview_levels(VolumeSource(v)))
        out.extend(da.stack([frame[k] for frame in per_t]) for k in range(len(reference_previews)))
    return out, n_levels


def _mosaic_levels(members: list[_Series], timepoints: list[int]) -> tuple[list[da.Array], np.ndarray, int]:
    layout: MosaicLayout = plan_layout([m.reference for m in members])
    if layout.residual_vx > 0.05:
        logger.info(
            "%s: tile positions rounded to the voxel grid (max %.2f voxel).",
            members[0].first.group_label, layout.residual_vx,
        )
    n_levels = min(len(layout.factors), _common_level_count(members))
    _warn_dropped_levels(members[0].first.group_label, n_levels, len(layout.factors))
    layout.factors = layout.factors[:n_levels]
    refs = [m.reference for m in members]
    per_t = []
    for t in timepoints:
        tiles = []
        for m, ref in zip(members, refs):
            v = m.by_time.get(t)
            if v is not None and not _compatible(v, ref):
                _warn(f"{m.first.label}: timepoint {t} has a different shape; left empty.")
                v = None
            tiles.append(v)
        per_t.append(build_mosaic_levels(layout, refs, tiles))
    if len(per_t) == 1:
        return per_t[0], layout.affine, n_levels
    n = min(len(levels) for levels in per_t)
    return [da.stack([levels[k] for levels in per_t]) for k in range(n)], layout.affine, n_levels


def _check_drift(members: list[_Series]) -> None:
    """Warn when a view's transform changes over time (the first one is used)."""
    for s in members:
        ref = s.reference.affine
        if ref is None:
            continue
        for t, v in s.by_time.items():
            if v.affine is not None and not np.allclose(v.affine, ref, rtol=1e-6, atol=1e-3):
                _warn(
                    f"{s.first.label}: affine_to_sample changes over time (first at "
                    f"timepoint {t}); all timepoints are placed with the first one."
                )
                break


# --------------------------------------------------------------------------- #
# Display hints
# --------------------------------------------------------------------------- #


def _pick_colormap(view: _View, index: int, multi: bool) -> Any:
    """Header color, else a color named by the channel, else a default."""
    if view.color is not None:
        return {
            "colors": [[0.0, 0.0, 0.0, 1.0], [*view.color, 1.0]],
            "name": f"luxendo-{'-'.join(f'{c:.3g}' for c in view.color)}",
        }
    md = view.volume.metadata
    for text in (md.get("channel_description"), view.group_label, view.label):
        hint = colormap_from_description(text or "")
        if hint:
            return hint
    return _CHANNEL_COLORMAPS[index % len(_CHANNEL_COLORMAPS)] if multi else "gray"


_COLOR_WORDS = ("green", "magenta", "cyan", "yellow", "red", "blue")


def colormap_from_description(text: str) -> str | None:
    """Guess a napari colormap from a channel description.

    An excitation/emission wavelength (e.g. ``"561"`` in ``"Red-561"``) wins
    over a color word; returns None when neither is present.
    """
    if not text:
        return None
    for m in re.finditer(r"(?<!\d)(\d{3})(?!\d)", text):
        nm = int(m.group(1))
        if 350 <= nm <= 800:
            if nm < 450:
                return "blue"
            if nm < 510:
                return "green"
            if nm < 550:
                return "yellow"
            if nm < 620:
                return "red"
            return "magenta"
    lower = text.lower()
    for word in _COLOR_WORDS:
        if re.search(rf"(?<![a-z]){word}(?![a-z])", lower):
            return word
    return None


def _contrast_for(volumes: list[LuxVolume]) -> list[float] | None:
    """Robust contrast limits pooled over a few views of one channel.

    Each sampled view contributes its coarsest level when small, otherwise a
    central block one HDF5 chunk deep, so loading never scans whole volumes.
    """
    if not volumes:
        return None
    step = max(1, len(volumes) // _CONTRAST_MAX_VIEWS)
    samples = []
    for vol in volumes[::step][:_CONTRAST_MAX_VIEWS]:
        try:
            samples.append(_sample(vol).ravel())
        except Exception as exc:
            logger.debug("Could not sample %s: %s", vol.path, exc)
    samples = [s for s in samples if s.size]
    if not samples:
        return None
    pooled = np.concatenate(samples)
    lo, hi = (float(v) for v in np.percentile(pooled, (0.05, 99.95)))
    if hi <= lo:
        lo, hi = float(pooled.min()), float(pooled.max())
    if hi <= lo:
        hi = lo + 1.0
    return [lo, hi]


def _sample(vol: LuxVolume) -> np.ndarray:
    k = len(vol.levels) - 1
    shape = vol.levels[k].shape
    if int(np.prod(shape)) <= _CONTRAST_SAMPLE_VOXELS:
        return vol.read(k, tuple(slice(0, n) for n in shape))
    ds = vol.datasets[k]
    depth = (ds.chunks[0] if ds.chunks else 1) if ds.ndim == 3 else 1
    half = int(np.sqrt(_CONTRAST_SAMPLE_VOXELS / depth) // 2)
    z0 = max(0, shape[0] // 2 - depth // 2)
    if ds.chunks and ds.ndim == 3:
        z0 -= z0 % ds.chunks[0]  # align to one HDF5 chunk row
    cy, cx = shape[1] // 2, shape[2] // 2
    region = (
        slice(z0, min(shape[0], z0 + depth)),
        slice(max(0, cy - half), min(shape[1], cy + half)),
        slice(max(0, cx - half), min(shape[2], cx + half)),
    )
    return vol.read(k, region)
