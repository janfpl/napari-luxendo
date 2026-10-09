"""Lazy access to Luxendo Image (``.lux.h5``) volumes.

A Luxendo view holds the full-resolution volume in a ``Data`` dataset,
optional downsampled copies named ``Data_W_H_D`` (integer X/Y/Z factors), and a
JSON ``metadata`` dataset whose ``processingInformation`` block carries the
voxel size, the view identity (time point, channel, stack, objective, camera)
and ``affine_to_sample``, the transform from voxel indices to sample space.

A view is either the root of a flat ``.lux.h5`` file or a group inside a
nested / "main" file; any of its members may be an external link into another
file. See https://github.com/Luxendo/luxendo-image for the format.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import dask.array as da
import numpy as np

from ._fastio import close_readers, reader_for

logger = logging.getLogger(__name__)

# Pyramid dataset names: Data_W_H_D (all integers).
_PYRAMID_RE = re.compile(r"^Data_(\d+)_(\d+)_(\d+)$")

# h5py File objects must stay open while dask arrays reference their datasets.
# They are kept here, keyed by resolved path, so opening the same file twice
# (e.g. from a main file and directly) reuses one handle.
_OPEN_FILES: dict[str, Any] = {}
_OPEN_LOCK = threading.Lock()


def open_h5(path: Path | str) -> Any:
    """Open (or reuse) a read-only h5py handle for *path*."""
    import h5py

    key = str(Path(path).resolve())
    with _OPEN_LOCK:
        f = _OPEN_FILES.get(key)
        if f is None or not f.id.valid:
            f = h5py.File(key, "r")
            _OPEN_FILES[key] = f
        return f


def close_all() -> None:
    """Close every file handle opened by this plugin.

    Any layer still backed by those files will fail to read afterwards, so
    only call this once the layers are gone.
    """
    close_readers()
    with _OPEN_LOCK:
        for f in _OPEN_FILES.values():
            try:
                f.close()
            except Exception:
                pass
        _OPEN_FILES.clear()


def resolve_link_target(owner_file: Path, filename: str) -> Path:
    """Find the file an external link points at.

    Links are normally relative to the file holding them (Luxendo writes
    ``raw/stack_.../Cam_long_00000.lux.h5``). A link written as an absolute
    path on the acquisition PC is resolved by its basename next to the owner.
    """
    base = owner_file.parent
    candidate = base / filename
    if candidate.is_file():
        return candidate
    if Path(filename).is_absolute() and Path(filename).is_file():
        return Path(filename)
    return base / PureWindowsPath(filename).name


def get_member(group: Any, name: str, owner_file: Path) -> tuple[Any, Path] | None:
    """Return ``(object, file_it_lives_in)`` for ``group[name]``.

    External links are followed by hand (rather than by HDF5) so that paths
    written on Windows acquisition machines still resolve, and so a missing
    target is reported as missing instead of raising deep inside h5py.
    """
    import h5py

    link = group.get(name, getlink=True)
    if link is None:
        return None
    if isinstance(link, h5py.ExternalLink):
        target = resolve_link_target(owner_file, link.filename)
        if not target.is_file():
            raise FileNotFoundError(str(target))
        f = open_h5(target)
        obj = f.get(link.path)
        return (obj, target) if obj is not None else None
    obj = group.get(name)
    return (obj, owner_file) if obj is not None else None


def is_lux_view(group: Any) -> bool:
    """True if *group* (a file root or a nested group) holds a ``Data`` item."""
    try:
        return name_in(group, "Data")
    except Exception:
        return False


def name_in(group: Any, name: str) -> bool:
    return group.get(name, getlink=True) is not None


def parse_metadata(raw: Any) -> dict[str, Any]:
    """Parse the Luxendo JSON ``metadata`` payload (bytes/str/array or dataset).

    Returns an empty dict when it is missing or unreadable. Possible keys:

    - ``voxel_size_um``: ``(z, y, x)`` floats, only when all three are > 0
    - ``affine_zyx``: 4x4 voxel-index -> sample-space matrix in napari order
    - ``channel_description``, ``channel``, ``stack``, ``objective``,
      ``camera``, ``time_point``: view identity, as strings
    - ``raw``: the whole parsed JSON document
    """
    result: dict[str, Any] = {}
    if raw is None:
        return result
    try:
        if hasattr(raw, "shape") and hasattr(raw, "dtype") and not isinstance(raw, np.ndarray):
            raw = raw[()]  # h5py dataset
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes() if raw.dtype.kind in "SV" else raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8").rstrip("\x00")
        meta = json.loads(raw)
    except Exception as exc:
        logger.warning("Could not parse Luxendo metadata: %s", exc)
        return result

    result["raw"] = meta
    proc = meta.get("processingInformation", {}) if isinstance(meta, dict) else {}
    if not isinstance(proc, dict):
        return result

    voxel = proc.get("voxel_size_um")
    if isinstance(voxel, dict):
        try:
            zyx = (float(voxel["depth"]), float(voxel["height"]), float(voxel["width"]))
            if all(v > 0 for v in zyx):
                result["voxel_size_um"] = zyx
        except (KeyError, TypeError, ValueError):
            pass

    affine = compose_affine_to_sample(proc.get("affine_to_sample"))
    if affine is not None:
        result["affine_zyx"] = affine

    for key in ("channel_description", "channel", "stack", "objective", "camera", "time_point"):
        value = proc.get(key)
        if value not in (None, ""):
            result[key] = str(value)
    return result


def compose_affine_to_sample(transforms: Any) -> np.ndarray | None:
    """Combine ``affine_to_sample`` into one 4x4 matrix in napari (z, y, x) order.

    Luxendo lists transforms in the order they are applied, each with a 3x3
    ``matrix`` (rows) and a ``translation``, acting on voxel coordinates
    ``(x, y, z) = (width, height, plane)``. napari wants ``(z, y, x)``, so
    the composed transform is conjugated with the axis reversal.
    Returns None if the field is missing or malformed.
    """
    if isinstance(transforms, dict):
        transforms = [transforms]
    if not isinstance(transforms, list) or not transforms:
        return None
    m = np.eye(3)
    t = np.zeros(3)
    try:
        for tr in transforms:
            mi = np.asarray(tr.get("matrix", np.eye(3)), dtype=float)
            ti = np.asarray(tr.get("translation", np.zeros(3)), dtype=float)
            if mi.shape != (3, 3) or ti.shape != (3,):
                return None
            m, t = mi @ m, mi @ t + ti
    except (AttributeError, TypeError, ValueError):
        return None
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(t))) or abs(np.linalg.det(m)) < 1e-12:
        return None
    rev = np.eye(3)[::-1]
    out = np.eye(4)
    out[:3, :3] = rev @ m @ rev
    out[:3, 3] = t[::-1]
    return out


def detect_pyramid_levels(group: Any) -> list[tuple[str, int, int, int]]:
    """Return ``(name, factor_w, factor_h, factor_d)`` for each pyramid level.

    ``Data`` itself is not included. Levels are sorted by total downsampling.
    """
    levels = []
    for name in group.keys():
        m = _PYRAMID_RE.match(name)
        if m:
            levels.append((name, int(m.group(1)), int(m.group(2)), int(m.group(3))))
    levels.sort(key=lambda t: t[1] * t[2] * t[3])
    return levels


def _as_dask(ds: Any) -> da.Array:
    """Expose individual planes; storage chunks must not dictate display reads."""
    if ds.ndim == 2:
        return da.from_array(ds, chunks=(-1, -1))[np.newaxis]
    return da.from_array(reader_for(ds), chunks=(1, 512, 512),
                         name='luxendo-data-' + uuid.uuid4().hex,
                         asarray=False, fancy=False, meta=np.empty((0, 0, 0), dtype=ds.dtype))


@dataclass
class LuxVolume:
    """One Luxendo view (a single tile/camera/channel/timepoint)."""

    path: Path  # file the full-resolution data lives in
    levels: list[da.Array]
    level_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    datasets: list[Any] = field(default_factory=list)  # h5py dataset per level
    factors: list[tuple[int, int, int]] = field(default_factory=list)  # (z, y, x) per level
    view_name: str | None = None  # group name inside a nested/main file

    @property
    def data(self) -> da.Array:
        """Full-resolution ``(Z, Y, X)`` array."""
        return self.levels[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.levels[0].shape

    @property
    def dtype(self) -> np.dtype:
        return self.levels[0].dtype

    @property
    def voxel_size_um(self) -> tuple[float, float, float] | None:
        return self.metadata.get("voxel_size_um")

    @property
    def affine(self) -> np.ndarray | None:
        """4x4 voxel (z, y, x) -> sample-space (um) transform, if known."""
        return self.metadata.get("affine_zyx")

    @property
    def name(self) -> str:
        return self.metadata.get("channel_description") or _strip_ext(self.path.name)

    def read(self, level: int, region: tuple[slice, slice, slice]) -> np.ndarray:
        """Read a ``(z, y, x)`` region of *level* straight from HDF5."""
        ds = self.datasets[level]
        if ds.ndim == 2:
            return np.asarray(ds[region[1], region[2]])[np.newaxis][region[0]]
        return reader_for(ds)[region]


def _strip_ext(name: str) -> str:
    for ext in (".lux.h5", ".h5"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def open_lux_volume(path: Path | str) -> LuxVolume:
    """Open a flat ``.lux.h5`` file lazily."""
    path = Path(path)
    f = open_h5(path)
    if not is_lux_view(f):
        raise ValueError(f"{path.name}: not a flat Luxendo file (no 'Data' dataset)")
    return open_lux_group(f, path)


def open_lux_dataset(path: Path | str, dataset_path: str) -> LuxVolume:
    """Open the Luxendo view whose full-resolution data is *dataset_path* in *path*.

    Headers link to a specific dataset, e.g. ``/Data`` of a flat file or
    ``/timepoint_0/channel_0/raw_tile/Data`` of a nested one. The view is the
    group holding that link, so its metadata and pyramid levels are the
    ones next to it, not whatever sits at the file root.
    """
    import h5py

    path = Path(path)
    target = "/" + dataset_path.strip("/")
    if target.rsplit("/", 1)[-1] != "Data":
        raise ValueError(f"{path.name}:{target}: expected a link to a Luxendo 'Data' dataset")
    f = open_h5(path)
    # Take the group holding the link, not the parent of the dataset it points
    # at: if Data is an external link, the dataset's parent belongs to the
    # pixel file and the wrapper view's own metadata would be skipped.
    parent_path, _, name = target.rpartition("/")
    group = f.get(parent_path or "/")
    if not isinstance(group, h5py.Group) or not name_in(group, name):
        raise ValueError(f"{path.name}:{target}: no such dataset")
    # The group may live in yet another file (an external link inside path).
    owner = Path(group.file.filename)
    view_name = group.name.rsplit("/", 1)[-1] or None
    return open_lux_group(group, owner, view_name=view_name)


def open_lux_group(group: Any, owner_file: Path, view_name: str | None = None) -> LuxVolume:
    """Open the Luxendo view stored in *group* (following external links).

    Pyramid levels that are not strictly smaller than the previous level are
    dropped, because napari requires each multiscale level to shrink. If
    ``Data`` links to another file, that file's own pyramid levels are used
    when the view itself lists none (main files often link only ``Data``).
    """
    import h5py

    # Links inside the group are relative to the file that holds it, which is
    # not the caller's file when an external link led here.
    owner_file = Path(group.file.filename)
    data = get_member(group, "Data", owner_file)
    if data is None or not isinstance(data[0], h5py.Dataset) or data[0].ndim not in (2, 3):
        raise ValueError(f"{owner_file.name}:{group.name}: no usable 'Data' dataset")
    data_ds, data_file = data

    level_source, level_file = group, owner_file
    if not detect_pyramid_levels(group) and data_ds.parent is not None:
        level_source, level_file = data_ds.parent, data_file

    # The view's own metadata wins; otherwise use the one next to the
    # full-resolution data, wherever the pyramid levels come from.
    meta_obj = get_member(group, "metadata", owner_file)
    if meta_obj is None and data_ds.parent != group:
        meta_obj = get_member(data_ds.parent, "metadata", data_file)
    metadata = parse_metadata(meta_obj[0]) if meta_obj else {}

    levels, names, datasets, factors = [_as_dask(data_ds)], ["Data"], [data_ds], [(1, 1, 1)]
    for name, fw, fh, fd in detect_pyramid_levels(level_source):
        try:
            member = get_member(level_source, name, level_file)
        except FileNotFoundError:
            continue
        if member is None or not isinstance(member[0], h5py.Dataset):
            continue
        ds = member[0]
        arr = _as_dask(ds)
        prev = levels[-1].shape
        if (
            arr.dtype != levels[0].dtype
            or any(a > p for a, p in zip(arr.shape, prev))
            or arr.shape == prev
            or 0 in arr.shape
        ):
            logger.info("Skipping pyramid level %s with shape %s", name, arr.shape)
            continue
        levels.append(arr)
        names.append(name)
        datasets.append(ds)
        factors.append((fd, fh, fw))

    return LuxVolume(
        path=data_file,
        levels=levels,
        level_names=names,
        metadata=metadata,
        datasets=datasets,
        factors=factors,
        view_name=view_name,
    )
