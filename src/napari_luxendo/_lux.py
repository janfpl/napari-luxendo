"""Lazy access to Luxendo flat-structure ``.lux.h5`` volumes.

A Luxendo channel file holds the full-resolution volume in a ``Data`` dataset,
optional downsampled copies named ``Data_W_H_D`` (integer X/Y/Z factors), and a
JSON ``metadata`` dataset whose ``processingInformation`` block carries the
voxel size and channel description.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)

# Pyramid dataset names: Data_W_H_D (all integers).
_PYRAMID_RE = re.compile(r"^Data_(\d+)_(\d+)_(\d+)$")

# h5py File objects must stay open while dask arrays reference their datasets.
# They are kept here, keyed by resolved path, so opening the same file twice
# (e.g. from a header and directly) reuses one handle.
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
    with _OPEN_LOCK:
        for f in _OPEN_FILES.values():
            try:
                f.close()
            except Exception:
                pass
        _OPEN_FILES.clear()


def is_lux_file(h5file: Any) -> bool:
    """True if *h5file* looks like a flat Luxendo channel file."""
    try:
        data = h5file.get("Data")
    except Exception:
        return False
    import h5py

    return isinstance(data, h5py.Dataset) and data.ndim in (2, 3)


def parse_metadata(h5file: Any) -> dict[str, Any]:
    """Parse the Luxendo JSON ``metadata`` dataset.

    Returns an empty dict when it is missing or unreadable. Possible keys:

    - ``voxel_size_um``: ``(z, y, x)`` floats, only when all three are > 0
    - ``image_size_vx``: the raw width/height/depth dict
    - ``channel_description``: str
    - ``channel_id``
    - ``raw``: the whole parsed JSON document
    """
    result: dict[str, Any] = {}
    if "metadata" not in h5file:
        return result
    try:
        raw = h5file["metadata"][()]
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

    voxel = proc.get("voxel_size_um")
    if isinstance(voxel, dict):
        try:
            zyx = (
                float(voxel["depth"]),
                float(voxel["height"]),
                float(voxel["width"]),
            )
            if all(v > 0 for v in zyx):
                result["voxel_size_um"] = zyx
        except (KeyError, TypeError, ValueError):
            pass

    if isinstance(proc.get("image_size_vx"), dict):
        result["image_size_vx"] = proc["image_size_vx"]
    if proc.get("channel_description"):
        result["channel_description"] = str(proc["channel_description"])
    if proc.get("channel_id", "") != "":
        result["channel_id"] = proc["channel_id"]
    return result


def detect_pyramid_levels(h5file: Any) -> list[tuple[str, int, int, int]]:
    """Return ``(name, factor_w, factor_h, factor_d)`` for each pyramid level.

    ``Data`` itself is not included. Levels are sorted by total downsampling.
    """
    levels = []
    for name in h5file.keys():
        m = _PYRAMID_RE.match(name)
        if m:
            levels.append((name, int(m.group(1)), int(m.group(2)), int(m.group(3))))
    levels.sort(key=lambda t: t[1] * t[2] * t[3])
    return levels


def _as_dask(ds: Any) -> da.Array:
    """Wrap an h5py dataset as a dask array chunked in whole XY planes.

    The Z chunk follows the HDF5 chunking so viewing one plane only reads the
    HDF5 chunks that intersect it.
    """
    z_chunk = ds.chunks[0] if ds.chunks else 1
    if ds.ndim == 2:
        return da.from_array(ds, chunks=(-1, -1))[np.newaxis]
    return da.from_array(ds, chunks=(z_chunk, -1, -1))


@dataclass
class LuxVolume:
    """One Luxendo channel file, exposed as lazy dask arrays."""

    path: Path
    levels: list[da.Array]
    level_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

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
    def name(self) -> str:
        return self.metadata.get("channel_description") or _strip_ext(self.path.name)


def _strip_ext(name: str) -> str:
    for ext in (".lux.h5", ".h5"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def open_lux_volume(path: Path | str) -> LuxVolume:
    """Open a ``.lux.h5`` channel file lazily.

    Pyramid levels that are not strictly smaller than the previous level are
    dropped, because napari requires each multiscale level to shrink.
    """
    path = Path(path)
    f = open_h5(path)
    if not is_lux_file(f):
        raise ValueError(f"{path.name}: not a Luxendo file (no 'Data' dataset)")

    levels = [_as_dask(f["Data"])]
    names = ["Data"]
    for name, *_ in detect_pyramid_levels(f):
        arr = _as_dask(f[name])
        prev = levels[-1].shape
        if arr.dtype != levels[0].dtype or any(a > p for a, p in zip(arr.shape, prev)) \
                or arr.shape == prev or 0 in arr.shape:
            logger.info("Skipping pyramid level %s with shape %s", name, arr.shape)
            continue
        levels.append(arr)
        names.append(name)

    return LuxVolume(path=path, levels=levels, level_names=names, metadata=parse_metadata(f))
