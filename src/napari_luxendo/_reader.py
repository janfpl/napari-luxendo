"""napari reader for Luxendo ``.lux.h5`` channel files and their headers."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import dask.array as da
import numpy as np

from ._headers import HeaderChannel, HeaderInfo, read_bdv_header, read_ims_header
from ._lux import LuxVolume, is_lux_file, open_lux_volume

logger = logging.getLogger(__name__)

LayerData = tuple[Any, dict[str, Any], str]

# Colormaps cycled through for channels with no color hint.
_CHANNEL_COLORMAPS = ("green", "magenta", "cyan", "yellow", "red", "blue")

# Contrast limits are estimated from at most this many voxels.
_CONTRAST_SAMPLE_VOXELS = 16_000_000


def napari_get_reader(path: str | list[str]) -> Callable[..., list[LayerData]] | None:
    """Return a reader if every path is a Luxendo file or header we can open."""
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if not paths or not all(_classify(Path(p)) for p in paths):
        return None
    return read_luxendo


def _classify(path: Path) -> str | None:
    """Return ``"lux"``, ``"ims"``, ``"bdv"`` or None for *path*."""
    name = path.name.lower()
    if not path.is_file() or not name.endswith((".h5", ".ims")):
        return None
    try:
        import h5py

        with h5py.File(str(path), "r") as f:
            if name.endswith(".ims"):
                return "ims" if read_ims_header(path, f) else None
            if is_lux_file(f):
                return "lux"
            if read_bdv_header(path, f):
                return "bdv"
    except Exception as exc:
        logger.debug("Not a Luxendo file %s: %s", path, exc)
    return None


def read_luxendo(path: str | list[str]) -> list[LayerData]:
    """Read one or more ``.lux.h5`` / ``.ims`` / ``*_bdv.h5`` files into layers."""
    paths = [path] if isinstance(path, (str, Path)) else list(path)

    channels: list[tuple[HeaderChannel, HeaderInfo | None]] = []
    for p in map(Path, paths):
        kind = _classify(p)
        if kind == "lux":
            channels.append((HeaderChannel(files=[p]), None))
        elif kind in ("ims", "bdv"):
            header = _read_header(p, kind)
            for ch in header.channels:
                channels.append((ch, header))
        else:
            raise ValueError(f"{p.name}: not a Luxendo .lux.h5 file or header")

    multi = len(channels) > 1
    layers = []
    for index, (channel, header) in enumerate(channels):
        layers.append(_channel_layer(channel, header, index, multi))
    return layers


def _read_header(path: Path, kind: str) -> HeaderInfo:
    import h5py

    with h5py.File(str(path), "r") as f:
        info = read_ims_header(path, f) if kind == "ims" else read_bdv_header(path, f)
    assert info is not None  # _classify already checked
    missing = [str(p) for ch in info.channels for p in ch.files if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{path.name} links to channel files that are missing: " + ", ".join(missing)
        )
    return info


def _channel_layer(
    channel: HeaderChannel, header: HeaderInfo | None, index: int, multi: bool
) -> LayerData:
    volumes = [open_lux_volume(p) for p in channel.files]
    first = volumes[0]
    levels = _stack_timepoints(volumes)
    has_time = levels[0].ndim == 4

    voxel = first.voxel_size_um or (header.voxel_size_um if header else None)
    scale = list(voxel) if voxel else [1.0, 1.0, 1.0]
    if has_time:
        scale = [1.0, *scale]

    name = channel.name or first.name
    kwargs: dict[str, Any] = {
        "name": name,
        "scale": scale,
        "colormap": _pick_colormap(channel, first, index, multi),
        "blending": "additive" if multi else "translucent",
        "multiscale": len(levels) > 1,
        "metadata": {
            "luxendo": first.metadata.get("raw", {}),
            "path": str(first.path),
            "files": [str(v.path) for v in volumes],
            "header": str(header.path) if header else None,
            "pyramid_levels": first.level_names[: len(levels)],
            "voxel_size_um": voxel,
        },
    }
    limits = _estimate_contrast_limits(first)
    if limits is not None:
        kwargs["contrast_limits"] = limits

    data = levels if len(levels) > 1 else levels[0]
    return data, kwargs, "image"


def _stack_timepoints(volumes: list[LuxVolume]) -> list[da.Array]:
    """Stack each resolution level across timepoints into ``(T, Z, Y, X)``.

    A single timepoint is returned unchanged as ``(Z, Y, X)``. If timepoints
    disagree in shape or dtype, only the first one is used.
    """
    if len(volumes) == 1:
        return volumes[0].levels
    first = volumes[0]
    if any(v.shape != first.shape or v.dtype != first.dtype for v in volumes[1:]):
        logger.warning(
            "%s: timepoints differ in shape or dtype; showing only the first.",
            first.path.name,
        )
        return first.levels
    n_levels = 1
    for k in range(1, len(first.levels)):
        if all(
            len(v.levels) > k and v.levels[k].shape == first.levels[k].shape
            for v in volumes
        ):
            n_levels = k + 1
        else:
            break
    return [da.stack([v.levels[k] for v in volumes]) for k in range(n_levels)]


def _pick_colormap(
    channel: HeaderChannel, volume: LuxVolume, index: int, multi: bool
) -> Any:
    """Header color, else a color named by the channel, else a default."""
    if channel.color is not None:
        return {
            "colors": [[0.0, 0.0, 0.0, 1.0], [*channel.color, 1.0]],
            "name": f"luxendo-{'-'.join(f'{c:.3g}' for c in channel.color)}",
        }
    hint = colormap_from_description(channel.name or volume.metadata.get("channel_description", ""))
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
        if re.search(rf"\b{word}\b", lower):
            return word
    return None


def _estimate_contrast_limits(volume: LuxVolume) -> list[float] | None:
    """Robust contrast limits from the coarsest level, or a central plane.

    Avoids napari scanning a full-resolution volume to find its range.
    """
    coarsest = volume.levels[-1]
    try:
        if coarsest.size <= _CONTRAST_SAMPLE_VOXELS:
            sample = np.asarray(coarsest)
        else:
            data = volume.levels[0]
            sample = np.asarray(data[data.shape[0] // 2])
        if sample.size == 0:
            return None
        lo, hi = np.percentile(sample, (0.05, 99.95))
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            lo, hi = float(sample.min()), float(sample.max())
        if hi <= lo:
            hi = lo + 1.0
        return [lo, hi]
    except Exception as exc:
        logger.debug("Could not estimate contrast limits: %s", exc)
        return None
