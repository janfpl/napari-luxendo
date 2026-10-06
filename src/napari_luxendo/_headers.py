"""Resolve Luxendo companion headers to the channel files they link to.

Luxendo acquisitions can ship an Imaris ``.ims`` header and/or a
BigDataViewer ``*_bdv.h5`` (+ ``*_bdv.xml``) header next to the per-channel
``.lux.h5`` files. Neither holds pixel data: both are trees of HDF5 external
links into the channel files. Reading them gives the channel order, the
timepoints, and (for ``.ims``) channel names and display colors.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._lux import resolve_link_target

logger = logging.getLogger(__name__)


@dataclass
class HeaderChannel:
    """One header entry (an .ims channel or a BDV setup): files per timepoint.

    ``channel`` groups entries that are tiles of the same channel (a BDV
    setup's ``channel`` attribute); ``name`` is the entry's own label.
    """

    files: list[Path]
    name: str | None = None
    color: tuple[float, float, float] | None = None
    timepoints: list[int] = field(default_factory=list)
    channel: str | None = None
    channel_name: str | None = None


@dataclass
class HeaderInfo:
    path: Path
    kind: str  # "ims" or "bdv"
    channels: list[HeaderChannel] = field(default_factory=list)
    voxel_size_um: tuple[float, float, float] | None = None  # (z, y, x)


def _text(value: Any) -> str:
    """Decode an Imaris text attribute (char array, bytes or str)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    try:
        return value.tobytes().decode("utf-8", "replace")
    except Exception:
        return str(value)


def _numeric_suffix(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def _external_link(group: Any, key: str) -> Any:
    import h5py

    link = group.get(key, getlink=True)
    return link if isinstance(link, h5py.ExternalLink) else None


def read_ims_header(path: Path | str, h5file: Any) -> HeaderInfo | None:
    """Read an Imaris header. Returns None if it has no external links.

    A native Imaris file (pixels stored inside) is not ours to read, so it is
    left for other readers.
    """
    path = Path(path)
    level0 = h5file.get("DataSet/ResolutionLevel 0")
    if level0 is None:
        return None

    timepoints = sorted(
        (k for k in level0.keys() if k.startswith("TimePoint")), key=_numeric_suffix
    )
    per_channel: dict[int, dict[int, Path]] = {}
    for tp in timepoints:
        t = _numeric_suffix(tp)
        for ch in level0[tp].keys():
            if not ch.startswith("Channel"):
                continue
            link = _external_link(level0[tp][ch], "Data")
            if link is None:
                continue
            per_channel.setdefault(_numeric_suffix(ch), {})[t] = resolve_link_target(
                path, link.filename
            )
    if not per_channel:
        return None

    info = HeaderInfo(path=path, kind="ims")
    for c in sorted(per_channel):
        tps = sorted(per_channel[c])
        files = [per_channel[c][t] for t in tps]
        name = color = None
        attrs_grp = h5file.get(f"DataSetInfo/Channel {c}")
        if attrs_grp is not None:
            if "Name" in attrs_grp.attrs:
                name = _text(attrs_grp.attrs["Name"]).strip() or None
            if "Color" in attrs_grp.attrs:
                try:
                    rgb = tuple(float(v) for v in _text(attrs_grp.attrs["Color"]).split())
                    if len(rgb) == 3 and all(0 <= v <= 1 for v in rgb) and any(rgb):
                        color = rgb
                except ValueError:
                    pass
        info.channels.append(
            HeaderChannel(
                files=files, name=name, color=color, timepoints=tps,
                channel=str(c), channel_name=name,
            )
        )

    info.voxel_size_um = _ims_voxel_size(h5file)
    return info


def _ims_voxel_size(h5file: Any) -> tuple[float, float, float] | None:
    """Voxel size from ``DataSetInfo/Image`` extents, as ``(z, y, x)``."""
    img = h5file.get("DataSetInfo/Image")
    if img is None:
        return None
    try:
        sizes = []
        for axis, dim in ((2, "Z"), (1, "Y"), (0, "X")):
            n = float(_text(img.attrs[dim]))
            lo = float(_text(img.attrs[f"ExtMin{axis}"]))
            hi = float(_text(img.attrs[f"ExtMax{axis}"]))
            sizes.append((hi - lo) / n)
        if all(s > 0 for s in sizes):
            return tuple(sizes)  # type: ignore[return-value]
    except (KeyError, ValueError, ZeroDivisionError):
        pass
    return None


def read_bdv_header(path: Path | str, h5file: Any) -> HeaderInfo | None:
    """Read a BigDataViewer HDF5 header. Returns None if it has no external links.

    Channel names and voxel size come from the paired ``*_bdv.xml`` when present.
    """
    path = Path(path)
    timepoints = sorted(
        (k for k in h5file.keys() if re.fullmatch(r"t\d+", k)), key=_numeric_suffix
    )
    per_setup: dict[int, dict[int, Path]] = {}
    for tp in timepoints:
        t = _numeric_suffix(tp)
        for s in h5file[tp].keys():
            if not re.fullmatch(r"s\d+", s) or "0" not in h5file[tp][s]:
                continue
            link = _external_link(h5file[tp][s]["0"], "cells")
            if link is None:
                continue
            per_setup.setdefault(_numeric_suffix(s), {})[t] = resolve_link_target(
                path, link.filename
            )
    if not per_setup:
        return None

    xml = _read_bdv_xml(path.with_name(path.name[: -len(".h5")] + ".xml"))
    info = HeaderInfo(path=path, kind="bdv", voxel_size_um=xml.voxel)
    for s in sorted(per_setup):
        tps = sorted(per_setup[s])
        channel = xml.setup_channel.get(s)
        info.channels.append(
            HeaderChannel(
                files=[per_setup[s][t] for t in tps],
                name=xml.names.get(s),
                timepoints=tps,
                channel=channel,
                channel_name=xml.channel_names.get(channel) if channel is not None else None,
            )
        )
    return info


@dataclass
class _BdvXml:
    names: dict[int, str] = field(default_factory=dict)  # setup id -> name
    setup_channel: dict[int, str] = field(default_factory=dict)  # setup id -> channel id
    channel_names: dict[str, str] = field(default_factory=dict)  # channel id -> name
    voxel: tuple[float, float, float] | None = None


def _read_bdv_xml(xml_path: Path) -> _BdvXml:
    """Setup names, channel attributes and voxel size from a BDV XML, if present."""
    out = _BdvXml()
    if not xml_path.is_file():
        return out
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        logger.warning("Could not parse %s: %s", xml_path.name, exc)
        return out
    for ch in root.iter("Channel"):
        cid, cname = (ch.findtext("id") or "").strip(), (ch.findtext("name") or "").strip()
        if cid and cname:
            out.channel_names[cid] = cname
    for vs in root.iter("ViewSetup"):
        try:
            sid = int((vs.findtext("id") or "").strip())
        except ValueError:
            continue
        name = (vs.findtext("name") or "").strip()
        if name:
            out.names[sid] = name
        channel = (vs.findtext("attributes/channel") or "").strip()
        if channel:
            out.setup_channel[sid] = channel
        size = vs.findtext("voxelSize/size")
        if out.voxel is None and size:
            try:
                x, y, z = (float(v) for v in size.split())
                if x > 0 and y > 0 and z > 0:
                    out.voxel = (z, y, x)
            except ValueError:
                pass
    return out
