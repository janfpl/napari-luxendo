"""Nested Luxendo files and "main" files (``main_raw.lux.h5`` etc.).

Layout (from the Luxendo Image spec)::

    timepoint_<name>/
      channel_<name>/
        <view>/          # raw_<name>, proc_<name> or any name
          Data, Data_2_2_2, ..., metadata   (datasets or external links)

A main file links every tile, camera and view of an experiment, so opening it
loads the whole acquisition.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._lux import is_lux_view

logger = logging.getLogger(__name__)


@dataclass
class NestedView:
    timepoint: str  # group name without the "timepoint_" prefix
    channel: str  # group name without the "channel_" prefix
    view: str  # view group name, e.g. "raw_stack_1-x00-y00_obj_bottom"
    group: Any

    @property
    def kind(self) -> str:
        """``"raw"``, ``"proc"`` or ``"other"``, from the view-name prefix."""
        lower = self.view.lower()
        if lower.startswith("raw_"):
            return "raw"
        if lower.startswith("proc_"):
            return "proc"
        return "other"


def is_nested_file(h5file: Any) -> bool:
    """True if *h5file* has the nested ``timepoint_*/channel_*/<view>`` layout."""
    if is_lux_view(h5file):
        return False
    return any(True for _ in iter_nested_views(h5file, limit=1))


def iter_nested_views(h5file: Any, limit: int | None = None):
    """Yield every view of a nested file, in timepoint/channel/view order."""
    count = 0
    for tp_name in sorted(_prefixed(h5file, "timepoint_"), key=_natural_key):
        tp = h5file[tp_name]
        for ch_name in sorted(_prefixed(tp, "channel_"), key=_natural_key):
            ch = tp[ch_name]
            for view_name in sorted(ch.keys(), key=_natural_key):
                group = ch.get(view_name)
                if group is None or not hasattr(group, "keys") or not is_lux_view(group):
                    continue
                yield NestedView(
                    timepoint=tp_name[len("timepoint_"):],
                    channel=ch_name[len("channel_"):],
                    view=view_name,
                    group=group,
                )
                count += 1
                if limit is not None and count >= limit:
                    return


def _prefixed(group: Any, prefix: str) -> list[str]:
    import h5py

    out = []
    for name in group.keys():
        if not name.startswith(prefix):
            continue
        try:
            if isinstance(group.get(name), h5py.Group):
                out.append(name)
        except Exception:  # dangling link
            continue
    return out


def _natural_key(name: str) -> list[Any]:
    """Sort ``x2`` before ``x10``."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def timepoint_index(name: str, fallback: int) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else fallback
