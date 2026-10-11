"""Show overview-cache builds as live progress bars in napari's activity panel.

While :mod:`._preview_cache` averages overview levels in the background, each
build gets a napari progress bar (bottom right of the window) that shows the
size of the cache it is writing and how much of it is done, for example
``Overview cache: 35% of 528 MB built``. The bars
are updated from a Qt timer on the main thread; the builder thread only
counts what it has averaged.
"""
from __future__ import annotations

import logging

from ._preview_cache import building_caches

logger = logging.getLogger(__name__)

# How often the bars are refreshed (ms).
_POLL_MS = 500

_TIMER = None
_BARS = None  # kept here: Qt holds bound-method slots only weakly


def install() -> None:
    """Start refreshing progress bars (once per process; needs a Qt application)."""
    global _TIMER, _BARS
    if _TIMER is not None:
        return
    try:
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is None:
        return
    _BARS = CacheProgress()
    _TIMER = QTimer(app)
    _TIMER.setInterval(_POLL_MS)
    _TIMER.timeout.connect(_BARS.poll)
    _TIMER.start()


def format_size(nbytes: int) -> str:
    for unit, scale in (('GB', 2**30), ('MB', 2**20), ('KB', 2**10)):
        if nbytes >= scale:
            value = nbytes / scale
            return f'{value:.1f} {unit}' if value < 10 else f'{value:.0f} {unit}'
    return f'{nbytes} B'


def describe(cache, percent: int) -> str:
    text = f'Overview cache: {percent}% of {format_size(cache.nbytes)} built'
    return text + (' (paused while the display reads)' if cache.paused else '')


class CacheProgress:
    """One napari progress bar per cache being built, refreshed by :meth:`poll`."""

    def __init__(self) -> None:
        self._bars: dict = {}  # cache -> napari progress
        self._descs: dict = {}  # cache -> text last shown (tqdm appends ': ' to desc)

    def poll(self) -> None:
        try:
            self._poll()
        except Exception as exc:  # a progress bar must never break napari
            logger.debug('Overview cache progress: %s', exc)

    def _poll(self) -> None:
        building = building_caches()
        for cache in [c for c in self._bars if c not in building]:
            self._bars.pop(cache).close()
            self._descs.pop(cache, None)
        for cache in building:
            bar = self._bars.get(cache)
            if bar is None:
                from napari.utils import progress

                desc = self._descs[cache] = describe(cache, 0)
                bar = self._bars[cache] = progress(total=100, desc=desc)
            percent = int(cache.fraction_built * 100)
            if percent > bar.n:
                bar.update(percent - bar.n)
            desc = describe(cache, percent)
            if desc != self._descs[cache]:
                self._descs[cache] = desc
                bar.set_description(desc)
