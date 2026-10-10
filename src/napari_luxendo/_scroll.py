"""Show a coarser pyramid level while a slider moves, and full detail once it stops.

In 2D napari loads exactly one level of a multiscale layer, the one that fits
the canvas. While the user scrolls through Z (or time) every step therefore
waits for a full-detail plane. This controller makes napari pick a level
about two steps coarser while a slider is moving, and goes back to the level
napari would pick once the slider has been still for a moment.

It only switches when the coarser level is cheap to read: a native pyramid
level, or a generated preview whose on-disk cache is complete (see
:mod:`._preview_cache`). An uncached preview is a sample of full resolution
and would be slower, not faster.

``NAPARI_LUXENDO_SCROLL_PREVIEW=0`` turns it off.
"""
from __future__ import annotations

import logging
import os
import weakref
from typing import Any, Callable, Optional

from ._preview import level_is_cheap

logger = logging.getLogger(__name__)

# The canvas size napari compares the field of view against is divided by
# this while scrolling: 4 selects a level about two halvings coarser.
_SCROLL_DOWNSCALE = 4
# Slider stillness (ms) after which full detail is loaded.
_SETTLE_MS = 200

_CONTROLLERS: "weakref.WeakKeyDictionary[Any, ScrollPreview]" = weakref.WeakKeyDictionary()


def enabled() -> bool:
    return os.environ.get('NAPARI_LUXENDO_SCROLL_PREVIEW', '1') != '0'


def install_current_viewer() -> None:
    """Attach the controller to the napari viewer that is opening a file, if any."""
    if not enabled():
        return
    try:
        import napari

        viewer = napari.current_viewer()
    except Exception:
        return
    if viewer is not None:
        install(viewer, timer=_qt_timer())


def install(viewer, timer: Optional[Callable[[Callable[[], None]], Callable[[], None]]] = None):
    """Attach (once) a :class:`ScrollPreview` to *viewer* and return it."""
    controller = _CONTROLLERS.get(viewer)
    if controller is None:
        controller = _CONTROLLERS[viewer] = ScrollPreview(viewer, timer)
    return controller


def _qt_timer():
    """A restartable single-shot Qt timer factory, or None without a Qt app."""
    try:
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication
    except Exception:
        return None
    if QApplication.instance() is None:
        return None

    def make(callback):
        timer = QTimer()
        timer.setSingleShot(True)
        timer.setInterval(_SETTLE_MS)
        timer.timeout.connect(callback)
        return timer.start

    return make


class ScrollPreview:
    """Per-viewer controller; see the module docstring.

    *timer* builds a restart function from a callback (Qt in the app). Without
    one, call :meth:`settle` yourself (tests, scripts).
    """

    def __init__(self, viewer, timer=None) -> None:
        self.viewer = viewer
        self.scrolling = False
        self._restart = timer(self.settle) if timer is not None else None
        self._layers: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self._coarse: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self._tried: "weakref.WeakSet[Any]" = weakref.WeakSet()  # this scroll
        viewer.layers.events.inserted.connect(self._on_inserted)
        for layer in viewer.layers:
            self.track(layer)

    def _on_inserted(self, event) -> None:
        self.track(event.value)

    def track(self, layer) -> None:
        """Let *layer* use coarse levels while scrolling (Luxendo multiscale images)."""
        if layer in self._layers or not getattr(layer, 'multiscale', False) \
                or 'luxendo' not in getattr(layer, 'metadata', {}):
            return
        original = layer._update_draw
        dims = self.viewer.dims
        state: dict[str, Any] = {'step': (tuple(dims.current_step), dims.ndisplay)}
        controller = self

        def update_draw(scale_factor, corner_pixels_displayed, shape_threshold):
            state['args'] = (scale_factor, corner_pixels_displayed, shape_threshold)
            if controller.scrolling and layer in controller._coarse:
                shape_threshold = tuple(max(1, int(s) // _SCROLL_DOWNSCALE) for s in shape_threshold)
            return original(scale_factor, corner_pixels_displayed, shape_threshold)

        # napari's own dims handler runs before any plugin callback, so a
        # slider step is caught where the layer is asked for its new slice:
        # _slice_dims (sync slicing) or _make_slice_request (async slicing).
        for name in ('_slice_dims', '_make_slice_request'):
            method = getattr(layer, name, None)
            if method is not None:
                setattr(layer, name, _before_slice(controller, layer, method))
        layer._update_draw = update_draw
        layer._luxendo_draw_state = state
        self._layers.add(layer)

    def _on_slice(self, layer, dims) -> None:
        """A slice is about to be made: if a slider moved, go coarse first."""
        if dims is None:
            return
        step = (tuple(dims.current_step), dims.ndisplay)
        last = layer._luxendo_draw_state.get('step')
        layer._luxendo_draw_state['step'] = step
        if last is None or last == step or dims.ndisplay != 2 or last[1] != 2:
            return
        self.scrolling = True
        if layer not in self._tried:
            self._go_coarse(layer)
        if self._restart is not None:
            self._restart()

    def _go_coarse(self, layer) -> None:
        state = getattr(layer, '_luxendo_draw_state', {})
        if 'args' not in state or not layer.visible:
            return
        self._tried.add(layer)
        fine = layer.data_level
        self._coarse.add(layer)
        _redraw(layer, state['args'], refresh=False)  # the slice being made uses it
        coarse = layer.data_level
        try:
            cheap = coarse != fine and level_is_cheap(layer.data[coarse])
        except Exception as exc:
            logger.debug('Scroll preview: %s', exc)
            cheap = False
        if not cheap:
            self._coarse.discard(layer)
            _redraw(layer, state['args'], refresh=False)

    def settle(self) -> None:
        """The slider stopped: load the level napari would pick for every layer."""
        if not self.scrolling:
            return
        self.scrolling = False
        layers = list(self._coarse)
        self._coarse = weakref.WeakSet()
        self._tried = weakref.WeakSet()
        for layer in layers:
            state = getattr(layer, '_luxendo_draw_state', {})
            if 'args' in state:
                _redraw(layer, state['args'], refresh=True)


def _before_slice(controller, layer, method):
    def wrapped(dims, *args, **kwargs):
        try:
            controller._on_slice(layer, dims)
        except Exception as exc:  # never break slicing
            logger.debug('Scroll preview: %s', exc)
        return method(dims, *args, **kwargs)

    return wrapped


def _redraw(layer, args, refresh: bool) -> None:
    """Run the layer's (wrapped) level selection with the last canvas state."""
    if refresh:
        layer._update_draw(*args)
        return
    layer.refresh = _no_refresh
    try:
        layer._update_draw(*args)
    finally:
        del layer.refresh


def _no_refresh(*args, **kwargs) -> None:
    pass
