"""Show a coarser pyramid level while the view moves, full detail once it stops,
and prefetch the surroundings of the view.

In 2D napari loads exactly one level of a multiscale layer, the one that fits
the canvas, and only the part of it on screen. While the user scrolls through
Z (or time) or zooms, every step therefore waits for a full-detail read, and
every pan step reads the newly visible area from disk. This controller:

* makes napari pick a level about two steps coarser while a slider moves or
  the camera zooms, and goes back to the level napari would pick once the
  view has been still for a moment. A zoom step whose full-detail area is
  already in memory is shown at full detail right away;
* once the view stops, reads the area around the view
  (one screen in every direction by default) at the displayed level and the
  two coarser ones into the in-memory block cache (:mod:`._tilecache`) in a
  background thread, so a pan into it shows from memory.

The coarser level is only used when it is cheap to read: a native pyramid
level, or a generated preview whose on-disk cache is complete (see
:mod:`._preview_cache`). An uncached preview is a sample of full resolution
and would be slower, not faster.

``NAPARI_LUXENDO_SCROLL_PREVIEW=0`` turns it all off.
``NAPARI_LUXENDO_PREFETCH`` is the prefetch distance in screens (default 1,
``0`` turns prefetching off).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import weakref
from typing import Any, Callable, Optional

from . import _tilecache
from ._preview import level_is_cheap

logger = logging.getLogger(__name__)

# The canvas size napari compares the field of view against is divided by
# this while scrolling: 4 selects a level about two halvings coarser.
_SCROLL_DOWNSCALE = 4
# Slider stillness (ms) after which full detail is loaded.
_SETTLE_MS = 200
# Coarser levels prefetched around the view, besides the displayed one.
_PREFETCH_COARSER = 2

_CONTROLLERS: "weakref.WeakKeyDictionary[Any, ScrollPreview]" = weakref.WeakKeyDictionary()


def enabled() -> bool:
    return os.environ.get('NAPARI_LUXENDO_SCROLL_PREVIEW', '1') != '0'


def prefetch_screens() -> float:
    try:
        return max(0.0, float(os.environ.get('NAPARI_LUXENDO_PREFETCH', '1')))
    except ValueError:
        return 1.0


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
        self.zooming = False
        self._restart = timer(self.settle) if timer is not None else None
        self.prefetcher = Prefetcher()
        self._layers: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self._coarse: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self._tried: "weakref.WeakSet[Any]" = weakref.WeakSet()  # this scroll
        viewer.layers.events.inserted.connect(self._on_inserted)
        viewer.camera.events.zoom.connect(self._on_zoom)
        viewer.camera.events.center.connect(self._on_pan)
        for layer in viewer.layers:
            self.track(layer)

    @property
    def moving(self) -> bool:
        return self.scrolling or self.zooming

    def _on_zoom(self, event=None) -> None:
        """The camera zoomed: show coarse first wherever full detail is not in memory."""
        self.zooming = True
        self.prefetcher.cancel()
        if self._restart is not None:
            self._restart()

    def _on_pan(self, event=None) -> None:
        """The camera moved: pause prefetching until it stops.

        A prefetch read in flight holds the HDF5 library lock (and, for
        compressed data, decompresses whole chunks), so it would delay the
        display reads of the drag itself.
        """
        self.prefetcher.cancel()
        if self._restart is not None:
            self._restart()

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
            if controller.zooming and not controller.scrolling:
                try:
                    controller._choose_for_zoom(layer, original, state['args'])
                except Exception as exc:  # never break drawing
                    logger.debug('Zoom preview: %s', exc)
                    controller._coarse.discard(layer)
            if controller.moving and layer in controller._coarse:
                shape_threshold = _coarse_threshold(shape_threshold)
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
        self.prefetcher.cancel()
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

    def _choose_for_zoom(self, layer, original, args) -> None:
        """Decide whether this zoom step shows *layer* coarse or at full detail."""
        if not layer.visible or self.viewer.dims.ndisplay != 2:
            return
        before = (layer._data_level, layer.corner_pixels.copy())
        scale_factor, corners, threshold = args
        layer.refresh = _no_refresh
        try:
            original(scale_factor, corners, threshold)
            fine = layer.data_level
            if region_in_memory(layer, fine, layer.corner_pixels):
                coarse_ok = False
            else:
                original(scale_factor, corners, _coarse_threshold(threshold))
                coarse = layer.data_level
                coarse_ok = coarse != fine and level_is_cheap(layer.data[coarse])
        finally:
            del layer.refresh
            layer._data_level, layer.corner_pixels = before
        if coarse_ok:
            self._coarse.add(layer)
        else:
            self._coarse.discard(layer)

    def settle(self) -> None:
        """The view stopped: load the level napari would pick, then prefetch around it."""
        if self.moving:
            self.scrolling = self.zooming = False
            layers = list(self._coarse)
            self._coarse = weakref.WeakSet()
            self._tried = weakref.WeakSet()
            for layer in layers:
                state = getattr(layer, '_luxendo_draw_state', {})
                if 'args' in state:
                    _redraw(layer, state['args'], refresh=True)
        self.prefetch()

    def prefetch(self) -> None:
        """Queue background reads of the area around the view, nearest first."""
        screens = prefetch_screens()
        if screens <= 0 or _tilecache.store() is None:
            return
        jobs = []
        for layer in list(self._layers):
            try:
                jobs.extend(prefetch_jobs(layer, screens))
            except Exception as exc:
                logger.debug('Prefetch: %s', exc)
        jobs.sort(key=lambda job: job[0])
        self.prefetcher.schedule([job for _, job in jobs])

    def wait_prefetch(self, timeout: Optional[float] = None) -> bool:
        """Block until queued prefetching is done (benchmarks, tests)."""
        return self.prefetcher.wait(timeout)


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


def _coarse_threshold(shape_threshold):
    return tuple(max(1, int(s) // _SCROLL_DOWNSCALE) for s in shape_threshold)


def _plane_index(layer, level):
    """napari's index of the displayed plane at *level*: ints, and slice(None) on screen."""
    request = layer._make_slice_request_internal(
        slice_input=layer._slice_input, data_slice=layer._data_slice, dask_indexer=None)
    return list(request._point_to_slices(request._thick_slice_at_level(level).point))


def locate(layer, level):
    """The cached source and plane napari shows *layer* at *level* from, or None."""
    if _tilecache.store() is None or layer.projection_mode != 'none':
        return None
    array = layer.data[level]
    displayed = list(layer._slice_input.displayed)
    if displayed != [array.ndim - 2, array.ndim - 1]:
        return None  # only the usual (..., Y, X) display is prefetched
    index = _plane_index(layer, level)
    memo = layer._luxendo_draw_state.setdefault('located', {})
    memo_key = (level, tuple(i for i in index if isinstance(i, int)))
    if memo_key not in memo:
        import dask
        import numpy as np

        key = tuple(slice(0, 1) if isinstance(i, slice) else i for i in index)
        with _tilecache.locate() as found, dask.config.set(scheduler='synchronous'):
            np.asarray(array[key])
        sources = {(id(s), z): (s, z) for s, z in found}
        memo[memo_key] = next(iter(sources.values())) if len(sources) == 1 else None
    return memo[memo_key]


def region_in_memory(layer, level, corners) -> bool:
    """True if the on-screen part of *level* (napari corner pixels) is in the block cache."""
    where = locate(layer, level)
    if where is None:
        return False
    source, z = where
    (y0, x0), (y1, x1) = corners[:, -2:]
    return source.has_region(z, int(y0), int(y1) + 1, int(x0), int(x1) + 1)


def prefetch_jobs(layer, screens: float):
    """(distance, read) pairs for the blocks around the view of *layer*."""
    import numpy as np

    if not layer.visible or layer._slice_input.ndisplay != 2 or not layer.multiscale \
            or 'luxendo' not in layer.metadata \
            or 'args' not in getattr(layer, '_luxendo_draw_state', {}):
        return []  # not drawn yet: napari's corners are not a view
    level = int(layer.data_level)
    factors = np.asarray(layer.downsample_factors)[:, -2:]
    corners = np.asarray(layer.corner_pixels)[:, -2:].astype(float)
    corners[1] += 1
    # The view in full-resolution pixels, its centre and size.
    view0 = corners * factors[level]
    centre0 = view0.mean(axis=0)
    size0 = np.maximum(view0[1] - view0[0], 1)
    b = _tilecache.BLOCK_YX
    jobs = []
    for k in range(level, min(len(layer.data) - 1, level + _PREFETCH_COARSER) + 1):
        if k > level and not level_is_cheap(layer.data[k]):
            continue
        where = locate(layer, k)
        if where is None:
            continue
        source, z = where
        shape = np.asarray(source.shape[1:])
        lo = np.clip(np.floor((view0[0] - screens * size0) / factors[k]), 0, shape).astype(int)
        hi = np.clip(np.ceil((view0[1] + screens * size0) / factors[k]), 0, shape).astype(int)
        for by in range(lo[0] // b, -(-hi[0] // b)):
            for bx in range(lo[1] // b, -(-hi[1] // b)):
                mid = (np.array([by, bx]) + 0.5) * b * factors[k]
                # Nearest first; coarser levels just after blocks at the same distance.
                distance = float(np.max(np.abs(mid - centre0) / size0)) + 0.01 * (k - level)
                jobs.append((distance, _filler(source, z, by, bx)))
    return jobs


def _filler(source, z, by, bx):
    return lambda: source.fill_block(z, by, bx)


class Prefetcher:
    """One background thread running the latest batch of reads; a new batch
    (or :meth:`cancel`) abandons the rest of the previous one."""

    def __init__(self) -> None:
        self._generation = 0
        self._lock = threading.Lock()
        self._queue: "queue.Queue" = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: Optional[threading.Thread] = None

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1

    def schedule(self, jobs) -> None:
        with self._lock:
            self._generation += 1
            if not jobs:
                return
            self._idle.clear()
            self._queue.put((self._generation, jobs))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='luxendo-prefetch',
                                                daemon=True)
                self._thread.start()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._idle.wait(timeout)

    def _run(self) -> None:
        while True:
            generation, jobs = self._queue.get()
            for job in jobs:
                if generation != self._generation:
                    break
                try:
                    job()
                except Exception as exc:  # a closed file, a removed layer
                    logger.debug('Prefetch: %s', exc)
                    break
            with self._lock:
                if self._queue.empty():
                    self._idle.set()
