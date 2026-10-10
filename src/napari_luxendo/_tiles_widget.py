"""Dock widget: unstitch mosaics into their tiles, and move each tile's Z plane."""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _tiles


class TilesWidget(QWidget):
    """*Unstitch tiles* shows every tile of a mosaic as its own layer.

    Each tile then gets a Z slider here. With *Sync planes* on, moving one
    tile's slider moves the viewer's Z position, so every tile moves with it.
    With it off, only that tile changes plane.
    """

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self._viewer = napari_viewer
        self._rows: list[tuple[Any, QSlider, QSpinBox]] = []
        self._busy = False

        self.unstitch = QCheckBox("Unstitch tiles")
        self.unstitch.setToolTip(
            "On: every tile of a mosaic is shown as its own layer, and mosaics "
            "opened later are unstitched too.\nOff: the tiles are stitched back "
            "into one mosaic layer."
        )
        self.unstitch.toggled.connect(self._on_unstitch)
        self.sync = QCheckBox("Sync planes")
        self.sync.setChecked(True)
        self.sync.setToolTip(
            "On: moving one tile's Z plane moves every tile's plane by the same "
            "amount.\nOff: each tile's slider moves only that tile. napari's own "
            "Z slider always moves all tiles together."
        )
        self.status = QLabel()
        self.status.setWordWrap(True)

        self._grid = QGridLayout()
        planes = QWidget()
        planes.setLayout(self._grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(planes)
        self._scroll = scroll

        layout = QVBoxLayout(self)
        layout.addWidget(self.unstitch)
        layout.addWidget(self.sync)
        layout.addWidget(self.status)
        layout.addWidget(scroll, 1)

        callbacks = [
            (napari_viewer.layers.events.inserted, self._on_inserted),
            (napari_viewer.layers.events.removed, self._refresh),
            (napari_viewer.dims.events.current_step, self._update_values),
        ]
        for event, cb in callbacks:
            event.connect(cb)

        def _disconnect(*_: Any) -> None:
            for event, cb in callbacks:
                try:
                    event.disconnect(cb)
                except RuntimeError:  # the Qt side is already gone at exit
                    pass

        self.destroyed.connect(_disconnect)
        # Tiles already open (from an earlier widget) mean the toggle starts on.
        if _tiles.tile_layers(napari_viewer):
            self.unstitch.blockSignals(True)
            self.unstitch.setChecked(True)
            self.unstitch.blockSignals(False)
        self._refresh()

    # ------------------------------------------------------------------ #
    # Unstitch / stitch
    # ------------------------------------------------------------------ #

    def _on_unstitch(self, checked: bool) -> None:
        self._busy = True
        try:
            if checked:
                for mosaic in _tiles.mosaic_layers(self._viewer):
                    _tiles.unstitch(self._viewer, mosaic)
            else:
                for mosaic in _tiles.stitched_mosaics(self._viewer):
                    _tiles.stitch(self._viewer, mosaic)
        finally:
            self._busy = False
        self._refresh()

    def _on_inserted(self, event: Any) -> None:
        if not self._busy and self.unstitch.isChecked() and _tiles.is_mosaic(event.value):
            # Changing the layer list from inside its own event is unsafe.
            QTimer.singleShot(0, lambda layer=event.value: self._unstitch_new(layer))
        self._refresh()

    def _unstitch_new(self, layer: Any) -> None:
        if layer in self._viewer.layers and self.unstitch.isChecked():
            self._busy = True
            try:
                _tiles.unstitch(self._viewer, layer)
            finally:
                self._busy = False
            self._refresh()

    # ------------------------------------------------------------------ #
    # Per-tile Z sliders
    # ------------------------------------------------------------------ #

    def _refresh(self, *_: Any) -> None:
        if self._busy:
            return
        tiles = _tiles.tile_layers(self._viewer)
        self.sync.setEnabled(bool(tiles))
        if [row[0] for row in self._rows] != tiles:
            self._rebuild(tiles)
        mosaics = len(_tiles.mosaic_layers(self._viewer))
        if tiles:
            self.status.setText(f"{len(tiles)} tile layer(s). Each slider is that tile's Z plane.")
        elif mosaics:
            self.status.setText(f"{mosaics} mosaic layer(s) can be unstitched.")
        else:
            self.status.setText("No Luxendo mosaic layers open.")
        self._update_values()

    def _rebuild(self, tiles: list[Any]) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = []
        for row, layer in enumerate(tiles):
            n = _tiles.plane_count(layer)
            label = QLabel(layer.name)
            label.setToolTip(layer.name)
            slider = QSlider(Qt.Orientation.Horizontal)
            spin = QSpinBox()
            for w in (slider, spin):
                w.setRange(0, n - 1)
            slider.valueChanged.connect(lambda v, layer=layer: self._on_plane(layer, v))
            spin.valueChanged.connect(lambda v, layer=layer: self._on_plane(layer, v))
            self._grid.addWidget(label, row, 0)
            self._grid.addWidget(slider, row, 1)
            self._grid.addWidget(spin, row, 2)
            self._rows.append((layer, slider, spin))
        self._grid.setColumnStretch(1, 1)
        self._scroll.setVisible(bool(tiles))

    def _update_values(self, *_: Any) -> None:
        point = self._viewer.dims.point
        for layer, slider, spin in self._rows:
            plane = _tiles.shown_plane(layer, point)
            for w in (slider, spin):
                w.blockSignals(True)
                w.setValue(plane)
                w.blockSignals(False)

    def _on_plane(self, layer: Any, plane: int) -> None:
        if self.sync.isChecked():
            _tiles.move_all_to_plane(self._viewer, layer, plane)
        else:
            _tiles.show_plane(layer, plane, self._viewer.dims.point)
        self._update_values()
