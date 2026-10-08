"""Dock widget with a toggle for showing Luxendo data in camera coordinates."""

from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

from ._coordinates import CAMERA, SAMPLE, luxendo_layers, set_coordinates


class CoordinatesWidget(QWidget):
    """Switch every Luxendo layer between sample and camera coordinates."""

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self._viewer = napari_viewer

        self.toggle = QCheckBox("Show raw data in camera coordinates")
        self.toggle.setToolTip(
            "On: every Luxendo layer is placed by voxel size only, as the camera "
            "recorded it, so 2D slices are camera planes.\n"
            "Off: layers are placed in sample space (affine_to_sample), so views "
            "overlap; angled raw views are then only correct in 3D."
        )
        self.toggle.toggled.connect(self._apply_all)
        self.status = QLabel()
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toggle)
        layout.addWidget(self.status)
        layout.addStretch(1)

        napari_viewer.layers.events.inserted.connect(self._on_inserted)
        napari_viewer.layers.events.removed.connect(self._update_status)
        self._update_status()

    @property
    def mode(self) -> str:
        return CAMERA if self.toggle.isChecked() else SAMPLE

    def _apply_all(self, *_: Any) -> None:
        layers = luxendo_layers(self._viewer)
        for layer in layers:
            set_coordinates(layer, self.mode)
        if layers:
            self._viewer.reset_view()
        self._update_status()

    def _on_inserted(self, event: Any) -> None:
        # Newly opened data follows the current setting.
        if self.toggle.isChecked():
            set_coordinates(event.value, self.mode)
        self._update_status()

    def _update_status(self, *_: Any) -> None:
        n = len(luxendo_layers(self._viewer))
        where = "camera coordinates" if self.toggle.isChecked() else "sample space"
        self.status.setText(
            f"{n} Luxendo layer(s), shown in {where}." if n else "No Luxendo layers open."
        )
