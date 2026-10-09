"""Export panel: write Luxendo layers to ``.lux.h5``, BigTIFF, OME-TIFF or JPEG.

Ported from Shifter's export section. Every ticked Luxendo layer is exported
as shown (a mosaic as its fused volume), one file per layer and timepoint,
either whole or cropped to a box drawn or typed in world coordinates.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._export import (
    FORMAT_JPEG,
    FORMAT_LUX_H5,
    FORMAT_OME_TIFF,
    FORMAT_TIFF,
    METADATA_CSV,
    METADATA_JSON,
    METADATA_TXT,
    ExportPlan,
    jpeg_contrast_limits,
    plan_export,
    run_export,
)
from ._export_layers import (
    export_source,
    exportable_layers,
    layer_timepoints,
    voxel_roi,
    world_extent,
)

logger = logging.getLogger(__name__)

ROI_LAYER_NAME = "Export ROI"
_LIMIT = 1e9  # spin box range for world coordinates (um)


class ExportWorker(QThread):
    """Runs :func:`run_export` off the GUI thread."""

    # object (not int): byte counts of large exports exceed 2**31.
    progress = Signal(object, object)
    export_done = Signal(object)  # summary Path, or None when cancelled
    failed = Signal(str)

    def __init__(self, plan: ExportPlan, ram_percent: int) -> None:
        super().__init__()
        self.plan = plan
        self.ram_percent = ram_percent
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            summary = run_export(
                self.plan,
                ram_percent=self.ram_percent,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancelled,
            )
            self.export_done.emit(summary)
        except MemoryError:
            self.failed.emit(
                "Ran out of memory while exporting.\n\nLower the RAM allocation slider, "
                "close other programs, or export a smaller ROI."
            )
        except Exception:  # reported to the user
            self.failed.emit(traceback.format_exc())


class ExportWidget(QWidget):
    """Dock widget exporting Luxendo layers to ``.lux.h5``, BigTIFF, OME-TIFF or JPEG."""

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._roi_layer: Any = None
        self._worker: Optional[ExportWorker] = None

        lay = QVBoxLayout(self)
        lay.addWidget(self._build_source_section())
        lay.addWidget(self._build_region_section())
        lay.addWidget(self._build_output_section())
        lay.addStretch(1)

        events = self.viewer.layers.events
        callbacks = [
            (events.inserted, self._refresh_layers),
            (events.removed, self._refresh_layers),
        ]
        for event, cb in callbacks:
            event.connect(cb)

        def _disconnect(*_: Any) -> None:
            for event, cb in callbacks:
                event.disconnect(cb)

        self.destroyed.connect(_disconnect)

        self._refresh_layers()
        self._on_format_changed()
        self._on_region_changed()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_source_section(self) -> QGroupBox:
        grp = QGroupBox("Layers")
        lay = QVBoxLayout(grp)

        self.list_layers = QListWidget()
        self.list_layers.setToolTip(
            "Luxendo layers to export. A mosaic is written as its fused volume."
        )
        self.list_layers.setMaximumHeight(110)
        self.list_layers.itemChanged.connect(self._on_selection_changed)
        lay.addWidget(self.list_layers)

        form = QFormLayout()
        row_t = QHBoxLayout()
        self.spin_t_first = QSpinBox()
        self.spin_t_last = QSpinBox()
        self.btn_t_current = QPushButton("Current")
        self.btn_t_current.setToolTip("Export only the timepoint shown on the time slider.")
        self.btn_t_current.clicked.connect(self._set_current_timepoint)
        row_t.addWidget(self.spin_t_first)
        row_t.addWidget(QLabel("to"))
        row_t.addWidget(self.spin_t_last)
        row_t.addWidget(self.btn_t_current)
        form.addRow("Timepoints:", row_t)

        self.combo_format = QComboBox()
        self.combo_format.addItem("Luxendo H5 (.lux.h5)", FORMAT_LUX_H5)
        self.combo_format.addItem("BigTIFF (.tif)", FORMAT_TIFF)
        self.combo_format.addItem("OME-TIFF (.ome.tif)", FORMAT_OME_TIFF)
        self.combo_format.addItem("JPEG, one per Z-plane (8-bit)", FORMAT_JPEG)
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Output format:", self.combo_format)
        lay.addLayout(form)

        self.lbl_format_note = QLabel(
            "JPEG is lossy and 8-bit: each layer is scaled with its current contrast "
            "limits and written as a folder of grayscale images, one per Z-plane."
        )
        self.lbl_format_note.setWordWrap(True)
        lay.addWidget(self.lbl_format_note)
        return grp

    def _build_region_section(self) -> QGroupBox:
        grp = QGroupBox("Export region")
        lay = QVBoxLayout(grp)

        row = QHBoxLayout()
        self.radio_full = QRadioButton("Full volume")
        self.radio_roi = QRadioButton("ROI only")
        self.radio_full.setChecked(True)
        self._region_group = QButtonGroup(self)
        self._region_group.addButton(self.radio_full, 0)
        self._region_group.addButton(self.radio_roi, 1)
        self.radio_full.toggled.connect(self._on_region_changed)
        row.addWidget(self.radio_full)
        row.addWidget(self.radio_roi)
        row.addStretch(1)
        lay.addLayout(row)

        self.roi_box = QWidget()
        roi_lay = QVBoxLayout(self.roi_box)
        roi_lay.setContentsMargins(0, 0, 0, 0)

        row_btns = QHBoxLayout()
        self.btn_draw_roi = QPushButton("Draw ROI rectangle")
        self.btn_draw_roi.setToolTip(
            "Draw a rectangle in the viewer to set the Y/X bounds. The last one drawn is used."
        )
        self.btn_draw_roi.clicked.connect(self._on_draw_roi)
        self.btn_reset_roi = QPushButton("Reset to layer extent")
        self.btn_reset_roi.clicked.connect(self._reset_roi)
        row_btns.addWidget(self.btn_draw_roi)
        row_btns.addWidget(self.btn_reset_roi)
        roi_lay.addLayout(row_btns)

        hint = QLabel("Box in world coordinates (µm, as shown in the viewer):")
        hint.setWordWrap(True)
        roi_lay.addWidget(hint)
        grid = QGridLayout()
        grid.addWidget(QLabel("min"), 0, 1)
        grid.addWidget(QLabel("max"), 0, 2)
        self.roi_spins: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        for r, axis in enumerate("ZYX", start=1):
            lo, hi = QDoubleSpinBox(), QDoubleSpinBox()
            for spin in (lo, hi):
                spin.setRange(-_LIMIT, _LIMIT)
                spin.setDecimals(2)
            grid.addWidget(QLabel(f"{axis}:"), r, 0)
            grid.addWidget(lo, r, 1)
            grid.addWidget(hi, r, 2)
            self.roi_spins[axis] = (lo, hi)
        roi_lay.addLayout(grid)

        row_z = QHBoxLayout()
        self.btn_z_min_here = QPushButton("Z min = current slice")
        self.btn_z_min_here.clicked.connect(lambda: self._set_z_from_slice(upper=False))
        self.btn_z_max_here = QPushButton("Z max = current slice")
        self.btn_z_max_here.clicked.connect(lambda: self._set_z_from_slice(upper=True))
        row_z.addWidget(self.btn_z_min_here)
        row_z.addWidget(self.btn_z_max_here)
        roi_lay.addLayout(row_z)

        lay.addWidget(self.roi_box)
        return grp

    def _build_output_section(self) -> QGroupBox:
        grp = QGroupBox("Export")
        lay = QVBoxLayout(grp)

        row_outdir = QHBoxLayout()
        self.btn_select_outdir = QPushButton("Output directory...")
        self.btn_select_outdir.clicked.connect(self._on_select_output_dir)
        self.edit_outdir = QLineEdit()
        self.edit_outdir.setPlaceholderText("No output directory selected")
        row_outdir.addWidget(self.btn_select_outdir)
        row_outdir.addWidget(self.edit_outdir)
        lay.addLayout(row_outdir)

        row_ram = QHBoxLayout()
        row_ram.addWidget(QLabel("RAM allocation:"))
        self.slider_ram = QSlider(Qt.Horizontal)
        self.slider_ram.setRange(50, 95)
        self.slider_ram.setValue(90)
        self.slider_ram.setTickPosition(QSlider.TicksBelow)
        self.slider_ram.setTickInterval(5)
        self.slider_ram.setToolTip("Share of currently available RAM a Z-slab may use.")
        self.lbl_ram = QLabel("90%")
        self.slider_ram.valueChanged.connect(lambda v: self.lbl_ram.setText(f"{v}%"))
        row_ram.addWidget(self.slider_ram)
        row_ram.addWidget(self.lbl_ram)
        lay.addLayout(row_ram)

        # Pyramids are built from each slab in memory (no reread), so they are
        # cheap enough to write by default. Untick for the fastest export.
        self.chk_write_pyramids = QCheckBox("Write low-resolution pyramid layers (H5 only)")
        self.chk_write_pyramids.setChecked(True)
        self.chk_write_pyramids.setToolTip(
            "Regenerate the Data_W_H_D levels the layer has (as listed in the source files)."
        )
        lay.addWidget(self.chk_write_pyramids)

        self.chk_write_header = QCheckBox("Write Imaris .ims header (H5 only)")
        self.chk_write_header.setChecked(True)
        self.chk_write_header.setToolTip(
            "An .ims file linking every exported layer (as a channel) and timepoint, "
            "for Imaris. Needs all layers to export at the same size."
        )
        lay.addWidget(self.chk_write_header)

        row_meta = QHBoxLayout()
        row_meta.addWidget(QLabel("Metadata summary:"))
        self.combo_metadata = QComboBox()
        self.combo_metadata.addItem("JSON", METADATA_JSON)
        self.combo_metadata.addItem("JSON + TXT", METADATA_TXT)
        self.combo_metadata.addItem("JSON + CSV", METADATA_CSV)
        self.combo_metadata.setToolTip(
            "luxendo_export.json is always written. TXT and CSV add a copy with one "
            "key and value per line."
        )
        row_meta.addWidget(self.combo_metadata, 1)
        lay.addLayout(row_meta)

        row_btns = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self._on_export)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.setVisible(False)
        row_btns.addWidget(self.btn_export)
        row_btns.addWidget(self.btn_cancel)
        lay.addLayout(row_btns)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setWordWrap(True)
        self.lbl_progress.setVisible(False)
        lay.addWidget(self.lbl_progress)
        return grp

    # ------------------------------------------------------------------ #
    # Layer selection
    # ------------------------------------------------------------------ #

    def _items(self) -> list[QListWidgetItem]:
        return [self.list_layers.item(i) for i in range(self.list_layers.count())]

    def selected_layers(self) -> list[Any]:
        """Ticked layers that are still in the viewer, in list order."""
        layers = {id(layer): layer for layer in self.viewer.layers}
        return [
            layers[item.data(Qt.UserRole)]
            for item in self._items()
            if item.checkState() == Qt.Checked and item.data(Qt.UserRole) in layers
        ]

    def _refresh_layers(self, *_: Any) -> None:
        unchecked = {
            item.data(Qt.UserRole) for item in self._items() if item.checkState() != Qt.Checked
        }
        self.list_layers.blockSignals(True)
        self.list_layers.clear()
        for layer in exportable_layers(self.viewer):
            item = QListWidgetItem(layer.name)
            item.setData(Qt.UserRole, id(layer))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # New layers start ticked; keep the user's choice for known ones.
            item.setCheckState(Qt.Unchecked if id(layer) in unchecked else Qt.Checked)
            self.list_layers.addItem(item)
        self.list_layers.blockSignals(False)
        self._on_selection_changed()

    def _on_selection_changed(self, *_: Any) -> None:
        layers = self.selected_layers()
        tps = sorted({t for layer in layers for t in layer_timepoints(layer)})
        lo, hi = (tps[0], tps[-1]) if tps else (0, 0)
        for spin in (self.spin_t_first, self.spin_t_last):
            spin.setRange(lo, hi)
        self.spin_t_first.setValue(lo)
        self.spin_t_last.setValue(hi)
        for w in (self.spin_t_first, self.spin_t_last, self.btn_t_current):
            w.setEnabled(len(tps) > 1)
        if self._roi_untouched():
            self._reset_roi()
        self._update_export_enabled()

    def _set_current_timepoint(self) -> None:
        layers = self.selected_layers()
        layer = next((lay for lay in layers if lay.ndim == 4), None)
        if layer is None:
            return
        frame = int(round(self.viewer.dims.point[0]))
        tps = layer_timepoints(layer)
        t = tps[int(np.clip(frame, 0, len(tps) - 1))]
        self.spin_t_first.setValue(t)
        self.spin_t_last.setValue(t)

    # ------------------------------------------------------------------ #
    # Format / region
    # ------------------------------------------------------------------ #

    def _format(self) -> str:
        return self.combo_format.currentData()

    def _on_format_changed(self, *_: Any) -> None:
        is_h5 = self._format() == FORMAT_LUX_H5
        self.chk_write_pyramids.setEnabled(is_h5)
        self.chk_write_header.setEnabled(is_h5)
        self.lbl_format_note.setVisible(self._format() == FORMAT_JPEG)

    def _on_region_changed(self, *_: Any) -> None:
        self.roi_box.setEnabled(self.radio_roi.isChecked())

    def _update_export_enabled(self) -> None:
        self.btn_export.setEnabled(self._worker is None and bool(self.selected_layers()))

    # ------------------------------------------------------------------ #
    # ROI
    # ------------------------------------------------------------------ #

    def world_box(self) -> tuple[float, ...]:
        """``(z_min, z_max, y_min, y_max, x_min, x_max)`` from the spin boxes."""
        values: list[float] = []
        for axis in "ZYX":
            lo, hi = self.roi_spins[axis]
            values += [lo.value(), hi.value()]
        return tuple(values)

    def _roi_untouched(self) -> bool:
        return all(lo.value() == 0 and hi.value() == 0 for lo, hi in self.roi_spins.values())

    def _reset_roi(self) -> None:
        """Set the box to the union of the selected layers' extents."""
        layers = self.selected_layers()
        if not layers:
            return
        extents = [world_extent(layer) for layer in layers]
        lo = np.min([e[0] for e in extents], axis=0)
        hi = np.max([e[1] for e in extents], axis=0)
        for i, axis in enumerate("ZYX"):
            spin_lo, spin_hi = self.roi_spins[axis]
            spin_lo.setValue(float(np.floor(lo[i] * 100) / 100))
            spin_hi.setValue(float(np.ceil(hi[i] * 100) / 100))

    def _on_draw_roi(self) -> None:
        if self._roi_layer is None or self._roi_layer not in self.viewer.layers:
            # 2-D in world coordinates, so the rectangle applies to every slice.
            self._roi_layer = self.viewer.add_shapes(
                ndim=2, name=ROI_LAYER_NAME, edge_color="yellow",
                face_color="transparent", edge_width=2,
            )
            self._roi_layer.events.data.connect(self._on_roi_shape_changed)
        self.viewer.layers.selection.active = self._roi_layer
        self._roi_layer.mode = "add_rectangle"
        self.radio_roi.setChecked(True)

    def _on_roi_shape_changed(self, *_: Any) -> None:
        shapes = self._roi_layer
        if shapes is None or len(shapes.data) == 0:
            return
        verts = np.asarray(shapes.data[-1], dtype=float)[:, -2:]
        world = verts * np.asarray(shapes.scale[-2:]) + np.asarray(shapes.translate[-2:])
        for col, axis in enumerate("YX"):
            lo, hi = self.roi_spins[axis]
            lo.setValue(float(world[:, col].min()))
            hi.setValue(float(world[:, col].max()))

    def _set_z_from_slice(self, upper: bool) -> None:
        z = float(self.viewer.dims.point[-3])
        lo, hi = self.roi_spins["Z"]
        (hi if upper else lo).setValue(z)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _on_select_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.edit_outdir.text()
        )
        if path:
            self.edit_outdir.setText(path)

    def build_plan(self) -> ExportPlan:
        """Plan the export from the panel's settings (raises ValueError)."""
        layers = self.selected_layers()
        if not layers:
            raise ValueError("Tick at least one layer.")
        outdir = self.edit_outdir.text().strip()
        if not outdir:
            raise ValueError("Select an output directory.")

        sources = []
        for layer in layers:
            roi = None
            if self.radio_roi.isChecked():
                roi = voxel_roi(layer, self.world_box())
                if roi is None:
                    raise ValueError(
                        f"The ROI does not overlap layer '{layer.name}'. "
                        "Untick it or adjust the ROI."
                    )
            sources.append(export_source(layer, roi))
        return plan_export(
            sources,
            Path(outdir),
            fmt=self._format(),
            timepoint_range=(self.spin_t_first.value(), self.spin_t_last.value()),
            write_pyramids=self.chk_write_pyramids.isChecked(),
            write_header=self.chk_write_header.isChecked(),
            metadata_format=self.combo_metadata.currentData(),
        )

    def _confirm_text(self, plan: ExportPlan) -> str:
        lines = [
            f"Format: {self.combo_format.currentText()}",
            f"Output directory: {plan.output_dir}",
            "",
        ]
        seen: set[int] = set()
        for job in plan.jobs:
            src = job.source
            if id(src) in seen:
                continue
            seen.add(id(src))
            nz, ny, nx = src.output_shape
            region = "full volume"
            if src.roi is not None:
                z0, z1, y0, y1, x0, x1 = src.roi
                region = f"voxels X=[{x0},{x1}) Y=[{y0},{y1}) Z=[{z0},{z1})"
            n_files = sum(1 for j in plan.jobs if j.source is src)
            fused = f", fused from {src.fused_views} tiles" if src.fused_views > 1 else ""
            lines.append(f"{src.name}{fused}")
            unit = "folder(s) of JPEGs" if plan.fmt == FORMAT_JPEG else "file(s)"
            lines.append(f"  {region} -> {nx} x {ny} x {nz} (X x Y x Z), {n_files} {unit}")
            if plan.fmt == FORMAT_JPEG:
                lo, hi = jpeg_contrast_limits(src)
                lines.append(f"  8-bit scaling: {lo:g} -> 0, {hi:g} -> 255")
            if plan.fmt == FORMAT_LUX_H5:
                levels = ", ".join(lvl[0] for lvl in job.levels) or "none"
                lines.append(f"  pyramid levels: {levels if plan.write_pyramids else 'disabled'}")
        if plan.header is not None:
            lines.append(f"\nImaris header: {plan.header.name}")
        elif plan.header_note:
            lines.append(f"\nNo Imaris header: {plan.header_note}.")
        size = f"{plan.total_bytes / 1024**3:.2f} GB"
        if plan.fmt == FORMAT_JPEG:  # compressed size is unknown until written
            lines.append(f"\n{len(plan.jobs)} folder(s), at most {size}")
        else:
            lines.append(f"\n{len(plan.jobs)} file(s), estimated {size}")
        lines.append(f"RAM allocation: {self.slider_ram.value()}%")
        return "\n".join(lines)

    def _on_export(self) -> None:
        try:
            plan = self.build_plan()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot export", str(exc))
            return

        existing = [p.name for p in plan.output_paths if p.exists()]
        if existing:
            shown = existing[:15]
            if len(existing) > 15:
                shown.append(f"... and {len(existing) - 15} more")
            ans = QMessageBox.question(
                self, "Overwrite?",
                "These files already exist and will be overwritten:\n"
                + "\n".join(f"  {f}" for f in shown) + "\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        ans = QMessageBox.question(
            self, "Confirm Export", self._confirm_text(plan), QMessageBox.Ok | QMessageBox.Cancel
        )
        if ans != QMessageBox.Ok:
            return
        self.start_export(plan)

    def start_export(self, plan: ExportPlan) -> None:
        self._set_running(True)
        self.lbl_progress.setText("Starting export...")
        self._worker = ExportWorker(plan, self.slider_ram.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.export_done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_progress.setText("Cancelling...")

    def _set_running(self, running: bool) -> None:
        for w in (
            self.list_layers, self.spin_t_first, self.spin_t_last, self.btn_t_current,
            self.combo_format, self.radio_full, self.radio_roi, self.roi_box,
            self.btn_select_outdir, self.edit_outdir, self.slider_ram,
            self.chk_write_pyramids, self.chk_write_header, self.combo_metadata,
        ):
            w.setEnabled(not running)
        if not running:
            self._on_region_changed()
            self._on_format_changed()
            self._on_selection_changed()
        self.btn_cancel.setVisible(running)
        self.btn_cancel.setEnabled(running)
        self.progress_bar.setVisible(running)
        self.progress_bar.setValue(0)
        self.lbl_progress.setVisible(running)
        self.btn_export.setEnabled(not running)

    def _finish_worker(self) -> None:
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self._set_running(False)

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = int(100 * done / total)
            self.progress_bar.setValue(pct)
            self.lbl_progress.setText(
                f"{done / 1024**3:.2f} / {total / 1024**3:.2f} GB written ({pct}%)"
            )

    def _on_done(self, summary: Optional[Path]) -> None:
        self._finish_worker()
        if summary is None:
            QMessageBox.information(
                self, "Cancelled",
                "Export was cancelled. Files already finished were kept; the partial one "
                "was removed.",
            )
            return
        out_dir = Path(summary).parent
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Export Complete")
        box.setText(f"Export finished.\nOutput folder: {out_dir}")
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
        close_btn = box.addButton("Close", QMessageBox.RejectRole)
        box.setDefaultButton(close_btn)
        box.exec() if hasattr(box, "exec") else box.exec_()
        if box.clickedButton() is open_btn:
            _open_in_file_browser(out_dir)

    def _on_failed(self, message: str) -> None:
        self._finish_worker()
        QMessageBox.critical(self, "Export Error", message)


def _open_in_file_browser(path: Path) -> None:
    """Reveal *path* in the OS file browser."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError as exc:
        logger.warning("Could not open %s in file browser: %s", path, exc)
