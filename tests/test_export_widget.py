"""The export dock widget; skipped without a Qt binding and pytest-qt."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("pytestqt")
pytest.importorskip("qtpy.QtWidgets")

from napari_luxendo._export_widget import ExportWidget  # noqa: E402

from conftest import ground_truth  # noqa: E402


@pytest.fixture
def viewer(make_napari_viewer, tiled_experiment):
    viewer = make_napari_viewer()
    viewer.open(str(tiled_experiment / "main_raw.lux.h5"), plugin="napari-luxendo")
    return viewer


def test_lists_luxendo_layers_and_timepoints(viewer):
    viewer.add_image(np.zeros((4, 4, 4)), name="not luxendo")
    widget = ExportWidget(viewer)
    assert widget.list_layers.count() == 2
    assert len(widget.selected_layers()) == 2
    assert (widget.spin_t_first.value(), widget.spin_t_last.value()) == (0, 1)
    assert widget.btn_export.isEnabled()

    # The default ROI is the layers' full extent.
    widget.radio_roi.setChecked(True)
    widget.edit_outdir.setText("/nonexistent/out")
    plan = widget.build_plan()
    shape = ground_truth(0, 0).shape
    assert {job.source.roi for job in plan.jobs} == {(0, shape[0], 0, shape[1], 0, shape[2])}

    widget.list_layers.item(1).setCheckState(0)  # untick channel 1
    assert [layer.name for layer in widget.selected_layers()] == [viewer.layers[0].name]


def test_drawn_rectangle_sets_world_yx(viewer):
    widget = ExportWidget(viewer)
    widget._on_draw_roi()
    assert widget.radio_roi.isChecked()
    widget._roi_layer.add(
        np.array([[-1900.0, 1100.0], [-1900.0, 1200.0], [-1850.0, 1200.0], [-1850.0, 1100.0]]),
        shape_type="rectangle",
    )
    box = widget.world_box()
    assert box[2:] == pytest.approx((-1900.0, -1850.0, 1100.0, 1200.0))


def test_export_runs_in_background(viewer, tmp_path, qtbot, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    shown = []  # the completion dialog is modal; record it instead
    for name in ("exec", "exec_"):
        monkeypatch.setattr(QMessageBox, name, lambda box: shown.append(box.text()), raising=False)

    widget = ExportWidget(viewer)
    widget.edit_outdir.setText(str(tmp_path / "out"))
    widget.spin_t_first.setValue(1)
    widget.start_export(widget.build_plan())
    assert not widget.btn_export.isEnabled()

    qtbot.waitUntil(lambda: widget._worker is None, timeout=30000)
    assert widget.btn_export.isEnabled()
    assert shown and "Export finished" in shown[0]
    assert len(list((tmp_path / "out").glob("*_tp-1.lux.h5"))) == 2
    assert not list((tmp_path / "out").glob("*_tp-0.lux.h5"))
