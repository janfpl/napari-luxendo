"""Synthetic Luxendo data: channel files plus Imaris and BigDataViewer headers."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import napari_luxendo._lux as lux

SHAPE = (24, 40, 48)  # Z, Y, X
VOXEL = {"width": 0.4, "height": 0.5, "depth": 2.0}
DESCRIPTIONS = ["Green-488", "Red-561"]


@pytest.fixture(autouse=True)
def _close_handles():
    yield
    lux.close_all()


def _ims_text(value: object) -> np.ndarray:
    return np.frombuffer(str(value).encode("ascii"), dtype="S1")


def make_volume(seed: int, shape=SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(100, 5000, size=shape, dtype=np.uint16)


def write_lux(path: Path, data: np.ndarray, description: str | None = "Green-488",
              pyramids=((2, 2, 2), (4, 4, 4)), voxel=VOXEL) -> Path:
    with h5py.File(path, "w") as f:
        f.create_dataset("Data", data=data, chunks=(8, 20, 24))
        for fw, fh, fd in pyramids:
            f.create_dataset(f"Data_{fw}_{fh}_{fd}", data=data[::fd, ::fh, ::fw])
        proc = {"image_size_vx": {"width": data.shape[-1], "height": data.shape[-2],
                                  "depth": data.shape[0]}}
        if voxel is not None:
            proc["voxel_size_um"] = voxel
        if description is not None:
            proc["channel_description"] = description
            proc["channel_id"] = 1
        f.create_dataset("metadata", data=json.dumps({"processingInformation": proc}))
    return path


@pytest.fixture
def lux_dir(tmp_path: Path) -> Path:
    """Two channels x two timepoints, plus .ims and BDV headers linking them."""
    files = {}
    for t in range(2):
        for c, desc in enumerate(DESCRIPTIONS):
            p = tmp_path / f"uni_tp-{t}_ch-{c}.lux.h5"
            files[t, c] = write_lux(p, make_volume(10 * t + c), desc)

    with h5py.File(tmp_path / "dataset.ims", "w") as f:
        f.attrs["ImarisDataSet"] = _ims_text("ImarisDataSet")
        for lvl, path in enumerate(["/Data", "/Data_2_2_2"]):
            for (t, c), p in files.items():
                g = f.create_group(f"DataSet/ResolutionLevel {lvl}/TimePoint {t}/Channel {c}")
                g["Data"] = h5py.ExternalLink(p.name, path)
        img = f.create_group("DataSetInfo/Image")
        nz, ny, nx = SHAPE
        for axis, (dim, n, vox) in enumerate(
            [("X", nx, 0.4), ("Y", ny, 0.5), ("Z", nz, 2.0)]
        ):
            img.attrs[dim] = _ims_text(n)
            img.attrs[f"ExtMin{axis}"] = _ims_text(0)
            img.attrs[f"ExtMax{axis}"] = _ims_text(n * vox)
        for c, (name, color) in enumerate([("GFP", "0 1 0"), ("mCherry", "1 0 1")]):
            g = f.create_group(f"DataSetInfo/Channel {c}")
            g.attrs["Name"] = _ims_text(name)
            g.attrs["Color"] = _ims_text(color)

    with h5py.File(tmp_path / "dataset_bdv.h5", "w") as f:
        for (t, c), p in files.items():
            g = f.create_group(f"t{t:05d}/s{c:02d}/0")
            g["cells"] = h5py.ExternalLink(p.name, "/Data")
        for c in range(2):
            f.create_dataset(f"s{c:02d}/resolutions", data=np.ones((1, 3)))
    (tmp_path / "dataset_bdv.xml").write_text(
        "<SpimData><SequenceDescription><ViewSetups>"
        "<ViewSetup><id>0</id><name>ch-488</name>"
        "<voxelSize><unit>um</unit><size>0.4 0.5 2.0</size></voxelSize></ViewSetup>"
        "<ViewSetup><id>1</id><name>ch-561</name></ViewSetup>"
        "</ViewSetups></SequenceDescription></SpimData>"
    )
    return tmp_path
