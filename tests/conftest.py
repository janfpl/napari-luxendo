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
def _close_handles(monkeypatch, tmp_path_factory):
    # Previews stay exact nearest-neighbour samples unless a test turns the
    # background cache on; never write into the real user cache directory.
    monkeypatch.setenv("NAPARI_LUXENDO_PYRAMID_CACHE", "0")
    monkeypatch.setenv("NAPARI_LUXENDO_CACHE_DIR", str(tmp_path_factory.mktemp("preview-cache")))
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


# --------------------------------------------------------------------------- #
# A tiled LCS-SPIM-style experiment, modelled on a real acquisition:
#   raw/stack_1-xII-yJJ_channel_C_obj_bottom/Cam_long_TTTTT.lux.h5
#   main_raw.lux.h5: timepoint_T/channel_C_cam_long/raw_stack_1-xII-yJJ_obj_bottom/{Data,metadata}
# affine_to_sample = [centre XY, diag(2.925, 2.925, -5) + stage translation, identity],
# exactly the chain the microscope writes.
# --------------------------------------------------------------------------- #

TILE = (12, 40, 48)  # Z, Y, X voxels per tile
VX = (5.0, 2.925, 2.925)  # Z, Y, X voxel size (um)
GRID = (2, 3)  # tiles along Y (x-index) and X (y-index)
STEP = (30, 37)  # tile spacing in voxels along Y and X (overlaps 10 and 11)


def tile_affine_to_sample(oy: int, ox: int) -> list:
    """The real LCS-SPIM transform chain for a tile at grid offset (oy, ox) voxels."""
    nz, ny, nx = TILE
    tx = VX[2] * (ox + (nx - 1) / 2) + 1000.0
    ty = VX[1] * (oy + (ny - 1) / 2) - 2000.0
    return [
        {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "translation": [-(nx - 1) / 2, -(ny - 1) / 2, 0.0]},
        {"matrix": [[VX[2], 0, -0.0], [0, VX[1], -0.0], [0, 0, -VX[0]]],
         "translation": [tx, ty, 9500.0]},
        {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 0]},
    ]


def lux_metadata(t: int, channel: int, stack: str, affine: list, description=None) -> str:
    nz, ny, nx = TILE
    proc = {
        "version": "1.0.0",
        "time_point": str(t),
        "channel": str(channel),
        "stack": stack,
        "objective": "bottom",
        "camera": "long",
        "voxel_size_um": {"width": VX[2], "height": VX[1], "depth": VX[0]},
        "image_size_vx": {"width": nx, "height": ny, "depth": nz},
        "affine_to_sample": affine,
        "acquisition": [{"microscope_type": "LCS-SPIM", "stack": stack}],
    }
    if description:
        proc["channel_description"] = description
    return json.dumps({"processingInformation": proc})


def ground_truth(t: int, channel: int) -> np.ndarray:
    """The 'sample' every tile is cut from, on the mosaic voxel grid."""
    shape = (TILE[0], STEP[0] * (GRID[0] - 1) + TILE[1], STEP[1] * (GRID[1] - 1) + TILE[2])
    rng = np.random.default_rng(1000 * t + channel)
    return rng.integers(1, 60000, size=shape, dtype=np.uint16)


@pytest.fixture
def tiled_experiment(tmp_path: Path) -> Path:
    """2 timepoints x 2 channels x (2 x 3) tiles, plus main_raw.lux.h5."""
    root = tmp_path / "2025-08-15_150000"
    root.mkdir(parents=True)
    with h5py.File(root / "main_raw.lux.h5", "w") as main:
        for t in range(2):
            for c in range(2):
                truth = ground_truth(t, c)
                for i in range(GRID[0]):
                    for j in range(GRID[1]):
                        oy, ox = STEP[0] * i, STEP[1] * j
                        stack = f"1-x{i:02d}-y{j:02d}"
                        folder = root / "raw" / f"stack_{stack}_channel_{c}_obj_bottom"
                        folder.mkdir(parents=True, exist_ok=True)
                        rel = f"raw/{folder.name}/Cam_long_{t:05d}.lux.h5"
                        tile = truth[:, oy:oy + TILE[1], ox:ox + TILE[2]]
                        with h5py.File(root / rel, "w") as f:
                            f.create_dataset("Data", data=tile, chunks=(4, 20, 24))
                            f.create_dataset("Data_2_2_2", data=tile[::2, ::2, ::2])
                            f.create_dataset("metadata", data=lux_metadata(
                                t, c, stack, tile_affine_to_sample(oy, ox)))
                        view = main.create_group(
                            f"timepoint_{t}/channel_{c}_cam_long/raw_stack_{stack}_obj_bottom")
                        view["Data"] = h5py.ExternalLink(rel, "Data")
                        view["metadata"] = h5py.ExternalLink(rel, "metadata")
    return root


def centre_sample(vol, factor):
    """Nearest-neighbour preview of *vol*: the middle voxel of each block
    (the last voxel of a clipped edge block)."""
    idx = [np.minimum(np.arange(-(-n // factor)) * factor + factor // 2, n - 1)
           for n in vol.shape]
    return vol[np.ix_(*idx)]
