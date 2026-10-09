"""Open-and-browse benchmark on a synthetic raw LCS-SPIM mosaic.

    python benchmarks/synthetic_benchmark.py generate DIR [--tiles 2x2] [--tile 192x2048x2048]
    python benchmarks/synthetic_benchmark.py run DIR [--source PATH] [--cold]

``generate`` writes a ``main_raw.lux.h5`` with one channel and one timepoint
whose tiles are uint16, chunked 64x64x64 and have no pyramid levels, like raw
acquisitions. ``run`` opens it in a headless napari viewer and times what a
user waits for: opening, the first displayed slice, stepping Z, and a
full-resolution crop. ``--source`` points at another checkout's ``src``
directory to compare against it; ``--cold`` drops the OS page cache before
each step (Linux, root only).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

VX = (2.0, 0.5, 0.5)  # Z, Y, X in um
OVERLAP = 0.1


def _affine(oy: int, ox: int, shape: tuple[int, int, int]) -> list:
    nz, ny, nx = shape
    tx = VX[2] * (ox + (nx - 1) / 2)
    ty = VX[1] * (oy + (ny - 1) / 2)
    return [
        {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         "translation": [-(nx - 1) / 2, -(ny - 1) / 2, 0.0]},
        {"matrix": [[VX[2], 0, 0], [0, VX[1], 0], [0, 0, -VX[0]]],
         "translation": [tx, ty, 0.0]},
    ]


def generate(root: Path, grid: tuple[int, int], tile: tuple[int, int, int]) -> None:
    import h5py
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = tile
    step = (int(ny * (1 - OVERLAP)), int(nx * (1 - OVERLAP)))
    yy, xx = np.mgrid[0:ny, 0:nx]
    rng = np.random.default_rng(0)
    with h5py.File(root / "main_raw.lux.h5", "w") as main:
        for i in range(grid[0]):
            for j in range(grid[1]):
                oy, ox = step[0] * i, step[1] * j
                stack = f"1-x{i:02d}-y{j:02d}"
                folder = root / "raw" / f"stack_{stack}_channel_0_obj_bottom"
                folder.mkdir(parents=True, exist_ok=True)
                rel = f"raw/{folder.name}/Cam_long_00000.lux.h5"
                proc = {
                    "time_point": "0", "channel": "0", "stack": stack,
                    "objective": "bottom", "camera": "long",
                    "voxel_size_um": {"width": VX[2], "height": VX[1], "depth": VX[0]},
                    "image_size_vx": {"width": nx, "height": ny, "depth": nz},
                    "affine_to_sample": _affine(oy, ox, tile),
                    "channel_description": "Green-488",
                }
                with h5py.File(root / rel, "w") as f:
                    ds = f.create_dataset("Data", shape=tile, dtype="u2", chunks=(64, 64, 64))
                    base = (((yy + oy) // 32 + (xx + ox) // 32) % 2 * 2000 + 500).astype("u2")
                    for z0 in range(0, nz, 64):
                        z1 = min(nz, z0 + 64)
                        noise = rng.integers(0, 200, size=(z1 - z0, ny, nx), dtype="u2")
                        ds[z0:z1] = base[None] + noise
                    f.create_dataset("metadata", data=json.dumps({"processingInformation": proc}))
                view = main.create_group(f"timepoint_0/channel_0_cam_long/raw_stack_{stack}_obj_bottom")
                view["Data"] = h5py.ExternalLink(rel, "Data")
                view["metadata"] = h5py.ExternalLink(rel, "metadata")
                print(f"wrote {rel}", flush=True)


def _drop_caches(enabled: bool) -> None:
    if enabled:
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3\n")


def run(root: Path, cold: bool) -> dict:
    import numpy as np
    from napari.components import ViewerModel
    from napari.layers import Layer

    import napari_luxendo
    from napari_luxendo import close_all, read_luxendo

    report = {"module": napari_luxendo.__file__, "cold_cache": cold}

    def timed(name, fn):
        t = time.perf_counter()
        out = fn()
        report[name + "_s"] = round(time.perf_counter() - t, 3)
        return out

    _drop_caches(cold)
    layers = timed("open", lambda: read_luxendo(str(root / "main_raw.lux.h5"), views="raw"))
    data, kwargs, kind = layers[0]
    levels = data if isinstance(data, list) else [data]
    report["levels"] = [list(a.shape) for a in levels]

    viewer = ViewerModel()
    layer = Layer.create(data, kwargs, kind)
    _drop_caches(cold)
    # Converting the slice forces any pixels napari has not fetched yet.
    shown = timed("first_slice", lambda: (viewer.add_layer(layer), np.asarray(layer._slice.image.raw))[1])
    report["first_slice_shape"] = list(shown.shape)

    _drop_caches(cold)
    step = list(viewer.dims.current_step)
    # Far enough to leave the HDF5 chunk row (and any cached blocks) of the first slice.
    step[0] = (step[0] + viewer.dims.nsteps[0] // 3) % viewer.dims.nsteps[0]
    timed("next_z", lambda: (viewer.dims.__setattr__("current_step", tuple(step)),
                             np.asarray(layer._slice.image.raw))[1])

    full = levels[0]
    zc, yc, xc = (n // 2 for n in full.shape)
    _drop_caches(cold)
    timed("roi_512_full_res", lambda: full[zc, yc - 256:yc + 256, xc - 256:xc + 256].compute())
    _drop_caches(cold)
    timed("plane_full_res", lambda: full[zc].compute())

    viewer.layers.clear()
    close_all()
    report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("dir", type=Path)
    g.add_argument("--tiles", default="2x2")
    g.add_argument("--tile", default="192x2048x2048", help="Z x Y x X voxels per tile")
    r = sub.add_parser("run")
    r.add_argument("dir", type=Path)
    r.add_argument("--source", type=Path, help="src directory of the checkout to benchmark")
    r.add_argument("--cold", action="store_true")
    r.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cmd == "generate":
        generate(args.dir, tuple(int(n) for n in args.tiles.split("x")),
                 tuple(int(n) for n in args.tile.split("x")))
    elif args._child:
        print(json.dumps(run(args.dir, args.cold)))
    else:
        # A fresh interpreter, so the module under test is the one requested.
        env = dict(os.environ)
        if args.source:
            env["PYTHONPATH"] = str(args.source.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, __file__, "run", str(args.dir), "--_child"] + (["--cold"] if args.cold else [])
        out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
        print(json.dumps(json.loads(out.strip().splitlines()[-1]), indent=2))


if __name__ == "__main__":
    main()
