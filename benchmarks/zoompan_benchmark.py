"""Zoom and pan benchmark in a real (offscreen) napari viewer.

    python benchmarks/synthetic_benchmark.py generate DIR --tiles 2x2 --tile 128x2048x2048
    python benchmarks/zoompan_benchmark.py DIR [--source SRC] [--cold] [--cache-dir D] [--build-cache]

Opens ``DIR/main_raw.lux.h5`` in ``napari.Viewer(show=False)`` with a
1600 x 1000 canvas and times what a user waits for, the way napari does it:
the camera changes, napari picks a level and the visible corners from the
canvas, slices the layer (synchronously, napari's default) and redraws.

* ``zoom_in``: from the whole mosaic to 2 screen pixels per voxel, 1.5x per
  wheel step, towards a point off the centre.
* ``pan_full``: at one screen pixel per voxel, a drag in 1/8-canvas steps
  to the right, down, then back to the left.
* ``pan_level1``: the same at half that zoom.

For each, every step's wait, and the time to load full detail once the
camera stops (when the checkout has a view controller), are reported.
``--cold`` drops the OS page cache before each sequence (Linux, root only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CANVAS = (1000, 1600)  # canvas height, width in screen pixels


def _drop_caches(enabled: bool) -> None:
    if enabled:
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3\n")


def _ms(seconds: float) -> float:
    return round(seconds * 1e3, 1)


def run(root: Path, cold: bool, build_cache: bool, idle: float) -> dict:
    import numpy as np
    import napari

    import napari_luxendo
    from napari_luxendo import read_luxendo

    report: dict = {"module": napari_luxendo.__file__, "cold_cache": cold}
    path = str(root / "main_raw.lux.h5")

    if build_cache:
        try:
            from napari_luxendo._preview_cache import wait_for_builds
        except ImportError:
            report["cache_build_s"] = None
        else:
            [(levels, _, _)] = read_luxendo(path, views="raw")
            t = time.perf_counter()
            for level in levels[1:]:
                np.asarray(level[0, :1, :1])  # first read queues the build
            wait_for_builds()
            report["cache_build_s"] = round(time.perf_counter() - t, 1)
            napari_luxendo.close_all()

    viewer = napari.Viewer(show=False)
    viewer.open(path, plugin="napari-luxendo")
    layer = viewer.layers[0]
    controller = None
    try:
        from napari_luxendo._scroll import install
        controller = install(viewer)
    except ImportError:
        pass
    settle = getattr(controller, "settle", None)
    wait_prefetch = getattr(controller, "wait_prefetch", None)
    report["controller"] = type(controller).__name__ if controller else None

    viewer.dims.set_current_step(0, viewer.dims.nsteps[0] // 3)
    displayed = list(viewer.dims.displayed)
    extent = layer.extent.world[:, displayed]
    voxel = np.linalg.norm(layer.affine.affine_matrix[:-1, :-1], axis=0)[displayed]
    canvas = np.array(CANVAS, dtype=float)

    def draw(centre, world_per_px):
        half = canvas * world_per_px / 2
        corners = np.stack([centre - half, centre + half])
        layer._update_draw(scale_factor=float(world_per_px), corner_pixels_displayed=corners,
                           shape_threshold=CANVAS)

    def camera_moved(centre, world_per_px):
        # The camera events the app emits (a view controller listens to them).
        viewer.camera.center = tuple(float(c) for c in centre)
        viewer.camera.zoom = 1.0 / float(world_per_px)

    def sequence(label, views):
        """Time each (centre, world_per_px) view after the first, then the stop."""
        camera_moved(*views[0])
        draw(*views[0])
        if settle:
            settle()
        draw(*views[0])
        if wait_prefetch:
            t = time.perf_counter()
            wait_prefetch()
            prefetch_s = time.perf_counter() - t
        _drop_caches(cold)
        times, levels = [], []
        for centre, wpp in views[1:]:
            t = time.perf_counter()
            camera_moved(centre, wpp)
            draw(centre, wpp)
            times.append(time.perf_counter() - t)
            levels.append(int(layer.data_level))
            if idle:
                time.sleep(idle)  # the user's hand between wheel ticks / drag events
        out = {"step_ms": [_ms(x) for x in times], "levels": levels,
               "median_ms": _ms(float(np.median(times))), "max_ms": _ms(max(times)),
               "total_ms": _ms(sum(times))}
        if wait_prefetch:
            out["initial_prefetch_ms"] = _ms(prefetch_s)
        if settle:
            t = time.perf_counter()
            settle()
            draw(*views[-1])
            out["full_detail_after_stop_ms"] = _ms(time.perf_counter() - t)
            out["level_after_stop"] = int(layer.data_level)
        report[label] = out

    fit = float(np.max((extent[1] - extent[0]) / canvas))
    full = float(voxel.max())
    target = extent[0] + (extent[1] - extent[0]) * np.array([0.37, 0.62])

    # Zoom towards *target*, keeping it under the cursor, from fit to 0.5 voxel/px.
    centre0 = extent.mean(axis=0)
    views, wpp = [], fit
    while wpp > full / 2:
        frac = wpp / fit
        views.append((target + (centre0 - target) * frac, wpp))
        wpp /= 1.5
    views.append((target, full / 2))
    sequence("zoom_in", views)

    def pan(label, wpp):
        """A drag in 1/8-canvas steps: right, down, then back left."""
        view = canvas * wpp
        lo, hi = extent[0] + view / 2, extent[1] - view / 2  # centres inside the data
        room = np.maximum(hi - lo, 0)
        step = np.minimum(view / 8, room / 6)
        n = np.floor(room / np.maximum(step, 1e-9)).astype(int).clip(max=10)
        pos = lo.copy()
        views = [(pos.copy(), wpp)]
        for axis, sign, count in ((1, 1, n[1]), (0, 1, n[0]), (1, -1, n[1])):
            for _ in range(count):
                pos[axis] += sign * step[axis]
                views.append((pos.copy(), wpp))
        sequence(label, views)

    pan("pan_full", full)
    pan("pan_level1", full * 2)

    viewer.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path)
    parser.add_argument("--source", type=Path, help="src directory of the checkout to benchmark")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--cache-dir", type=Path, help="preview cache directory (default: the user cache)")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--idle", type=float, default=0.05,
                        help="seconds between camera events (default 0.05)")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child:
        print(json.dumps(run(args.dir, args.cold, args.build_cache, args.idle)))
        return
    env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    if args.cache_dir:
        env["NAPARI_LUXENDO_CACHE_DIR"] = str(args.cache_dir.resolve())
    cmd = [sys.executable, __file__, str(args.dir), "--_child", "--idle", str(args.idle)]
    cmd += ["--cold"] if args.cold else []
    cmd += ["--build-cache"] if args.build_cache else []
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
    print(json.dumps(json.loads(out.strip().splitlines()[-1]), indent=2))


if __name__ == "__main__":
    main()
