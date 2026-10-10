"""Z-scroll benchmark in a real (offscreen) napari viewer.

    python benchmarks/synthetic_benchmark.py generate DIR --tiles 2x2 --tile 128x2048x2048
    python benchmarks/zscroll_benchmark.py DIR [--source SRC] [--cold] [--cache-dir D] [--build-cache]

Opens ``DIR/main_raw.lux.h5`` in ``napari.Viewer(show=False)`` with a
1600 x 1000 canvas and times what a user waits for per Z step, the way napari
does it: the slider moves, the layer is sliced (synchronously, napari's
default) and the canvas redraws. Two views are measured: the whole mosaic
(zoomed out) and a full-resolution crop (zoomed in). After the steps, the
time to load full detail once scrolling stops is measured too (only when the
checkout has the scroll preview).

It also times whole-plane reads per display level, bypassing napari.

``--source`` benchmarks another checkout's ``src``; ``--cold`` drops the OS
page cache before each measurement (Linux, root only); ``--build-cache``
first builds the preview cache and reports how long that took. Needs a Qt
binding (it runs with ``QT_QPA_PLATFORM=offscreen``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = 12
CANVAS = (1000, 1600)  # canvas height, width in screen pixels


def _drop_caches(enabled: bool) -> None:
    if enabled:
        os.sync()
        Path("/proc/sys/vm/drop_caches").write_text("3\n")


def _ms(seconds: float) -> float:
    return round(seconds * 1e3, 1)


def run(root: Path, cold: bool, build_cache: bool) -> dict:
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

    # Whole-plane reads per display level, outside napari.
    [(levels, _, _)] = read_luxendo(path, views="raw")
    plane = {}
    for k, level in enumerate(levels):
        nz = level.shape[0]
        _drop_caches(cold)
        times = []
        for z in range(nz // 3, nz // 3 + 6):
            t = time.perf_counter()
            np.asarray(level[z])
            times.append(time.perf_counter() - t)
        plane[f"level{k} {'x'.join(map(str, level.shape[1:]))}"] = {
            "first_ms": _ms(times[0]), "next_median_ms": _ms(float(np.median(times[1:])))}
    report["plane_read"] = plane
    napari_luxendo.close_all()

    viewer = napari.Viewer(show=False)
    viewer.open(path, plugin="napari-luxendo")
    layer = viewer.layers[0]
    try:
        from napari_luxendo._scroll import install
        controller = install(viewer)
    except ImportError:
        controller = None

    # What a 1600 x 1000 canvas redraw does: napari picks the level and the
    # visible corners from the canvas size and the world region on screen.
    displayed = list(viewer.dims.displayed)
    extent = layer.extent.world[:, displayed]
    pixel = np.linalg.norm(layer.affine.affine_matrix[:-1, :-1], axis=0)[displayed]
    view = {"corners": extent}

    def draw():
        layer._update_draw(scale_factor=1.0, corner_pixels_displayed=view["corners"],
                           shape_threshold=CANVAS)

    def scroll(label):
        viewer.dims.set_current_step(0, viewer.dims.nsteps[0] // 3)
        draw()
        if controller is not None:
            controller.settle()
        draw()
        fine = int(layer.data_level)
        _drop_caches(cold)
        times, levels_seen = [], set()
        z0 = viewer.dims.current_step[0]
        for i in range(1, STEPS + 1):
            t = time.perf_counter()
            viewer.dims.set_current_step(0, z0 + i)
            draw()
            times.append(time.perf_counter() - t)
            levels_seen.add(int(layer.data_level))
        out = {"level_when_still": fine, "levels_while_scrolling": sorted(levels_seen),
               "first_step_ms": _ms(times[0]), "step_median_ms": _ms(float(np.median(times))),
               "step_max_ms": _ms(max(times))}
        if controller is not None:
            t = time.perf_counter()
            controller.settle()
            draw()
            out["full_detail_after_stop_ms"] = _ms(time.perf_counter() - t)
            out["level_after_stop"] = int(layer.data_level)
        report[label] = out

    scroll("overview_scroll")  # the whole mosaic on screen

    # Zoomed in: one canvas pixel per full-resolution pixel, at the centre.
    centre = extent.mean(axis=0)
    half = np.array(CANVAS) * pixel / 2
    view["corners"] = np.stack([centre - half, centre + half])
    scroll("zoomed_scroll")

    viewer.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=Path)
    parser.add_argument("--source", type=Path, help="src directory of the checkout to benchmark")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--cache-dir", type=Path, help="preview cache directory (default: a fresh one)")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child:
        print(json.dumps(run(args.dir, args.cold, args.build_cache)))
        return
    env = dict(os.environ, QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    if args.source:
        env["PYTHONPATH"] = str(args.source.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    if args.cache_dir:
        env["NAPARI_LUXENDO_CACHE_DIR"] = str(args.cache_dir.resolve())
    cmd = [sys.executable, __file__, str(args.dir), "--_child"]
    cmd += ["--cold"] if args.cold else []
    cmd += ["--build-cache"] if args.build_cache else []
    out = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True).stdout
    print(json.dumps(json.loads(out.strip().splitlines()[-1]), indent=2))


if __name__ == "__main__":
    main()
