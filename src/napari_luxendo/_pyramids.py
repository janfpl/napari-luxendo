"""Streaming resolution-pyramid generation for Luxendo .lux.h5 exports.

Ported from Shifter's exporter. Every pyramid level is built from the
full-resolution slabs while they are still in memory, so the exported
``Data`` is never read back.

Luxendo files store pyramid levels next to the full-resolution ``Data``
dataset as ``Data_W_H_D``, where W/H/D are the X/Y/Z downsample factors.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Regex for pyramid dataset names: Data_W_H_D (all integers).
_PYRAMID_RE = re.compile(r"^Data_(\d+)_(\d+)_(\d+)$")


def worker_count() -> int:
    """Threads to use for parallel work, leaving a few cores for napari/OS."""
    total = os.cpu_count() or 1
    reserved = min(4, total // 2)
    return max(1, total - reserved)


def level_factors(name: str) -> tuple[int, int, int] | None:
    """``(factor_w, factor_h, factor_d)`` of a ``Data_W_H_D`` level name, else None."""
    m = _PYRAMID_RE.match(name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def compute_pyramid_level_shape(
    data_shape: tuple[int, int, int],
    factor_w: int,
    factor_h: int,
    factor_d: int,
) -> tuple[int, int, int]:
    """Return the (nz, ny, nx) shape a pyramid level of *data_shape* has."""
    nz, ny, nx = data_shape
    return nz // factor_d, ny // factor_h, nx // factor_w


def pyramid_sum_dtype(factor_w: int, factor_h: int, factor_d: int) -> np.dtype:
    """Smallest unsigned dtype that can hold a block sum without overflow."""
    if 65535 * factor_w * factor_h * factor_d <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


# --------------------------------------------------------------------------- #
# XY block-sum backends
#
# The XY reduction of the full-resolution slab dominates pyramid compute.
# Integer addition is associative and pyramid_sum_dtype() guarantees the
# accumulator cannot overflow, so any evaluation order yields bit-identical
# sums. numba (optional) parallelises across cores; numpy is the fallback.
# --------------------------------------------------------------------------- #

# Below this many output elements the thread overhead outweighs the win.
_PARALLEL_MIN_ELEMS = 1 << 16

try:  # pragma: no cover - exercised only when numba is installed
    import numba

    @numba.njit(parallel=True, cache=True)
    def _block_sum_xy_numba(planes, factor_h, factor_w, out):
        """Fill *out* with the (factor_h, factor_w) XY block sums of *planes*."""
        n_planes = out.shape[0]
        oh = out.shape[1]
        ow = out.shape[2]
        for idx in numba.prange(n_planes * oh):
            p = idx // oh
            oy = idx - p * oh
            y0 = oy * factor_h
            for dy in range(factor_h):
                row = planes[p, y0 + dy]
                for ox in range(ow):
                    x0 = ox * factor_w
                    acc = 0
                    for dx in range(factor_w):
                        acc += row[x0 + dx]
                    out[p, oy, ox] += acc

    _HAVE_NUMBA = True
    _NUMBA_UNAVAILABLE_REASON = ""
except Exception as _exc:  # noqa: BLE001 - numba missing or failed to load
    _HAVE_NUMBA = False
    _NUMBA_UNAVAILABLE_REASON = f"{type(_exc).__name__}: {_exc}"

_numba_threads_set = False


def _ensure_numba_threads() -> None:
    """Limit numba's thread pool to :func:`worker_count` (once per process)."""
    global _numba_threads_set
    if _numba_threads_set or not _HAVE_NUMBA:
        return
    _numba_threads_set = True
    try:
        n = max(1, min(worker_count(), numba.config.NUMBA_NUM_THREADS))
        numba.set_num_threads(n)
    except Exception as exc:  # noqa: BLE001 - never break the export
        logger.warning("Could not set numba thread count: %s", exc)


def pyramid_backend_status() -> str:
    """Human-readable description of the XY-reduction backend."""
    if _HAVE_NUMBA:
        return "numba (multi-threaded)"
    reason = _NUMBA_UNAVAILABLE_REASON or "numba not available"
    return (
        f"numpy (single-threaded; {reason}). Install numba for a faster "
        "pyramid reduction."
    )


def _block_sum_xy(
    planes: np.ndarray, factor_h: int, factor_w: int, sum_dtype: np.dtype
) -> np.ndarray:
    """Sum ``(factor_h, factor_w)`` XY blocks of every plane in *planes*."""
    n, h, w = planes.shape
    oh, ow = h // factor_h, w // factor_w

    if _HAVE_NUMBA and n * oh * ow >= _PARALLEL_MIN_ELEMS:
        _ensure_numba_threads()
        out = np.zeros((n, oh, ow), dtype=sum_dtype)
        # Numba indexes only inside the trimmed region, so no copy is needed
        # even when the factors do not divide the plane dimensions.
        _block_sum_xy_numba(planes, factor_h, factor_w, out)
        return out

    trimmed = planes[:, : oh * factor_h, : ow * factor_w]
    if not trimmed.flags["C_CONTIGUOUS"]:
        trimmed = np.ascontiguousarray(trimmed)
    return trimmed.reshape(n, oh, factor_h, ow, factor_w).sum(
        axis=(2, 4), dtype=sum_dtype
    )


# --------------------------------------------------------------------------- #
# Streaming pyramid generation
#
# Two properties keep this cheap:
#
#   * Sums are accumulated as integers rather than float64.
#   * When one level's factors divide another's componentwise (the usual
#     2/4/8 Luxendo ladder), the coarser level is derived from the finer
#     level's *unrounded* sums instead of from the full-resolution slab again.
#     Summation is associative, so this is bit-identical for a fraction of the
#     work; non-divisible ladders fall back to summing the raw slab.
#
# Slab boundaries need not align to the depth factors: each level keeps a carry
# accumulator holding the partial Z-group that straddles two slabs.
# --------------------------------------------------------------------------- #


class _PyramidLevelState:
    """Per-level bookkeeping for :class:`StreamingPyramidWriter`."""

    def __init__(
        self,
        name: str,
        factors: tuple[int, int, int],
        out_shape: tuple[int, int, int],
        parent: int | None,
        rel_factors: tuple[int, int, int],
        dataset: Any,
    ) -> None:
        self.name = name
        self.fw, self.fh, self.fd = factors
        self.out_shape = out_shape
        self.parent = parent
        self.rfw, self.rfh, self.rfd = rel_factors
        self.ds = dataset
        self.divisor = self.fw * self.fh * self.fd
        self.sum_dtype = pyramid_sum_dtype(self.fw, self.fh, self.fd)
        self.carry: np.ndarray | None = None  # partial Z-group across slabs
        self.carry_n = 0
        self.next_oz = 0


class StreamingPyramidWriter:
    """Generate Luxendo pyramid levels from full-resolution slabs.

    Create it inside the open output file *before* the first slab is written,
    feed every output slab to :meth:`consume` in Z order, then call
    :meth:`finish`.

    Parameters
    ----------
    out_h5 : h5py.File
        Open, writable output file.
    levels : list[tuple[str, int, int, int]]
        ``(name, factor_w, factor_h, factor_d)`` per level, as returned by
        :func:`napari_luxendo._lux.detect_pyramid_levels`.
    output_shape_zyx : tuple[int, int, int]
        Shape of the full-resolution output.
    chunks_by_name : dict, optional
        Per-level chunk shape to mirror the source file's chunking.
    """

    def __init__(
        self,
        out_h5: Any,
        levels: list[tuple[str, int, int, int]],
        output_shape_zyx: tuple[int, int, int],
        chunks_by_name: dict[str, tuple[int, ...] | None] | None = None,
    ) -> None:
        self.output_shape = output_shape_zyx
        self.levels: list[_PyramidLevelState] = []
        self._expected_z = 0
        self.compute_s = 0.0
        self.write_s = 0.0
        self.bytes_written = 0
        self.skipped: list[str] = []

        chunks_by_name = chunks_by_name or {}
        nz, ny, nx = output_shape_zyx

        # Order by total downsample factor so a level's parent is always
        # built before it.
        ordered = sorted(levels, key=lambda t: t[1] * t[2] * t[3])
        specs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]] = (
            []
        )
        for name, fw, fh, fd in ordered:
            out_shape = (nz // fd, ny // fh, nx // fw)
            if any(d <= 0 for d in out_shape):
                logger.warning(
                    "Pyramid level %s: output dimensions would be zero, "
                    "skipping.",
                    name,
                )
                self.skipped.append(name)
                continue
            specs.append((name, (fw, fh, fd), out_shape))

        for idx, (name, (fw, fh, fd), out_shape) in enumerate(specs):
            parent, rel = self._pick_parent(specs, idx)
            chunks = chunks_by_name.get(name)
            if chunks is None:
                chunks = tuple(min(64, d) for d in out_shape)
            else:
                chunks = tuple(
                    min(c, d) for c, d in zip(chunks, out_shape)
                )
            if name in out_h5:
                del out_h5[name]
            ds = out_h5.create_dataset(
                name, shape=out_shape, dtype=np.uint16, chunks=chunks
            )
            self.levels.append(
                _PyramidLevelState(
                    name, (fw, fh, fd), out_shape, parent, rel, ds
                )
            )

    @staticmethod
    def _pick_parent(
        specs: list[tuple[str, tuple[int, int, int], tuple[int, int, int]]],
        idx: int,
    ) -> tuple[int | None, tuple[int, int, int]]:
        """Choose the coarsest earlier level whose factors divide this one's.

        Returns ``(parent_index_or_None, relative_factors)``. With no usable
        parent the level is built straight from the full-resolution slab.
        """
        _, (fw, fh, fd), _ = specs[idx]
        best: int | None = None
        best_total = 0
        for j in range(idx):
            _, (pw, ph, pd), _ = specs[j]
            if fw % pw or fh % ph or fd % pd:
                continue
            total = pw * ph * pd
            if total > best_total:
                best, best_total = j, total
        if best is None:
            return None, (fw, fh, fd)
        _, (pw, ph, pd), _ = specs[best]
        return best, (fw // pw, fh // ph, fd // pd)

    def consume(self, slab: np.ndarray, z_start: int) -> int:
        """Feed one full-resolution output slab; returns pyramid bytes written."""
        if z_start != self._expected_z:
            raise ValueError(
                "pyramid slabs must arrive in Z order: "
                f"expected z={self._expected_z}, got {z_start}"
            )
        self._expected_z = z_start + slab.shape[0]

        streams: dict[int, np.ndarray | None] = {}
        bytes_before = self.bytes_written
        for idx, level in enumerate(self.levels):
            source = (
                slab if level.parent is None else streams.get(level.parent)
            )
            if source is None or len(source) == 0:
                streams[idx] = None
                continue
            t0 = time.perf_counter()
            completed = self._accumulate(level, source)
            self.compute_s += time.perf_counter() - t0
            streams[idx] = completed
            if completed is not None and len(completed):
                self._write(level, completed)
        return self.bytes_written - bytes_before

    def _accumulate(
        self, level: _PyramidLevelState, source: np.ndarray
    ) -> np.ndarray | None:
        """Reduce *source* planes into completed absolute-sum output planes."""
        xy = _block_sum_xy(source, level.rfh, level.rfw, level.sum_dtype)
        n = xy.shape[0]
        rfd = level.rfd
        pieces: list[np.ndarray] = []
        idx = 0

        # Finish a group left partially accumulated by the previous slab.
        if level.carry is not None:
            take = min(rfd - level.carry_n, n)
            if take:
                level.carry += xy[:take].sum(axis=0, dtype=level.sum_dtype)
                level.carry_n += take
                idx = take
            if level.carry_n == rfd:
                pieces.append(level.carry)
                level.carry, level.carry_n = None, 0

        # Whole groups fully contained in this slab.
        full = (n - idx) // rfd
        if full:
            grouped = (
                xy[idx : idx + full * rfd]
                .reshape(full, rfd, xy.shape[1], xy.shape[2])
                .sum(axis=1, dtype=level.sum_dtype)
            )
            pieces.extend(grouped)
            idx += full * rfd

        # Remainder becomes the carry for the next slab.
        if idx < n:
            level.carry = xy[idx:].sum(axis=0, dtype=level.sum_dtype)
            level.carry_n = n - idx

        if not pieces:
            return None
        return np.stack(pieces)

    def _write(self, level: _PyramidLevelState, completed: np.ndarray) -> None:
        """Write completed planes as one batched HDF5 slice."""
        onz = level.out_shape[0]
        start = level.next_oz
        end = min(start + completed.shape[0], onz)
        if end <= start:
            return  # beyond the declared level depth; trailing block dropped
        block = (completed[: end - start] // level.divisor).astype(np.uint16)
        t0 = time.perf_counter()
        level.ds[start:end] = block
        self.write_s += time.perf_counter() - t0
        level.next_oz = end
        self.bytes_written += int(block.nbytes)

    def finish(self) -> dict[str, Any]:
        """Discard incomplete trailing groups and report what was written."""
        incomplete = []
        for level in self.levels:
            if level.next_oz != level.out_shape[0]:
                incomplete.append(
                    f"{level.name}: wrote {level.next_oz}/"
                    f"{level.out_shape[0]} planes"
                )
            level.carry, level.carry_n = None, 0
        if incomplete:
            logger.warning(
                "Pyramid levels incomplete: %s", "; ".join(incomplete)
            )
        return {
            "levels": [lv.name for lv in self.levels],
            "skipped": list(self.skipped),
            "compute_s": self.compute_s,
            "write_s": self.write_s,
            "bytes_written": self.bytes_written,
            "incomplete": incomplete,
        }
