"""Lazily stitch tiles of one channel into a single mosaic volume.

napari blends per layer, so one layer per tile cannot make tiles of a channel
cover each other while channels add together. Instead, tiles that share the
same orientation and voxel size (the same linear part of ``affine_to_sample``)
are placed on one voxel grid and exposed as one dask array per resolution
level:

* Each tile's offset is its translation expressed in voxels of the shared
  grid, rounded to the nearest voxel (the residual is at most half a voxel and
  is logged).
* Where tiles overlap, a voxel is taken from the tile whose centre is closest
  (in physical units), so seams fall at the middle of each overlap and every
  tile contributes its central, best-quality region.
* Nothing is read until napari asks for a block; each block reads only the
  part of each tile it actually shows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import dask.array as da
import numpy as np

from ._lux import LuxVolume

logger = logging.getLogger(__name__)

# YX block edge of the mosaic dask array (Z follows the tiles' HDF5 chunks).
_BLOCK_YX = 512


@dataclass
class MosaicLayout:
    """Where each tile sits on the shared grid, per resolution level."""

    affine: np.ndarray  # 4x4 mosaic voxel (z, y, x) -> sample space
    offsets: list[np.ndarray]  # full-resolution (z, y, x) offset per tile
    factors: list[tuple[int, int, int]]  # (z, y, x) downsampling per level
    residual_vx: float  # largest rounding error of a tile position, in voxels


def same_linear(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two 4x4 affines have the same linear (rotation/scale) part."""
    return np.allclose(a[:3, :3], b[:3, :3], rtol=1e-6, atol=1e-9)


def plan_layout(reference: Sequence[LuxVolume]) -> MosaicLayout:
    """Compute the shared grid from each tile's reference (first) timepoint."""
    affines = [v.affine for v in reference]
    linear = affines[0][:3, :3]
    inv = np.linalg.inv(linear)
    # Tile origins in grid voxels, relative to the first tile. Only these
    # relative offsets are rounded; the grid origin itself stays exact.
    origins = np.array([inv @ a[:3, 3] for a in affines])
    exact = origins - origins[0]
    rounded = np.round(exact)
    shift = rounded.min(axis=0)

    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = linear @ (origins[0] + shift)

    n_levels = min(len(v.levels) for v in reference)
    factors: list[tuple[int, int, int]] = []
    for k in range(n_levels):
        fk = {tuple(v.factors[k]) for v in reference}
        if len(fk) != 1:
            break
        factors.append(fk.pop())

    return MosaicLayout(
        affine=affine,
        offsets=[(r - shift).astype(np.int64) for r in rounded],
        factors=factors,
        residual_vx=float(np.abs(exact - rounded).max()) if len(exact) else 0.0,
    )


def build_mosaic_levels(
    layout: MosaicLayout,
    reference: Sequence[LuxVolume],
    tiles: Sequence[Optional[LuxVolume]],
) -> list[da.Array]:
    """Return one lazy ``(Z, Y, X)`` mosaic per level for one timepoint.

    The grid (shape, chunks, seams) comes from *reference*, the tiles the
    layout was planned from, so every timepoint has the same shape. *tiles*
    holds this timepoint's data aligned with *reference*; a missing tile (None)
    leaves its region empty (zeros).
    """
    dtype = reference[0].dtype
    voxel_um = np.linalg.norm(layout.affine[:3, :3], axis=0)  # (z, y, x) spacing

    levels = []
    for k, f in enumerate(layout.factors):
        f_arr = np.asarray(f)
        grid = [
            _Placed(ref, k, np.round(off / f_arr).astype(np.int64))
            for off, ref in zip(layout.offsets, reference)
        ]
        shape = tuple(int(x) for x in np.max([g.stop for g in grid], axis=0))
        z_chunk = max(1, min(g.vol.levels[k].chunks[0][0] for g in grid))
        chunks = (min(z_chunk, shape[0]), min(_BLOCK_YX, shape[1]), min(_BLOCK_YX, shape[2]))
        placed = [
            _Placed(vol, k, g.start, owner_shape=g.stop - g.start)
            for g, vol in zip(grid, tiles)
        ]
        template = da.zeros(shape, chunks=chunks, dtype=dtype)
        levels.append(
            template.map_blocks(
                _fill_block, placed=placed, spacing=voxel_um * f_arr, dtype=dtype,
                meta=np.empty((0, 0, 0), dtype=dtype),
            )
        )
    return levels


class _Placed:
    """One tile at one level, positioned on the mosaic grid.

    *vol* may be None (tile missing at this timepoint): it still claims its
    region, which then stays empty, so seams do not move between timepoints.
    """

    def __init__(self, vol: Optional[LuxVolume], level: int, offset: np.ndarray,
                 owner_shape: Optional[np.ndarray] = None) -> None:
        self.vol = vol
        self.level = level
        shape = np.asarray(owner_shape if owner_shape is not None else vol.levels[level].shape)
        self.start = offset
        self.stop = offset + shape
        self.center = offset + (shape - 1) / 2.0


def _fill_block(block: np.ndarray, placed: list[_Placed], spacing: np.ndarray,
                block_info=None) -> np.ndarray:
    (z0, z1), (y0, y1), (x0, x1) = block_info[0]["array-location"]
    lo, hi = np.array([z0, y0, x0]), np.array([z1, y1, x1])
    out = np.zeros(block.shape, dtype=block.dtype)
    hits = [p for p in placed if np.all(p.start < hi) and np.all(p.stop > lo)]
    if not hits:
        return out

    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    zs = np.arange(z0, z1)

    # Squared physical distance to each tile centre, split into an in-plane
    # part (Y, X) and a per-plane Z part; infinite where the tile does not
    # cover the voxel. Ownership is the nearest centre.
    d_yx = np.empty((len(hits), len(ys), xs.shape[1]), dtype=np.float32)
    d_z = np.empty((len(hits), len(zs)), dtype=np.float32)
    for i, p in enumerate(hits):
        inside = (ys >= p.start[1]) & (ys < p.stop[1]) & (xs >= p.start[2]) & (xs < p.stop[2])
        d = ((ys - p.center[1]) * spacing[1]) ** 2 + ((xs - p.center[2]) * spacing[2]) ** 2
        d_yx[i] = np.where(inside, d, np.inf)
        dz = ((zs - p.center[0]) * spacing[0]) ** 2
        d_z[i] = np.where((zs >= p.start[0]) & (zs < p.stop[0]), dz, np.inf)

    # Planes whose per-tile Z distances are identical share one 2D owner map
    # (for an ordinary XY tile grid that is every plane of the block).
    patterns: dict[bytes, list[int]] = {}
    for z in range(len(zs)):
        patterns.setdefault(d_z[:, z].tobytes(), []).append(z)

    for planes in patterns.values():
        dz = d_z[:, planes[0]]
        total = d_yx + dz[:, None, None]
        owner = np.argmin(total, axis=0).astype(np.int16)
        owner[~np.isfinite(total.min(axis=0))] = -1
        z_idx = np.asarray(planes)
        z_runs = np.split(z_idx, np.nonzero(np.diff(z_idx) != 1)[0] + 1)
        for i, p in enumerate(hits):
            mask = owner == i
            if p.vol is None or len(p.vol.datasets) <= p.level or not mask.any():
                continue
            iy = np.nonzero(mask.any(axis=1))[0]
            ix = np.nonzero(mask.any(axis=0))[0]
            by, bx = slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)
            sub = mask[by, bx]
            for run in z_runs:
                bz = slice(int(run[0]), int(run[-1]) + 1)
                region = tuple(
                    slice(int(b.start + l - s), int(b.stop + l - s))
                    for b, l, s in zip((bz, by, bx), lo, p.start)
                )
                data = p.vol.read(p.level, region)
                out[bz, by, bx][:, sub] = data[:, sub]
    return out
