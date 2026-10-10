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
import uuid
from dataclasses import dataclass
from typing import Optional, Sequence

import dask.array as da
import numpy as np

from ._lux import LuxVolume
from ._preview import dataset_identity, preview_levels
from ._tilecache import BLOCK_YX, CachedSource

logger = logging.getLogger(__name__)

# YX block edge of the mosaic dask array. Only requested Z planes are read.
_BLOCK_YX = BLOCK_YX


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
        chunks = (1, min(_BLOCK_YX, shape[1]), min(_BLOCK_YX, shape[2]))
        placed = [
            _Placed(vol, k, g.start, owner_shape=g.stop - g.start)
            for g, vol in zip(grid, tiles)
        ]
        source = _MosaicSource(shape, dtype, placed, voxel_um * f_arr)
        levels.append(
            da.from_array(
                CachedSource(source), chunks=chunks, name='luxendo-mosaic-' + uuid.uuid4().hex,
                asarray=False, fancy=False, meta=np.empty((0, 0, 0), dtype=dtype),
            )
        )
        if len(layout.factors) == 1:
            levels.extend(preview_levels(source))
    return levels


class _MosaicSource:
    """Sliceable mosaic: Dask can fuse crops into reads without making slabs.

    A compact from_array graph also avoids map_blocks' per-block block_info
    dictionaries, which become very large for plane-sized Z chunks.
    """

    ndim = 3

    def __init__(self, shape, dtype, placed, spacing):
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.placed = placed
        self.spacing = spacing

    def __getitem__(self, key):
        # from_array uses positive slice tuples (and empty meta selections).
        key = tuple(slice(*s.indices(n)) for s, n in zip(key, self.shape))
        return _fill_region(key, self.dtype, self.placed, self.spacing)

    def cache_key(self):
        """Identity of the stitched pixels, for the on-disk preview cache."""
        return [list(self.shape), [float(s) for s in self.spacing]] + [
            [None if p.vol is None else dataset_identity(p.vol.datasets[p.level]),
             p.level, p.start.tolist(), p.stop.tolist()]
            for p in self.placed
        ]


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


def _fill_region(key, dtype, placed, spacing):
    coords = [np.arange(s.start, s.stop, s.step) for s in key]
    out = np.zeros(tuple(len(c) for c in coords), dtype=dtype)
    if not out.size:
        return out
    lo = np.array([c[0] for c in coords])
    hi = np.array([c[-1] + 1 for c in coords])
    hits = [p for p in placed if np.all(p.start < hi) and np.all(p.stop > lo)]
    if not hits:
        return out

    if len(hits) == 1:
        p = hits[0]
        if p.vol is not None and len(p.vol.datasets) > p.level:
            bounds = [(int(np.searchsorted(c, a)), int(np.searchsorted(c, b)))
                      for c, a, b in zip(coords, p.start, p.stop)]
            if all(b > a for a, b in bounds):
                dst = tuple(slice(a, b) for a, b in bounds)
                src = tuple(slice(int(c[a] - origin), int(c[b-1] - origin + 1), s.step)
                            for c, (a, b), origin, s in zip(coords, bounds, p.start, key))
                out[dst] = p.vol.read(p.level, src)
        return out

    zs, yy, xx = coords
    ys = yy[:, None]
    xs = xx[None, :]

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

    # Planes whose per-tile Z distances differ only by a constant share one 2D
    # owner map: adding the same value to every tile's distance does not
    # change the nearest one. For an ordinary XY tile grid (all tiles span the
    # same planes) that is every plane of the block.
    shifted = d_z.copy()
    finite = np.isfinite(d_z)
    for z in range(len(zs)):
        if finite[:, z].any():
            shifted[:, z] -= d_z[finite[:, z], z].min()
    patterns: dict[bytes, list[int]] = {}
    for z in range(len(zs)):
        patterns.setdefault(shifted[:, z].tobytes(), []).append(z)

    for planes in patterns.values():
        dz = shifted[:, planes[0]]
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
                    slice(int(c[b.start] - origin), int(c[b.stop-1] - origin + 1), s.step)
                    for b, c, origin, s in zip((bz, by, bx), coords, p.start, key)
                )
                data = p.vol.read(p.level, region)
                out[bz, by, bx][:, sub] = data[:, sub]
    return out
