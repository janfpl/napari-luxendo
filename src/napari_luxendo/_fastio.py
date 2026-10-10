"""Exact, lazy reads of unfiltered HDF5 chunks through a read-only mapping.

Addresses are obtained from HDF5, never inferred from allocation order.
Moderate chunk indexes are visited once on the first read; enormous indexes
are queried per requested chunk. Compressed, virtual, external-storage and
unusual datasets use h5py. No pixel data is read during construction.
"""
from __future__ import annotations

import itertools
import mmap
import os
import threading

import numpy as np

_READERS = {}
_LOCK = threading.RLock()


def reader_for(ds):
    with _LOCK:
        reader = _READERS.get(ds)
        if reader is None:
            reader = _READERS[ds] = ChunkReader(ds)
        return reader


def close_readers():
    with _LOCK:
        for reader in _READERS.values():
            reader.close()
        _READERS.clear()


class ChunkReader:
    def __init__(self, ds):
        import h5py
        self.ds = ds
        self.shape, self.dtype, self.ndim = ds.shape, ds.dtype, ds.ndim
        self._lock = threading.RLock()
        self._mapping = None
        self._offsets = {}
        self._index = None
        dcpl = ds.id.get_create_plist()
        self._direct = (
            os.environ.get('NAPARI_LUXENDO_DIRECT_IO', '1') != '0'
            and ds.ndim == 3 and ds.chunks is not None
            and ds.dtype.kind in 'buifc' and not ds.dtype.hasobject
            and ds.id.get_type().equal(h5py.h5t.py_create(ds.dtype))
            and dcpl.get_nfilters() == 0 and dcpl.get_external_count() == 0
            and not ds.is_virtual and ds.file.driver in ('sec2', 'stdio', 'windows')
            and hasattr(ds.id, 'get_chunk_info_by_coord')
        )

    def close(self):
        with self._lock:
            if self._mapping is not None:
                try:
                    self._mapping.close()
                except BufferError:
                    pass  # a read in another thread still holds a view; GC closes it
                self._mapping = None
            self._offsets.clear()
            self._index = None

    def _build_index(self, chunk_bytes):
        # H5Dget_chunk_info_by_coord repeatedly traverses this acquisition's
        # v1 B-tree. A single chunk_iter traversal is much faster, and a dense
        # 45,056-entry int64 index takes just 352 KiB per real tile.
        grid = tuple(-(-n // c) for n, c in zip(self.shape, self.ds.chunks))
        if self._index is not None or not hasattr(self.ds.id, 'chunk_iter') \
                or np.prod(grid) > 1_000_000:
            return
        index = np.full(grid, -1, dtype=np.int64)
        chunks = self.ds.chunks
        bad = []

        def visit(info):
            if info.byte_offset is None:
                return
            if info.filter_mask != 0 or info.size != chunk_bytes \
                    or info.byte_offset + chunk_bytes > len(self._mapping):
                bad.append(True)
                return
            z, y, x = info.chunk_offset
            index[z // chunks[0], y // chunks[1], x // chunks[2]] = info.byte_offset

        self.ds.id.chunk_iter(visit)
        if bad:
            self._direct = False
        else:
            self._index = index

    def __getitem__(self, key):
        # Unsupported indexing is left to h5py (Dask handles advanced indexing).
        if not self._direct or not isinstance(key, tuple) or len(key) != 3 \
                or any(not isinstance(s, slice) or (s.step or 1) <= 0 for s in key):
            return np.asarray(self.ds[key])
        key = tuple(slice(*s.indices(n)) for s, n in zip(key, self.shape))
        counts = tuple(len(range(s.start, s.stop, s.step)) for s in key)
        if not all(counts):
            return np.empty(counts, dtype=self.dtype)
        with self._lock:
            if not self.ds.id.valid:
                raise ValueError('Luxendo source was closed')
            if self._mapping is None:
                try:
                    with open(self.ds.file.filename, 'rb') as handle:
                        self._mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                except (OSError, ValueError):
                    self._direct = False
                    return np.asarray(self.ds[key])
            chunks = self.ds.chunks
            chunk_bytes = int(np.prod(chunks)) * self.dtype.itemsize
            self._build_index(chunk_bytes)
            if not self._direct:
                return np.asarray(self.ds[key])
            mapping, dense = self._mapping, self._index
            if dense is None:
                return self._read_chunks(key, counts, mapping, None)
        # With a dense index nothing shared changes during a read, so it runs
        # outside the lock: the dask blocks of one tile load in parallel, and
        # a background preview-cache build does not stall the display.
        return self._read_chunks(key, counts, mapping, dense)

    def _read_chunks(self, key, counts, mapping, dense):
        """Copy *key* chunk by chunk; *dense* is the chunk index, if built."""
        chunks = self.ds.chunks
        chunk_bytes = int(np.prod(chunks)) * self.dtype.itemsize
        # Enumerate only chunks containing at least one requested sample.
        axes = [np.unique(np.arange(s.start, s.stop, s.step) // c)
                for s, c in zip(key, chunks)]
        out = np.full(counts, self.ds.fillvalue, dtype=self.dtype)
        for index in itertools.product(*axes):
            origin = tuple(int(i*c) for i, c in zip(index, chunks))
            if dense is not None:
                offset = int(dense[index])
                if offset < 0:
                    continue
            elif origin not in self._offsets:
                info = self.ds.id.get_chunk_info_by_coord(origin)
                offset = info.byte_offset
                if offset is not None and (info.filter_mask != 0 or info.size != chunk_bytes
                                          or offset + chunk_bytes > len(mapping)):
                    # Never interpret an unexpected storage layout as pixels.
                    self._direct = False
                    return np.asarray(self.ds[key])
                self._offsets[origin] = offset
            else:
                offset = self._offsets[origin]
            if offset is None:
                continue  # unwritten chunk retains the dataset fill value
            dest, source = [], []
            for s, n, start, width in zip(key, counts, origin, chunks):
                a = max(0, -(-(start-s.start) // s.step))
                b = min(n, -(-(start+width-s.start) // s.step))
                dest.append(slice(a, b))
                source.append(slice(s.start+a*s.step-start,
                                    s.start+(b-1)*s.step-start+1, s.step))
            chunk = np.ndarray(chunks, dtype=self.dtype, buffer=mapping, offset=offset)
            out[tuple(dest)] = chunk[tuple(source)]
        return out
