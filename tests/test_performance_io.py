"""Pixel fidelity and bounded reads for the interactive display path."""
from concurrent.futures import ThreadPoolExecutor
import itertools

import h5py
import numpy as np
import pytest

from napari_luxendo import close_all, open_lux_volume, read_luxendo
from napari_luxendo._fastio import ChunkReader, reader_for
from conftest import make_volume, write_lux


@pytest.mark.parametrize('dtype', ['<u2', '>u2', '<f4'])
def test_exact_offsets_random_allocation_sparse_fill_and_strides(tmp_path, dtype):
    path = tmp_path / 'random.lux.h5'
    shape, chunks = (13, 19, 23), (4, 5, 6)
    rng = np.random.default_rng(5)
    truth = rng.integers(0, 30000, size=shape).astype(dtype)
    order = list(itertools.product(range(4), repeat=3))
    rng.shuffle(order)
    with h5py.File(path, 'w', userblock_size=512) as f:
        ds = f.create_dataset('Data', shape=shape, dtype=dtype, chunks=chunks, fillvalue=123)
        # Some chunks remain unallocated; logical order differs from disk order.
        for index in order[::2]:
            key = tuple(slice(i*c, min((i+1)*c,n)) for i,c,n in zip(index,chunks,shape))
            ds[key] = truth[key]
    with h5py.File(path, 'r') as f:
        ds = f['Data']
        reader = ChunkReader(ds)
        assert reader._direct
        assert reader._index is None and reader._mapping is None
        for key in [(slice(None),)*3, (slice(2,12,3),slice(1,18,7),slice(3,23,4)),
                    (slice(12,13),slice(18,19),slice(22,23)),
                    (slice(0,0),slice(None),slice(None))]:
            np.testing.assert_array_equal(reader[key], ds[key])
        reader.close()
        assert reader._mapping is None and reader._index is None


@pytest.mark.parametrize('options', [{'compression':'gzip'}, {'fletcher32':True}, {'chunks':None}])
def test_filtered_and_contiguous_storage_falls_back(tmp_path, options):
    truth = make_volume(8)
    with h5py.File(tmp_path/'fallback.h5','w') as f:
        ds = f.create_dataset('Data', data=truth, **options)
        reader = ChunkReader(ds)
        assert not reader._direct
        np.testing.assert_array_equal(reader[(slice(None,None,3),)*3], truth[::3,::3,::3])


def test_explicit_h5py_fallback_and_lifecycle(tmp_path, monkeypatch):
    path = write_lux(tmp_path/'raw.lux.h5', make_volume(1), pyramids=())
    monkeypatch.setenv('NAPARI_LUXENDO_DIRECT_IO','0')
    vol = open_lux_volume(path)
    reader = reader_for(vol.datasets[0])
    assert not reader._direct
    np.testing.assert_array_equal(vol.data[4].compute(), make_volume(1)[4])
    close_all()
    monkeypatch.delenv('NAPARI_LUXENDO_DIRECT_IO')
    vol = open_lux_volume(path)
    reader = reader_for(vol.datasets[0])
    assert reader._direct
    vol.read(0,(slice(2,3),slice(None),slice(None)))
    assert reader._mapping is not None
    close_all()
    assert reader._mapping is None
    with pytest.raises(ValueError):
        reader[(slice(0,1),)*3]


def test_direct_concurrent_reads_match_h5py(tmp_path):
    vol = open_lux_volume(write_lux(tmp_path/'concurrent.lux.h5',make_volume(2),pyramids=()))
    keys = [(slice(z,z+1),slice(3,39,2),slice(2,47,3)) for z in range(24)]
    expected = [vol.datasets[0][key] for key in keys]
    with ThreadPoolExecutor(4) as pool:
        actual = list(pool.map(lambda key:vol.read(0,key),keys))
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)


def test_mosaic_crop_only_reads_requested_plane(tiled_experiment, monkeypatch):
    from napari_luxendo._lux import LuxVolume
    layers = read_luxendo(str(tiled_experiment/'main_raw.lux.h5'))
    calls = []
    original = LuxVolume.read
    def counted(self, level, region):
        calls.append(region)
        return original(self,level,region)
    monkeypatch.setattr(LuxVolume,'read',counted)
    pixels = layers[0][0][0][0,5,10:20,10:20].compute()
    assert pixels.shape == (10,10)
    assert calls
    assert all(len(range(*key[0].indices(12))) == 1 for key in calls)


def test_preview_is_sample_of_exact_mosaic_and_keeps_full_resolution(tiled_experiment, monkeypatch):
    import napari_luxendo._preview as preview
    # Remove native pyramids from synthetic files so previews are used.
    for path in (tiled_experiment/'raw').rglob('*.h5'):
        with h5py.File(path,'a') as f:
            del f['Data_2_2_2']
    monkeypatch.setattr(preview,'preview_factors',lambda shape:[2,4])
    layers = read_luxendo(str(tiled_experiment/'main_raw.lux.h5'))
    for levels,kw,_ in layers:
        full = levels[0].compute()
        for level,factor in zip(levels[1:],(2,4)):
            np.testing.assert_array_equal(level.compute(),full[:,::factor,::factor,::factor])
            np.testing.assert_array_equal(level[1,1,1:6,2:7].compute(),
                                          full[1,::factor,::factor,::factor][1,1:6,2:7])
        assert kw['multiscale']
        assert kw['metadata']['pyramid_levels'][0] == 'Data'
        assert 'nearest' in kw['metadata']['display_pyramid']


def test_single_volume_preview_and_disable(tmp_path,monkeypatch):
    import napari_luxendo._preview as preview
    truth=make_volume(3)
    path=write_lux(tmp_path/'single.lux.h5',truth,pyramids=())
    monkeypatch.setattr(preview,'preview_factors',lambda shape:[2,4])
    [(levels,kw,_)]=read_luxendo(str(path))
    np.testing.assert_array_equal(levels[-1].compute(),truth[::4,::4,::4])
    assert kw['multiscale']


def test_preview_policy_is_bounded_and_optional(monkeypatch):
    from napari_luxendo._preview import preview_factors
    shape=(2765,8601,8602)
    factors=preview_factors(shape)
    assert factors == [2,4,8,16,32]
    assert preview_factors((24,40,48)) == []
    monkeypatch.setenv('NAPARI_LUXENDO_PREVIEW','0')
    assert preview_factors(shape) == []


def test_huge_sparse_index_uses_cached_exact_chunk_queries(tmp_path):
    with h5py.File(tmp_path/'sparse.h5','w') as f:
        ds=f.create_dataset('Data',shape=(1001,1001,2),chunks=(1,1,1),dtype='u2',fillvalue=321)
        ds[5,10,1]=987
        ds[4,9,0]=654
    with h5py.File(tmp_path/'sparse.h5','r') as f:
        ds=f['Data']
        reader=ChunkReader(ds)
        key=(slice(4,7),slice(9,12),slice(0,2))
        for _ in range(2):
            np.testing.assert_array_equal(reader[key],ds[key])
        assert reader._index is None and len(reader._offsets)==18
        reader.close()


def test_virtual_storage_falls_back(tmp_path):
    source=tmp_path/'source.h5'
    truth=make_volume(4)
    with h5py.File(source,'w') as f:
        f['Data']=truth
    layout=h5py.VirtualLayout(shape=truth.shape,dtype=truth.dtype)
    layout[:]=h5py.VirtualSource(str(source),'Data',shape=truth.shape)
    with h5py.File(tmp_path/'virtual.h5','w') as f:
        ds=f.create_virtual_dataset('Data',layout)
        reader=ChunkReader(ds)
        assert not reader._direct
        np.testing.assert_array_equal(reader[(slice(None),)*3],truth)


def test_mosaic_reverse_integer_and_newaxis_slicing(tiled_experiment):
    array=read_luxendo(str(tiled_experiment/'main_raw.lux.h5'))[0][0][0]
    truth=array.compute()
    for key in [(0,slice(None,None,-1),slice(10,20),slice(3,9)),
                (None,0,4,slice(None,None,-2),slice(None,None,3)),
                (1,-1,-1,-1)]:
        np.testing.assert_array_equal(array[key].compute(),truth[key])


def test_preview_metadata_when_native_levels_differ(lux_dir, monkeypatch):
    import napari_luxendo._preview as preview
    monkeypatch.setattr(preview,'preview_factors',lambda shape:[2,4])
    with h5py.File(lux_dir/'uni_tp-1_ch-0.lux.h5','a') as f:
        del f['Data_2_2_2']
        del f['Data_4_4_4']
    with pytest.warns(UserWarning,match='pyramid levels differ'):
        layers=read_luxendo(str(lux_dir/'dataset.ims'))
    levels,kw,_=layers[0]
    assert kw['metadata']['pyramid_levels']==['Data','preview_nearest_1','preview_nearest_2']
    np.testing.assert_array_equal(levels[-1][1].compute(),make_volume(10)[::4,::4,::4])
