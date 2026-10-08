# napari-luxendo

A [napari](https://napari.org) reader plugin for Luxendo / Bruker light-sheet
[Luxendo Image](https://github.com/Luxendo/luxendo-image) (`.lux.h5`) data:
single volumes, tiled acquisitions, multiview and time series. It reads data
only: there are no widgets, no processing and no writer.

- **Lazy loading.** Volumes are opened as dask arrays, so napari reads only
  what you view, even for multi-hundred-GB experiments.
- **Whole experiments from one file.** Opening a main file
  (`main_raw.lux.h5`, `main_processed.lux.h5`) loads every tile, camera, view,
  channel and timepoint it links to.
- **Sample-space placement.** Each view is positioned with its
  `affine_to_sample` transform (scaling, camera flips, stage translation and
  rotation), so tiles and views land where they were in the sample.
- **Tile mosaics.** Tiles of the same channel are stitched into a single lazy
  mosaic layer. Each pixel comes from the tile whose centre is closest, so
  seams sit midway through the overlaps. Channels are separate layers added
  together.
- **Time series.** Timepoints are stacked on a T axis (napari's time slider).
- **Resolution pyramids.** `Data_W_H_D` datasets become napari multiscale
  levels. For a mosaic, the levels are stitched the same way.
- **Companion headers.** Imaris `.ims` and BigDataViewer `*.h5` headers work
  too. They resolve to the same `.lux.h5` files and are placed the same way.
- **Channel colours and contrast.** Every tile of a channel shares one
  colormap and one set of contrast limits.
- **Metadata.** The Luxendo JSON metadata is attached to `layer.metadata`.

## Installation

```bash
pip install git+https://github.com/janfpl/napari-luxendo.git
```

Into an existing napari environment (e.g. the Shifter conda env):

```bash
conda activate shifter
pip install git+https://github.com/janfpl/napari-luxendo.git
```

## Usage

Drag a file onto the napari window, use **File > Open File(s)…**, or from Python:

```python
import napari

viewer = napari.Viewer()
viewer.open("2025-08-15_150000/main_raw.lux.h5", plugin="napari-luxendo")  # whole experiment
viewer.open("Cam_long_00000.lux.h5", plugin="napari-luxendo")              # one view
napari.run()
```

Without napari, the reader returns plain layer data:

```python
from napari_luxendo import open_lux_volume

vol = open_lux_volume("uni_tp-0_ch-0.lux.h5")
vol.data            # full-resolution dask array, (Z, Y, X)
vol.levels          # [full-res, Data_2_2_2, ...]
vol.voxel_size_um   # (z, y, x) or None
```

### Supported files

| File | What is loaded |
|------|----------------|
| `main_raw.lux.h5`, `main_processed.lux.h5`, other nested `.lux.h5` | Everything under `timepoint_*/channel_*/<view>/`, following the external links into `raw/` and `processed/` |
| `*.lux.h5` (flat) | One view: `Data` plus any `Data_W_H_D` levels. Passing several tile / timepoint files in one call (*File > Open Files as Stack…*, or a list from Python) gives a mosaic and time series. Plain *Open Files* opens each file separately. |
| `*.ims` (Luxendo header) | One entry per channel. Each external link is followed to the exact dataset it names, including views inside nested files. Native Imaris files that store their own pixels are not claimed. |
| BigDataViewer `*.h5` (+ `*.xml`) | One entry per setup. Setups with the same `channel` attribute become one mosaic. |

A plain `.h5` file (not named `.lux.h5`) is only claimed if its `metadata`
holds a Luxendo `processingInformation` block. Any other `.h5` file is left to
other readers, even if it contains a dataset called `Data`.
Incomplete `.lux.h5.part` files are never opened.

### How data is arranged into layers

1. **Views.** Every source is reduced to views, one per tile, camera,
   objective, channel and timepoint.
2. **Series.** The same view across timepoints forms one series, stacked on a
   T axis. A timepoint a view doesn't have is shown empty.
3. **Mosaics.** Series of one channel that share orientation and voxel size
   (the same linear part of `affine_to_sample`) are stitched into one mosaic
   layer. Tile offsets are rounded to the nearest voxel of the shared grid, so
   a tile can be up to half a voxel from its exact position. Views at a
   different rotation angle, camera flip or voxel size get their own layer.
4. **Placement.** Each layer gets an `affine`, so napari's world coordinates
   are sample-space micrometres.

### Options

Set these as environment variables before starting napari, or pass the
matching keyword to `napari_luxendo.read_luxendo(...)`:

| Variable | Keyword | Values |
|----------|---------|--------|
| `NAPARI_LUXENDO_TRANSFORM` | `transform` | `sample` (default): place with `affine_to_sample`. `voxel`: scale by the voxel size only, as for a viewer that can't apply the affine. |
| `NAPARI_LUXENDO_TILES` | `tiles` | `mosaic` (default) or `separate`: one layer per tile. With separate tiles, overlaps blend additively. |
| `NAPARI_LUXENDO_VIEWS` | `views` | `ask` (default): when a main file has both `raw_*` and `proc_*` views, a dialog asks which to load; outside napari the default is processed. `raw` or `proc` skips the question. |

### Camera coordinates toggle

**Plugins > Luxendo H5 Reader > Luxendo coordinates** opens a dock widget with
a *Show raw data in camera coordinates* switch. When it's on, every Luxendo
layer is placed by voxel size only: the raw grid as the camera recorded it, with
no rotation, flip or stage offset. 2D slices are then the planes the camera
took, without the "non-orthogonal slicing" warning, but views no longer line up
with each other. Switch it off to return to sample space. Layers opened while it
is on follow it. From Python: `napari_luxendo._coordinates.set_coordinates(layer, "camera")`.

### Things to know

- **Rotated views (multiview / MuVi angles) look right in 3D only.** napari
  can't slice a rotated volume obliquely in 2D. It warns and shows the
  volume's own planes without the rotation. Tile grids with only flips and
  translations are exact in 2D. To browse raw planes in 2D, use the
  [camera coordinates toggle](#camera-coordinates-toggle).
- **Changing transforms over time.** If a view's `affine_to_sample` changes
  between timepoints (e.g. drift-corrected data), every timepoint is placed
  with the first one and a warning is shown.
- **Missing files** (moved, or not yet copied) are skipped with a warning.
  Their region stays empty.
- **Performance without pyramids.** Raw tiles are often stored without
  `Data_2_2_2` etc., and are chunked 64×64×64. Showing one plane then reads
  64 planes of every tile it covers, and the whole-mosaic 3D view reads
  everything. Generating pyramids, e.g. in the Luxendo Image Processor, makes
  large mosaics much faster to browse.
- **Contrast.** Initial contrast limits are the 0.05–99.95 percentiles of a
  small central block from up to six views of each channel, so loading never
  scans whole volumes.
- **File handles.** h5py files stay open as long as their layers can read
  from them. To release them, remove the layers and call
  `napari_luxendo.close_all()`.
- **Moving data.** Main files and headers link to the tile files by relative
  path. Copy or move the experiment folder as a whole. Links written as
  absolute Windows paths are resolved by filename next to the linking file.

## Development

```bash
pip install -e ".[testing]" napari
pytest
```

## License

BSD-3-Clause
