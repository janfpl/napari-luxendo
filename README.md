# napari-luxendo

A [napari](https://napari.org) reader plugin for Luxendo / Bruker light-sheet
[Luxendo Image](https://github.com/Luxendo/luxendo-image) (`.lux.h5`) data:
single volumes, tiled acquisitions, multiview and time series, plus an
[export panel](#export) that writes layers back to `.lux.h5`, BigTIFF,
OME-TIFF or JPEG.

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

### On a new PC with conda

These steps work the same on Windows, macOS and Linux. On Windows, run them in
the **Miniforge Prompt** (or **Anaconda Prompt**) from the Start menu; on
macOS and Linux, in a normal terminal.

1. **Install conda**, if the PC has none yet.
   [Miniforge](https://conda-forge.org/download/) is recommended (it uses
   the conda-forge channel by default). Miniconda or Anaconda work too.
   Accept the defaults during installation.

2. **Create an environment** with Python, napari, a Qt backend and git
   (git is needed to install the plugin straight from GitHub):

   ```bash
   conda create -n napari-luxendo -c conda-forge python=3.11 napari pyqt git
   ```

3. **Activate it and install the plugin:**

   ```bash
   conda activate napari-luxendo
   pip install git+https://github.com/janfpl/napari-luxendo.git
   ```

   Optionally add `numba` for faster pyramid generation when exporting to
   `.lux.h5`:

   ```bash
   conda install -c conda-forge numba
   ```

4. **Start napari:**

   ```bash
   napari
   ```

   Check that the **Plugins** menu lists *Luxendo H5 Reader* with the
   *Luxendo coordinates* and *Export (.lux.h5 / TIFF / JPEG)* widgets, then
   drag a `.lux.h5` file onto the window.

Every later session only needs `conda activate napari-luxendo` and then
`napari`.

**Updating** to the latest version:

```bash
conda activate napari-luxendo
pip install --upgrade --force-reinstall --no-deps git+https://github.com/janfpl/napari-luxendo.git
```

**Removing** everything: `conda env remove -n napari-luxendo`.

If `conda activate` fails with a message about `conda init` (common in
Windows PowerShell or a fresh terminal), run `conda init` once, close the
terminal and open a new one, or use the Miniforge / Anaconda Prompt instead.

### Into an existing environment

Into an environment that already has napari (e.g. the Shifter conda env):

```bash
conda activate shifter
pip install git+https://github.com/janfpl/napari-luxendo.git
```

If git is not installed there, `conda install -c conda-forge git` first, or
install from the source archive instead:
`pip install https://github.com/janfpl/napari-luxendo/archive/refs/heads/main.zip`.

Without conda, `pip install "napari-luxendo[napari] @ git+https://github.com/janfpl/napari-luxendo.git"`
installs the plugin together with napari.

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

### Export

**Plugins > Luxendo H5 Reader > Export (.lux.h5 / TIFF / JPEG)** writes
Luxendo layers to disk, one output per layer and timepoint, named
`<layer name>_tp-<t>` plus the format's extension:

| Format | Output | What it keeps |
|--------|--------|---------------|
| Luxendo H5 | `.lux.h5` | Pixels, pyramid levels and the Luxendo metadata, so it reopens in place (see below) |
| BigTIFF | `.tif`, one page per Z-plane | Pixels only |
| OME-TIFF | `.ome.tif`, one page per Z-plane (BigTIFF) | Pixels, plus OME-XML with the layer name, voxel size in µm and channel colour. Opens with physical units in Fiji/Bio-Formats, QuPath and other OME readers. The sample-space placement is not stored. |
| JPEG | A folder holding `z0000.jpg`, `z0001.jpg`, … | One 8-bit grayscale image per Z-plane, for slides and quick sharing. Lossy (quality 95). |

JPEG scales each layer to 8 bits with the contrast limits it has in napari
(the lower limit becomes 0, the upper 255, values outside are clipped), so the
images look like the viewer. Set the contrast before exporting. To get a single
image, export an ROI one slice deep. JPEG can't hold images wider or higher
than 65,500 pixels; export an ROI of a larger mosaic.

Options:

- **Layers**: tick the layers to export (all by default). A mosaic is written
  as its fused volume, exactly as shown, with seams midway through the overlaps.
- **Timepoints**: a range of timepoints, or *Current* for the one on the time
  slider.
- **Output format**: one of the formats above.
- **Export region**: the full volume, or a box in world coordinates (µm, as
  shown in the viewer). *Draw ROI rectangle* sets Y/X from a rectangle, the
  *Z min/max = current slice* buttons set Z, and all six bounds can be typed.
  Each layer exports the voxels of its own grid whose centres lie in the box
  (for a rotated view, the voxel bounding box of the region).
- **Pyramid layers** (`.lux.h5` only): regenerate the layer's `Data_W_H_D`
  levels while writing, or untick for the fastest export. Install
  `numba` for a multi-threaded reduction.
- **Imaris header** (`.lux.h5` only): `luxendo_export.ims` links every exported
  layer (as a channel) and timepoint. It needs all layers to export at the same
  size.
- **Metadata summary**: `luxendo_export.json` (layers, source files, ROI,
  outputs, pyramid levels, export date) is always written. *JSON + TXT* or
  *JSON + CSV* adds `luxendo_export.txt` / `.csv`, the same content with one
  entry per line: dotted keys such as `layers.0.outputs.1.file`, written as
  `key: value` (TXT) or as `key,value` rows under a header (CSV). From Python,
  `napari_luxendo._export.convert_json_metadata(path, "csv")` converts any
  JSON file the same way.

An exported `.lux.h5` holds `Data`, the pyramid levels and the source's JSON
metadata with `image_size_vx`, `time_point` and `affine_to_sample` rewritten
for the exported grid, so it opens in the same place in sample space. The
RAM slider sets the share of available memory a Z-slab may use. The output
directory must not hold the source files.

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
- **Performance without pyramids.** Large raw datasets get lazy display
  overviews automatically. Opening needs no preprocessing, and nothing is
  written next to the data; level zero keeps the original pixels. At first
  the overviews are nearest-neighbour samples, which read as much from disk
  per plane as full resolution. The first time one is shown, its levels are
  averaged once in a background thread into a cache file in your user cache
  folder (about 1/7 of the data size), and the display switches to it plane
  by plane as it fills. Opening the same data again uses the finished cache
  straight away. `NAPARI_LUXENDO_CACHE_DIR` moves the cache,
  `NAPARI_LUXENDO_PYRAMID_CACHE=0` turns it off and
  `NAPARI_LUXENDO_PREVIEW=0` disables generated display levels. Delete the
  cache folder to reclaim the space. The initial view is a 2D slice; opening
  a 3D overview can still require substantial disk access.
- **Scrolling through Z.** While a slider moves, napari shows a level about
  two steps coarser and loads full detail once the slider has been still for
  0.2 s. This only happens when the coarser level is cheap to read: a native
  pyramid level or a finished overview cache. `NAPARI_LUXENDO_SCROLL_PREVIEW=0`
  turns it off. napari's *Render Images Asynchronously* setting (Preferences >
  Experimental) additionally keeps the slider responsive while planes load.
- **Zooming and panning.** While you zoom, napari shows the same coarser level
  first and loads full detail 0.2 s after the zoom stops. A zoom into an area
  whose full detail is already in memory shows full detail straight away.
  When the view stops, the area one screen around the view is read into memory in the background at the displayed level and the
  two coarser ones, so panning into it does not touch the disk. These blocks
  live in an in-memory cache of up to 1 GB, or a tenth of RAM if that is
  less, and the least recently shown blocks are dropped first. Nothing is
  written to disk. `NAPARI_LUXENDO_TILE_CACHE_MB` sets the memory budget
  (`0` turns the cache off), and `NAPARI_LUXENDO_PREFETCH` sets the distance
  in screens (`0` turns prefetching off).
- **Reading raw chunks.** Display requests read individual planes and crops.
  Uncompressed standard numeric HDF5 chunks use a read-only memory mapping,
  with every physical address obtained from the actual HDF5 chunk index.
  Compressed and unsupported layouts use h5py, with a 64 MB chunk cache per
  file (`NAPARI_LUXENDO_H5_CACHE_MB`) so consecutive planes do not decompress
  the same chunks again. No offsets are guessed from file order.
  `NAPARI_LUXENDO_DIRECT_IO=0` forces h5py for troubleshooting.
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
# the export widget tests also need a Qt binding and pytest-qt:
pip install pyqt5 pytest-qt
```

`benchmarks/synthetic_benchmark.py` generates a raw mosaic without pyramids
and times opening, the first slice, a Z step and full-resolution reads;
`--source` points it at another checkout's `src` to compare.

## License

BSD-3-Clause
