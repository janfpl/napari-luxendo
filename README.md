# napari-luxendo

A [napari](https://napari.org) reader plugin for Luxendo / Bruker light-sheet
`.lux.h5` data. It reads data only: there are no widgets, no processing and no
writer.

- **Lazy loading.** Volumes are opened as dask arrays, so napari reads only the
  planes you view, even for multi-hundred-GB files.
- **Resolution pyramids.** `Data_W_H_D` datasets are shown as one napari
  multiscale layer.
- **Physical scale.** The voxel size (µm) in the embedded metadata sets the
  layer scale, so Z and XY have the right aspect ratio.
- **Companion headers.** Opening an Imaris `.ims` or BigDataViewer `*_bdv.h5`
  header loads every channel it links to, with timepoints stacked on a T axis.
- **Channel naming and colors.** Layers are named from the channel description
  or the header channel name. Colors come from the `.ims` channel color, else a
  wavelength or color word in the name (e.g. `Red-561`), else a per-channel
  default. Multi-channel loads use additive blending.
- **Metadata.** The full Luxendo JSON metadata is attached to `layer.metadata`.

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
viewer.open("uni_tp-0_ch-0.lux.h5", plugin="napari-luxendo")   # one channel
viewer.open("dataset.ims", plugin="napari-luxendo")            # every channel in the header
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
| `*.lux.h5` | `Data` plus any `Data_W_H_D` pyramid levels, as one layer |
| `*.ims` (Luxendo header) | One layer per channel, following the header's external links. Native Imaris files that store their own pixels are not claimed. |
| `*_bdv.h5` (+ optional `*_bdv.xml`) | One layer per BDV setup. Names and voxel size come from the XML when present. |

Any other `.h5` file without a `Data` dataset is left to other readers.

### Notes

- **File handles.** h5py files stay open as long as their layers can read
  from them. To release them, remove the layers and call
  `napari_luxendo.close_all()`.
- **Contrast.** Initial contrast limits are the 0.05–99.95 percentiles of the
  coarsest pyramid level, or of the central Z plane when there is no pyramid,
  so the reader never scans a full-resolution volume.
- **Timepoints.** If timepoints in a header don't share one shape, only the
  first timepoint is shown and a warning is logged.
- **Header links.** A header must sit in the same folder as the channel files
  it links to. Links written as absolute paths from the acquisition PC are
  resolved by filename next to the header.

## Development

```bash
pip install -e ".[testing]" napari
pytest
```

## License

BSD-3-Clause
