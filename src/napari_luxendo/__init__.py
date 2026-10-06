"""napari reader plugin for Luxendo light-sheet ``.lux.h5`` data."""

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - running from a source tree
    __version__ = "unknown"

from ._lux import LuxVolume, close_all, open_lux_volume
from ._reader import napari_get_reader, read_luxendo

__all__ = [
    "LuxVolume",
    "close_all",
    "napari_get_reader",
    "open_lux_volume",
    "read_luxendo",
]
