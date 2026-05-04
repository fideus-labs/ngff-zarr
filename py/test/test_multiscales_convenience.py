# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import sys
from unittest.mock import MagicMock, patch

# In ngff_zarr/__init__.py, function names `to_ngff_zarr` and `from_ngff_zarr`
# shadow the submodule names.  ``import ngff_zarr.to_ngff_zarr as X`` resolves
# ``X = ngff_zarr.to_ngff_zarr`` via attribute lookup on the package, which
# returns the *function*, not the *module*.  Use sys.modules to obtain the
# actual module objects so that patch.object works on all Python versions.
import ngff_zarr.from_ngff_zarr  # force module into sys.modules
import ngff_zarr.to_ngff_zarr  # noqa: F401 - force module into sys.modules
from ngff_zarr import Multiscales

to_ngff_zarr_mod = sys.modules["ngff_zarr.to_ngff_zarr"]
from_ngff_zarr_mod = sys.modules["ngff_zarr.from_ngff_zarr"]


def test_to_ome_zarr_delegates(tmp_path):
    """Test that Multiscales.to_ome_zarr delegates to ngff_zarr.to_ome_zarr."""
    ms = MagicMock(spec=Multiscales)
    store = str(tmp_path / "out.zarr")

    with patch.object(to_ngff_zarr_mod, "to_ome_zarr") as mock_to:
        Multiscales.to_ome_zarr(ms, store, version="0.4", overwrite=False)
        mock_to.assert_called_once_with(store, ms, version="0.4", overwrite=False)


def test_from_ome_zarr_delegates(tmp_path):
    """Test that Multiscales.from_ome_zarr delegates to ngff_zarr.from_ome_zarr."""
    store = str(tmp_path / "in.zarr")
    sentinel = MagicMock(spec=Multiscales)

    with patch.object(
        from_ngff_zarr_mod, "from_ome_zarr", return_value=sentinel
    ) as mock_from:
        result = Multiscales.from_ome_zarr(store, version="0.5")
        mock_from.assert_called_once_with(store, version="0.5")
        assert result is sentinel
