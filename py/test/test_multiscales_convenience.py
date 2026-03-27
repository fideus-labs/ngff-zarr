# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from unittest.mock import MagicMock, patch

import ngff_zarr.from_ngff_zarr as from_ngff_zarr_mod
import ngff_zarr.to_ngff_zarr as to_ngff_zarr_mod
from ngff_zarr import Multiscales


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
