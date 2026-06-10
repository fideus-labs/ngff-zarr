# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for actionable errors when writing arrays fails (gh-issue-343).

A corrupt TIFF/SVS source can surface only during lazy dask evaluation while
writing the output zarr array, as a cryptic ``OSError`` (e.g.
``[Errno 22] Invalid argument``) after a long conversion. ``_write_array_direct``
wraps such failures with context identifying the likely cause.
"""

import dask.array as da
import pytest
from ngff_zarr.to_ngff_zarr import _array_write_error, _write_array_direct


def _exploding_array():
    """A dask array whose computation raises an OSError, mimicking a read
    failure from a corrupt source / slab cache during the output write."""

    def boom(_block):
        raise OSError(22, "Invalid argument")

    base = da.zeros((4, 4), chunks=(2, 2), dtype="uint8")
    return base.map_blocks(boom, dtype="uint8")


def test_array_write_error_message_and_chain():
    original = OSError(22, "Invalid argument")
    wrapped = _array_write_error("scale0/image", original)

    assert isinstance(wrapped, OSError)
    msg = str(wrapped)
    assert "Failed to write data to the zarr array" in msg
    assert "scale0/image" in msg
    assert "corrupted" in msg.lower()
    assert "Invalid argument" in msg


def test_write_array_direct_wraps_compute_oserror():
    import zarr

    # MemoryStore exists on both zarr v2 and v3; empty format_kwargs routes
    # through the version-agnostic ``dask.array.to_zarr`` branch.
    store = zarr.storage.MemoryStore()
    arr = _exploding_array()

    with pytest.raises(OSError) as exc_info:
        _write_array_direct(
            arr,
            store,
            "scale0/image",
            sharding_kwargs={},
            zarr_kwargs={},
            format_kwargs={},
            dimension_names_kwargs={},
        )

    msg = str(exc_info.value)
    assert "Failed to write data to the zarr array" in msg
    assert "scale0/image" in msg
    # The original error is preserved in the chain.
    assert isinstance(exc_info.value.__cause__, OSError)
