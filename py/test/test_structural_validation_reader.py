# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Reader-integration tests for strict structural validation.

These exercise the image read path
(:meth:`~ngff_zarr.v04.zarr_metadata.Metadata._from_zarr_attrs` via
:func:`~ngff_zarr.from_ngff_zarr`) rather than the rule functions in isolation:

* a valid array round-tripped through :func:`~ngff_zarr.to_ngff_zarr` and read
  back with ``validate=True`` passes both the schema and structural layers;
* an in-memory store whose ``.zattrs`` violates one structural rule (a ``scale``
  vector whose length disagrees with the axis count -- a v0.4 MUST that the JSON
  Schema cannot express) raises
  :class:`~ngff_zarr.structural_validation.ValidationError` under
  ``validate=True`` but reads silently under the default ``validate=False``.

``validate=True`` runs the JSON-Schema pass first, so these tests require the
optional ``jsonschema`` dependency.
"""

import numpy as np
import pytest
import zarr
from ngff_zarr import from_ngff_zarr, to_multiscales, to_ngff_zarr
from ngff_zarr.structural_validation import SpecRule, ValidationError

pytest.importorskip("jsonschema", reason="jsonschema required for validate=True")


def _write_valid_2d_store() -> zarr.storage.MemoryStore:
    """Write a valid 2D ``(y, x)`` two-level multiscales to an in-memory store."""
    store = zarr.storage.MemoryStore()
    array = np.random.random((32, 16)).astype("float32")
    multiscales = to_multiscales(array, [2])
    to_ngff_zarr(store, multiscales, version="0.4")
    return store


def test_valid_roundtrip_passes_strict_validation():
    store = _write_valid_2d_store()
    multiscales = from_ngff_zarr(store, validate=True)
    assert multiscales is not None


def test_scale_length_violation_raises_only_when_validating():
    store = _write_valid_2d_store()
    # Corrupt one dataset's scale to length 3 against the 2-axis (y, x) image.
    # This stays schema-valid (scale has >= 2 entries) but violates the
    # structural scale-length MUST.
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    dataset0 = attrs["multiscales"][0]["datasets"][0]
    for transform in dataset0["coordinateTransformations"]:
        if transform.get("type") == "scale":
            transform["scale"] = [1.0, 1.0, 1.0]
    root.attrs["multiscales"] = attrs["multiscales"]

    # validate=True surfaces the structural violation as a ValidationError.
    with pytest.raises(ValidationError) as exc_info:
        from_ngff_zarr(store, validate=True)
    assert exc_info.value.rule == SpecRule.SCALE_LENGTH_MISMATCH

    # The same corrupted store reads cleanly when validation is disabled.
    multiscales = from_ngff_zarr(store, validate=False)
    assert multiscales is not None
