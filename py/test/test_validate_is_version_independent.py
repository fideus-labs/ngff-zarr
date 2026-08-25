# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""``validate=True`` means the same thing whatever the version of the store."""

import json
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from ngff_zarr import (
    NgffImage,
    ValidationError,
    from_ome_zarr,
    to_multiscales,
    to_ome_zarr,
)

VERSIONS = ["0.4", "0.5", "0.6"]


def _store(tmp_path, version):
    image = NgffImage(
        da.zeros((1, 8, 8), dtype=np.uint8, chunks=(1, 8, 8)),
        ["c", "y", "x"],
        {"c": 1.0, "y": 1.0, "x": 1.0},
        {"c": 0.0, "y": 0.0, "x": 0.0},
    )
    store = str(tmp_path / f"v{version}.ome.zarr")
    to_ome_zarr(
        store, to_multiscales(image, scale_factors=[2], cache=False), version=version
    )
    return store


def _entry_path(store, version):
    return Path(store) / (".zattrs" if version == "0.4" else "zarr.json")


def _reverse_datasets(store, version):
    path = _entry_path(store, version)
    doc = json.loads(path.read_text())
    entry = doc if version == "0.4" else doc["attributes"]["ome"]
    entry["multiscales"][0]["datasets"].reverse()
    path.write_text(json.dumps(doc, indent=4))


@pytest.mark.parametrize("version", VERSIONS)
def test_a_valid_store_reads_with_validate(tmp_path, version):
    store = _store(tmp_path, version)
    assert len(from_ome_zarr(store, validate=True).images) == 2


@pytest.mark.parametrize("version", VERSIONS)
def test_datasets_coarsest_to_finest_are_refused_at_every_version(tmp_path, version):
    """The structural rules run at 0.6 as they do at 0.4 and 0.5.

    Before this, a 0.6 store went through the schema pass alone, so the
    meaning of ``validate=True`` followed the version of the store.
    """
    store = _store(tmp_path, version)
    _reverse_datasets(store, version)

    with pytest.raises((ValidationError, ValueError)):
        from_ome_zarr(store, validate=True)


@pytest.mark.parametrize("version", VERSIONS)
def test_the_same_store_still_reads_without_validate(tmp_path, version):
    """The default read path is unchanged: the rules run only when asked."""
    store = _store(tmp_path, version)
    _reverse_datasets(store, version)

    assert len(from_ome_zarr(store).images) == 2
