# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""``validate_ome_zarr`` reports a store the spec refuses as invalid.

The tool runs two passes over a store: the JSON Schema of the store's own
version, and the structural rules that carry the spec MUSTs no schema states.
A failure in either is an error, so it reaches the caller through ``valid``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ngff_zarr import to_multiscales, to_ngff_image, to_ome_zarr

from ngff_zarr_mcp.tools import validate_ome_zarr

VERSIONS = ["0.4", "0.5", "0.6"]


def _write_store(path: Path, version: str) -> Path:
    image = to_ngff_image(
        np.zeros((32, 32), dtype=np.uint8), dims=["y", "x"], scale={"y": 1.0, "x": 1.0}
    )
    multiscales = to_multiscales(image, scale_factors=[2])
    to_ome_zarr(str(path), multiscales, version=version)
    return path


def _root_document(store: Path) -> tuple[Path, dict]:
    """The root attributes document and its path, whichever layout is on disk."""
    v3 = store / "zarr.json"
    if v3.exists():
        return v3, json.loads(v3.read_text())
    v2 = store / ".zattrs"
    return v2, json.loads(v2.read_text())


def _edit_multiscales(store: Path, mutate) -> None:
    path, document = _root_document(store)
    if "attributes" in document:
        multiscales = document["attributes"]["ome"]["multiscales"]
    else:
        multiscales = document["multiscales"]
    mutate(multiscales[0])
    path.write_text(json.dumps(document))
    consolidated = store / "zarr.json"
    if consolidated.exists() and path == consolidated:
        return


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_a_valid_store_is_valid_at_every_version(tmp_path, version):
    """Zarr v3 stores, which is every 0.5 and 0.6 store, get past the store check."""
    store = _write_store(tmp_path / f"valid-{version}.zarr", version)

    result = await validate_ome_zarr(str(store))

    assert result.valid, result.errors


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_datasets_ordered_coarsest_to_finest_are_refused(tmp_path, version):
    """`dataset-order-highest-to-lowest`, a rule no JSON Schema states."""
    store = _write_store(tmp_path / f"reversed-{version}.zarr", version)
    _edit_multiscales(store, lambda ms: ms["datasets"].reverse())

    result = await validate_ome_zarr(str(store))

    assert not result.valid
    assert any("dataset" in error.lower() for error in result.errors), result.errors


@pytest.mark.asyncio
async def test_a_document_the_schema_rejects_is_an_error_not_a_warning(tmp_path):
    """A schema failure used to be demoted to a warning, leaving `valid` True."""
    pytest.importorskip("jsonschema")
    store = _write_store(tmp_path / "no-axes.zarr", "0.4")
    _edit_multiscales(store, lambda ms: ms.pop("axes"))

    result = await validate_ome_zarr(str(store))

    assert not result.valid
    assert result.errors
