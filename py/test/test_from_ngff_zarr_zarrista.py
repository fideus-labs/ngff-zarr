# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Reading OME-Zarr through the zarrista compat layer and the v2 store reader."""

import json
import tempfile
import zipfile
from unittest.mock import patch

import dask.array
import numpy as np
import pytest
from dask_image import imread
from ngff_zarr import (
    from_ngff_zarr,
    from_ome_zarr,
    open_array,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
    to_ome_zarr,
)

# zarrista requires Python >= 3.11; the py310 environment runs without it.
pytest.importorskip("zarrista")

pytestmark = pytest.mark.filterwarnings(
    "ignore:use_tensorstore is deprecated:DeprecationWarning"
)


def _write_ramp_store(store_path, version: str):
    """A small two-scale OME-Zarr store with deterministic values."""
    data = np.arange(16 * 16, dtype="uint16").reshape(16, 16)
    image = to_ngff_image(data, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2], chunks=8)
    to_ngff_zarr(str(store_path), multiscales, version=version)
    return data


def _forbid_zarr_python():
    """Patches that make any zarr-python read explode."""
    return (
        patch("zarr.open_group", side_effect=AssertionError("zarr-python used")),
        patch("dask.array.from_zarr", side_effect=AssertionError("zarr-python used")),
    )


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_local_path_read_bypasses_zarr_python(tmp_path, version):
    """Local directory stores read through the compat layer, not zarr-python."""
    store_path = tmp_path / f"image_{version}.zarr"
    data = _write_ramp_store(store_path, version)

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(str(store_path))

    assert len(multiscales.images) == 2
    scale0 = multiscales.images[0].data
    # The read stays lazy with the on-disk chunking.
    assert isinstance(scale0, dask.array.Array)
    assert scale0.chunksize == (8, 8)
    assert np.array_equal(np.asarray(scale0), data)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_local_path_read_validate(tmp_path, version):
    """validate=True (schema + structural validation) works on compat reads."""
    store_path = tmp_path / f"image_{version}.zarr"
    _write_ramp_store(store_path, version)

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(str(store_path), validate=True)
    assert len(multiscales.images) == 2


def test_mapping_store_read_uses_v2_reader(tmp_path):
    """A plain dict store goes through the pure-Python v2 store reader."""
    store_path = tmp_path / "image.zarr"
    data = _write_ramp_store(store_path, "0.4")
    mapping = {
        p.relative_to(store_path).as_posix(): p.read_bytes()
        for p in store_path.rglob("*")
        if p.is_file()
    }

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(mapping)

    assert len(multiscales.images) == 2
    scale0 = multiscales.images[0].data
    assert isinstance(scale0, dask.array.Array)
    assert scale0.chunksize == (8, 8)
    assert np.array_equal(np.asarray(scale0), data)


def test_local_bioformats2raw_navigation(tmp_path):
    """bioformats2raw container navigation works on the compat-layer nodes."""
    container = tmp_path / "container.zarr"
    data = _write_ramp_store(container / "0", "0.4")

    # Hand-write the container-level metadata around the image at "0".
    (container / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (container / ".zattrs").write_text(json.dumps({"bioformats2raw.layout": 3}))
    ome_dir = container / "OME"
    ome_dir.mkdir()
    (ome_dir / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (ome_dir / ".zattrs").write_text(json.dumps({"series": ["0"]}))

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(str(container))
    assert np.array_equal(np.asarray(multiscales.images[0].data), data)

    # Without the OME series metadata the container falls back to enumerating
    # child groups, exercising keys()/__getitem__ on the compat nodes.
    (ome_dir / ".zattrs").write_text(json.dumps({}))
    with open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(str(container))
    assert np.array_equal(np.asarray(multiscales.images[0].data), data)


def test_local_subpath_navigation(tmp_path):
    """A path into a nested group ('store.zarr/sub') resolves on compat reads."""
    store_path = tmp_path / "nested.zarr"
    data = _write_ramp_store(store_path / "A" / "1" / "0", "0.4")
    # Parent groups so the root opens as a valid group.
    for rel in ("", "A", "A/1"):
        node = store_path / rel
        (node / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(str(store_path / "A" / "1" / "0"))
    assert np.array_equal(np.asarray(multiscales.images[0].data), data)


def test_local_read_missing_chunks_use_fill_value(tmp_path):
    """Chunks absent on disk materialize as fill_value through zarrista."""
    store_path = tmp_path / "holes.zarr"
    _write_ramp_store(store_path, "0.4")
    # Remove one stored chunk of the top scale; v2 fills missing chunks with 0.
    chunk_files = [
        p
        for p in (store_path / "scale0" / "image").rglob("*")
        if p.is_file() and not p.name.startswith(".")
    ]
    assert chunk_files
    chunk_files[0].unlink()

    multiscales = from_ngff_zarr(str(store_path))
    values = np.asarray(multiscales.images[0].data)
    assert (values == 0).any()


def test_from_ngff_zarr(input_images):
    dataset_name = "lung_series"
    data = imread.imread(input_images[dataset_name])
    image = to_ngff_image(
        data=data,
        dims=("z", "y", "x"),
        scale={"z": 2.5, "y": 1.40625, "x": 1.40625},
        translation={"z": 332.5, "y": 360.0, "x": 0.0},
        name="LIDC2",
    )
    multiscales = to_multiscales(image)
    multiscales.scale_factors = None
    multiscales.method = None
    multiscales.chunks = None
    with tempfile.TemporaryDirectory() as tmpdir:
        version = "0.4"
        to_ngff_zarr(tmpdir, multiscales, use_tensorstore=True, version=version)
        multiscales = from_ngff_zarr(tmpdir, version=version, validate=True)


def test_v3_group_with_empty_attributes_is_not_discarded(tmp_path):
    """A format 3 group with no attributes must survive the format 2 retry.

    Auto-detection retries at format 2 when the format 3 node has empty
    attributes (the spurious-``zarr.json`` case). When no format 2 node
    exists that retry returns ``None``, which must not erase the valid
    format 3 group and turn into a "group not found" error.
    """
    from ngff_zarr.from_ngff_zarr import _open_local_root

    store = tmp_path / "empty_attrs.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )

    node = _open_local_root(store, None)
    assert node is not None
    assert node.attrs == {}


def _write_gzip_default_level_store(store_path):
    """A format 2 store whose arrays record gzip ``level: -1``, zlib's default
    level, as numcodecs' ``GZip(level=-1)`` stores it."""
    data = np.arange(16 * 16, dtype="uint16").reshape(16, 16)
    image = to_ngff_image(data, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2], chunks=8)
    to_ome_zarr(str(store_path), multiscales, version="0.4", compressor={"id": "gzip"})
    for doc in store_path.rglob(".zarray"):
        metadata = json.loads(doc.read_text())
        metadata["compressor"]["level"] = -1
        doc.write_text(json.dumps(metadata))
    return data


@pytest.mark.parametrize("zipped", [False, True])
def test_read_gzip_default_level(tmp_path, zipped):
    """zarrs rejects gzip ``level: -1``; the read respells it (issue #833)."""
    store_path = tmp_path / "image.zarr"
    data = _write_gzip_default_level_store(store_path)
    target = store_path
    if zipped:
        target = tmp_path / "image.zarr.zip"
        with zipfile.ZipFile(target, "w") as archive:
            for path in sorted(store_path.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(store_path).as_posix())

    multiscales = from_ome_zarr(target)

    assert len(multiscales.images) == 2
    assert np.array_equal(np.asarray(multiscales.images[0].data), data)


def test_open_array_gzip_default_level_writes(tmp_path):
    """A region write into a gzip ``level: -1`` array reads back."""
    store_path = tmp_path / "image.zarr"
    _write_gzip_default_level_store(store_path)
    path = from_ome_zarr(store_path).metadata.datasets[0].path

    array = open_array(store_path, path)
    array[:8, :8] = 7

    assert np.all(open_array(store_path, path)[:8, :8] == 7)


@pytest.mark.parametrize(
    ("zarray", "expected"),
    [
        (b'{"compressor": {"id": "gzip", "level": -1}}', {"id": "gzip", "level": 6}),
        (b'{"compressor": {"id": "gzip", "level": 5}}', None),
        (b'{"compressor": {"id": "zlib", "level": -1}}', None),
        (b'{"compressor": null}', None),
        (b"[]", None),
        (b"not json", None),
        (None, None),
    ],
)
def test_zarrs_compatible_zarray(zarray, expected):
    """Only gzip ``level: -1`` is respelled; anything else is left alone."""
    from ngff_zarr._zarrista_utils import _zarrs_compatible_zarray

    doc = _zarrs_compatible_zarray(zarray)

    assert (doc if doc is None else doc["compressor"]) == expected


def test_read_other_rejected_zarray_keeps_zarrista_error(tmp_path):
    """Metadata zarrs rejects for another reason raises zarrista's error."""
    from zarrista.exceptions import ArrayCreateError

    store_path = tmp_path / "image.zarr"
    _write_gzip_default_level_store(store_path)
    for doc in store_path.rglob(".zarray"):
        metadata = json.loads(doc.read_text())
        metadata["compressor"] = {"id": "zlib", "level": -1}
        doc.write_text(json.dumps(metadata))

    with pytest.raises(ArrayCreateError, match="zlib"):
        np.asarray(from_ome_zarr(store_path).images[0].data)
