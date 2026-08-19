# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The remaining local readers dispatch through zarrista / the store reader.

Covers the Phase 04 "remaining local readers" migration: the CLI input
readers, the HCS plate reader, ``to_ngff_image``/``to_multiscales`` array
handles, and ``upgrade_ome_zarr`` source reads. zarr-python entry points are
patched to explode so any fallback to the legacy engine fails loudly.
"""

from contextlib import contextmanager
from unittest.mock import patch

import dask.array
import numpy as np
import pytest
from ngff_zarr import (
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)
from ngff_zarr.cli_input_to_ngff_image import cli_input_to_ngff_image
from ngff_zarr.detect_cli_io_backend import ConversionBackend, detect_cli_io_backend
from ngff_zarr.hcs import HCSPlate, from_hcs_zarr, to_hcs_zarr, write_hcs_well_image
from ngff_zarr.upgrade_ome_zarr import upgrade_ome_zarr
from ngff_zarr.v04.zarr_metadata import Plate, PlateColumn, PlateRow, PlateWell

# zarrista requires Python >= 3.11; the py310 environment runs without it.
pytest.importorskip("zarrista")

pytestmark = pytest.mark.filterwarnings(
    "ignore:use_tensorstore is deprecated:DeprecationWarning"
)

_DATA = np.arange(16 * 16, dtype="uint16").reshape(16, 16)


@contextmanager
def no_zarr_python_reads():
    """Patch zarr-python read entry points to explode inside this block.

    Applied around the read under test only — fixture data is authored with
    zarr-python before entering, so writes must happen outside the block.
    """
    patches = [
        patch("zarr.open_group", side_effect=AssertionError("zarr-python used")),
        patch("zarr.open_array", side_effect=AssertionError("zarr-python used")),
        patch("zarr.open", side_effect=AssertionError("zarr-python used")),
        patch("dask.array.from_zarr", side_effect=AssertionError("zarr-python used")),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def _write_v2_array(path):
    import zarr

    arr = zarr.open_array(
        str(path),
        mode="w",
        shape=_DATA.shape,
        chunks=(8, 8),
        dtype=_DATA.dtype,
        zarr_format=2,
    )
    arr[:] = _DATA


def _store_to_mapping(store_path):
    return {
        p.relative_to(store_path).as_posix(): p.read_bytes()
        for p in store_path.rglob("*")
        if p.is_file()
    }


def test_cli_zarr_array_reads_through_compat_layer(tmp_path):
    store_path = tmp_path / "plain.zarr"
    _write_v2_array(store_path)

    with no_zarr_python_reads():
        backend = detect_cli_io_backend([str(store_path)])
        assert backend == ConversionBackend.ZARR_ARRAY
        image = cli_input_to_ngff_image(backend, [str(store_path)])

        assert isinstance(image.data, dask.array.Array)
        assert image.data.chunksize == (8, 8)
        np.testing.assert_array_equal(np.asarray(image.data), _DATA)


def test_cli_tifffile_reads_through_store_reader(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tif_path = tmp_path / "image.tif"
    tifffile.imwrite(tif_path, _DATA, tile=(16, 16))

    with no_zarr_python_reads():
        image = cli_input_to_ngff_image(ConversionBackend.TIFFFILE, [str(tif_path)])

        assert isinstance(image.data, dask.array.Array)
        np.testing.assert_array_equal(np.asarray(image.data), _DATA)


def test_tiff_backend_reads_through_store_reader(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from ngff_zarr import tiff_file_to_ngff_images

    tif_path = tmp_path / "image.tif"
    tifffile.imwrite(tif_path, _DATA, tile=(16, 16))

    with no_zarr_python_reads():
        results = tiff_file_to_ngff_images(tif_path)

        assert len(results) == 1
        _, image = results[0]
        assert isinstance(image.data, dask.array.Array)
        np.testing.assert_array_equal(np.asarray(image.data), _DATA)


def test_tiff_pyramid_reads_through_store_reader(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from ngff_zarr import tiff_file_to_ngff_images
    from ngff_zarr.multiscales import NgffMultiscales

    level0 = np.arange(64 * 64, dtype="uint16").reshape(64, 64)
    level1 = level0[::2, ::2]
    tif_path = tmp_path / "pyramid.ome.tif"
    with tifffile.TiffWriter(tif_path, ome=True) as tif:
        tif.write(level0, metadata={"axes": "YX"}, tile=(32, 32), subifds=1)
        tif.write(level1, metadata={"axes": "YX"}, tile=(32, 32), subfiletype=1)

    with no_zarr_python_reads():
        results = tiff_file_to_ngff_images(tif_path, reuse_existing_pyramids=True)

        assert len(results) == 1
        _, multiscales = results[0]
        assert isinstance(multiscales, NgffMultiscales)
        assert len(multiscales.images) == 2
        for image in multiscales.images:
            assert isinstance(image.data, dask.array.Array)
        np.testing.assert_array_equal(np.asarray(multiscales.images[0].data), level0)
        np.testing.assert_array_equal(np.asarray(multiscales.images[1].data), level1)


def test_to_ngff_image_accepts_compat_layer_nodes(tmp_path):
    from ngff_zarr._zarrista_utils import open_local_node, open_zarrista_array

    store_path = tmp_path / "plain.zarr"
    _write_v2_array(store_path)

    with no_zarr_python_reads():
        local_node = open_local_node(str(store_path), zarr_format=2)
        image = to_ngff_image(local_node)
        assert image.data.chunksize == (8, 8)
        np.testing.assert_array_equal(np.asarray(image.data), _DATA)

        zarrista_array = open_zarrista_array(str(store_path))
        image = to_ngff_image(zarrista_array)
        assert image.data.chunksize == (8, 8)
        np.testing.assert_array_equal(np.asarray(image.data), _DATA)


def test_to_ngff_image_accepts_v2_reader_nodes(tmp_path):
    from ngff_zarr._v2_store_reader import open_v2

    store_path = tmp_path / "plain.zarr"
    _write_v2_array(store_path)
    mapping = _store_to_mapping(store_path)

    with no_zarr_python_reads():
        image = to_ngff_image(open_v2(mapping))
        np.testing.assert_array_equal(np.asarray(image.data), _DATA)


def test_to_multiscales_accepts_v2_reader_array(tmp_path):
    from ngff_zarr._v2_store_reader import open_v2
    from ngff_zarr.ngff_image import NgffImage

    store_path = tmp_path / "plain.zarr"
    _write_v2_array(store_path)
    mapping = _store_to_mapping(store_path)

    with no_zarr_python_reads():
        image = NgffImage(
            data=open_v2(mapping),
            dims=("y", "x"),
            scale={"y": 1.0, "x": 1.0},
            translation={"y": 0.0, "x": 0.0},
        )
        multiscales = to_multiscales(image, scale_factors=[2], chunks=8)
        np.testing.assert_array_equal(np.asarray(multiscales.images[0].data), _DATA)


def _write_plate(plate_path, version):
    plate_metadata = Plate(
        columns=[PlateColumn(name="1")],
        rows=[PlateRow(name="A")],
        wells=[PlateWell(path="A/1", rowIndex=0, columnIndex=0)],
        version=version,
    )
    plate = HCSPlate(store=str(plate_path), plate_metadata=plate_metadata)
    to_hcs_zarr(plate, str(plate_path))

    multiscales = to_multiscales(
        to_ngff_image(_DATA, dims=["y", "x"]), scale_factors=[2], chunks=8
    )
    write_hcs_well_image(
        store=str(plate_path),
        multiscales=multiscales,
        plate_metadata=plate_metadata,
        row_name="A",
        column_name="1",
        field_index=0,
        version=version,
    )


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_hcs_local_path_read_bypasses_zarr_python(tmp_path, version):
    plate_path = tmp_path / f"plate_{version}.ome.zarr"
    _write_plate(plate_path, version)

    with no_zarr_python_reads():
        plate = from_hcs_zarr(str(plate_path), validate=True)
        assert len(plate.wells) == 1
        well = plate.get_well("A", "1")
        image = well.get_image(0)

    assert image is not None
    np.testing.assert_array_equal(np.asarray(image.images[0].data), _DATA)
    # The image/well caches stay in play across repeat access.
    assert well.get_image(0) is image
    assert plate.get_well("A", "1") is well


@pytest.mark.parametrize(
    ("source_version", "target_version"),
    [("0.4", "0.6"), ("0.5", "0.6")],
)
def test_upgrade_in_place_reads_bypass_zarr_python(
    tmp_path, source_version, target_version
):
    store_path = tmp_path / "image.ome.zarr"
    multiscales = to_multiscales(
        to_ngff_image(_DATA, dims=["y", "x"]), scale_factors=[2], chunks=8
    )
    to_ngff_zarr(str(store_path), multiscales, version=source_version)

    with (
        no_zarr_python_reads(),
        patch(
            "zarr.consolidate_metadata",
            side_effect=AssertionError("zarr-python used"),
        ),
    ):
        upgrade_ome_zarr(str(store_path), version=target_version)

    upgraded = from_ngff_zarr(str(store_path))
    np.testing.assert_array_equal(np.asarray(upgraded.images[0].data), _DATA)


def test_upgrade_missing_local_store_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        upgrade_ome_zarr(str(tmp_path / "missing.ome.zarr"), version="0.6")
