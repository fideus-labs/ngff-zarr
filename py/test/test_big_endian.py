# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Big-endian pixel data round-trips and reads.

OME-Zarr data produced by other tools (e.g. raw2ometiff, NGFF-Converter,
ITK's OME-Zarr IO) can be stored big-endian.  These tests verify that
``ngff-zarr`` reads such data with correct values and round-trips big-endian
input without corruption, for both OME-Zarr 0.4 (Zarr v2, byte order encoded
in the ``dtype`` string) and 0.5 (Zarr v3, byte order encoded by the ``bytes``
codec).

See https://github.com/fideus-labs/ngff-zarr/issues/97
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from ngff_zarr import (
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)
from packaging import version as _pkg_version

zarr_version = _pkg_version.parse(zarr.__version__)

# OME-Zarr 0.5 writes Zarr v3, which needs a zarr-python with the v3 API
# (major >= 3, i.e. >= 3.0.0b1).  Older zarr (resolved on some CI legs) raises on
# the v0.5 path, so v0.5 cases are skipped there rather than failing.  The
# 3.0.0b1 boundary matches the shared test infrastructure in ``_data.py``.
requires_zarr_v3 = pytest.mark.skipif(
    zarr_version < _pkg_version.parse("3.0.0b1"),
    reason="OME-Zarr 0.5 (Zarr v3) requires zarr-python >= 3.0.0b1",
)

# Version parametrization where the 0.5 case carries the Zarr v3 guard.
VERSION_PARAMS = ["0.4", pytest.param("0.5", marks=requires_zarr_v3)]

# Multi-byte dtypes exercised in both byte orders.  ``|`` (single byte) types
# carry no byte order, so they are not relevant here.
BIG_ENDIAN_DTYPES = [">u2", ">i2", ">i4", ">u4", ">f4", ">f8"]


def _native(dtype_str: str) -> np.dtype:
    """The native-byte-order equivalent of ``dtype_str``."""
    return np.dtype(dtype_str).newbyteorder("=")


def _ramp(dtype_str: str, shape=(8, 8)) -> np.ndarray:
    """A deterministic ramp image in the requested (possibly big-endian) dtype."""
    n = int(np.prod(shape))
    return np.arange(n, dtype=dtype_str).reshape(shape)


def _find_array_meta(store_path: Path) -> dict:
    """Return the parsed metadata of the first scale array in ``store_path``."""
    # Zarr v2 -> .zarray, Zarr v3 -> zarr.json with node_type == "array".
    zarray = sorted(store_path.rglob(".zarray"))
    if zarray:
        return json.loads(zarray[0].read_text())
    for cand in sorted(store_path.rglob("zarr.json")):
        meta = json.loads(cand.read_text())
        if meta.get("node_type") == "array":
            return meta
    raise AssertionError(f"No array metadata found under {store_path}")


def _first_array_dir(store_path: Path) -> Path:
    """Directory of the first scale array node in a Zarr v3 store."""
    for cand in sorted(store_path.rglob("zarr.json")):
        meta = json.loads(cand.read_text())
        if meta.get("node_type") == "array":
            return cand.parent
    raise AssertionError(f"No array node found under {store_path}")


# ---------------------------------------------------------------------------
# Round-trips: big-endian input in, correct values out.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_str", BIG_ENDIAN_DTYPES)
def test_v04_big_endian_roundtrip(tmp_path, dtype_str):
    """OME-Zarr 0.4 preserves big-endian on disk and reads correct values."""
    expected = _ramp(dtype_str)
    image = to_ngff_image(expected, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[])

    store_path = tmp_path / "be_v04.zarr"
    to_ngff_zarr(str(store_path), multiscales, version="0.4")

    # Zarr v2 records the byte order in the dtype string and keeps it big-endian.
    meta = _find_array_meta(store_path)
    assert meta["dtype"].startswith(">"), meta["dtype"]

    multiscales_back = from_ngff_zarr(str(store_path))
    back = np.asarray(multiscales_back.images[0].data)

    # Values are preserved; byte order is normalized to native on read.
    assert np.array_equal(back, expected.astype(_native(dtype_str)))
    assert back.dtype.byteorder in ("=", "<", "|")


@requires_zarr_v3
@pytest.mark.parametrize("dtype_str", BIG_ENDIAN_DTYPES)
def test_v05_big_endian_roundtrip(tmp_path, dtype_str):
    """OME-Zarr 0.5 round-trips big-endian input with correct values."""
    expected = _ramp(dtype_str)
    image = to_ngff_image(expected, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[])

    store_path = tmp_path / "be_v05.zarr"
    to_ngff_zarr(str(store_path), multiscales, version="0.5")

    # Zarr v3 normalizes to little-endian storage via the ``bytes`` codec.
    meta = _find_array_meta(store_path)
    bytes_codec = next(c for c in meta["codecs"] if c["name"] == "bytes")
    assert bytes_codec["configuration"]["endian"] == "little"

    multiscales_back = from_ngff_zarr(str(store_path))
    back = np.asarray(multiscales_back.images[0].data)

    assert np.array_equal(back, expected.astype(_native(dtype_str)))
    assert back.dtype.byteorder in ("=", "<", "|")


@pytest.mark.parametrize("version", VERSION_PARAMS)
def test_big_endian_matches_little_endian(tmp_path, version):
    """Big- and little-endian inputs with equal values read back identically."""
    big = _ramp(">u2", shape=(4, 6))
    little = big.astype("<u2")
    assert np.array_equal(big, little)  # same logical values, different storage

    results = []
    for tag, arr in (("be", big), ("le", little)):
        ms = to_multiscales(to_ngff_image(arr, dims=["y", "x"]), scale_factors=[])
        store_path = tmp_path / f"{tag}_{version}.zarr"
        to_ngff_zarr(str(store_path), ms, version=version)
        results.append(np.asarray(from_ngff_zarr(str(store_path)).images[0].data))

    assert np.array_equal(results[0], results[1])


# ---------------------------------------------------------------------------
# Reading big-endian data produced "by another tool".
# ---------------------------------------------------------------------------


def test_read_external_big_endian_v04(tmp_path):
    """A Zarr v2 store whose array dtype is ``>u2`` reads with correct values."""
    expected = _ramp(">u2")
    ms = to_multiscales(to_ngff_image(expected, dims=["y", "x"]), scale_factors=[])
    store_path = tmp_path / "ext_v04.zarr"
    to_ngff_zarr(str(store_path), ms, version="0.4")

    # Confirm the bytes on disk really are big-endian: the Zarr v2 ``.zarray``
    # records the byte order in its ``dtype`` string and keeps it ``>``.  Read
    # the metadata directly so the check does not depend on zarr-version-specific
    # group-traversal APIs (``Group.members`` exists only in zarr-python 3).
    meta = _find_array_meta(store_path)
    assert meta["dtype"].startswith(">"), meta["dtype"]

    back = np.asarray(from_ngff_zarr(str(store_path)).images[0].data)
    assert np.array_equal(back, expected.astype(_native(">u2")))


@requires_zarr_v3
def test_read_external_big_endian_v05(tmp_path):
    """A Zarr v3 store whose array uses an ``endian: "big"`` bytes codec reads
    with correct values (the external-tool scenario for OME-Zarr 0.5)."""
    from zarr.codecs import BytesCodec

    expected = _ramp(">u2")
    ms = to_multiscales(to_ngff_image(expected, dims=["y", "x"]), scale_factors=[])
    store_path = tmp_path / "ext_v05.zarr"
    to_ngff_zarr(str(store_path), ms, version="0.5")

    # Re-encode the inner array as big-endian in place, mimicking a writer that
    # emits ``endian: "big"``.  Reuse ngff-zarr's own group metadata so the
    # store stays a valid OME-Zarr 0.5.
    arr_dir = _first_array_dir(store_path)
    src = zarr.open_array(str(arr_dir), mode="r")
    data = np.asarray(src[:])
    chunks = src.chunks
    shutil.rmtree(arr_dir)
    be = zarr.create_array(
        store=str(arr_dir),
        shape=data.shape,
        chunks=chunks,
        dtype="uint16",
        serializer=BytesCodec(endian="big"),
    )
    be[:] = data
    meta = json.loads((arr_dir / "zarr.json").read_text())
    bytes_codec = next(c for c in meta["codecs"] if c["name"] == "bytes")
    assert bytes_codec["configuration"]["endian"] == "big"
    zarr.consolidate_metadata(str(store_path))

    back = np.asarray(from_ngff_zarr(str(store_path)).images[0].data)
    assert np.array_equal(back, expected.astype(_native(">u2")))


# ---------------------------------------------------------------------------
# Downsampling big-endian input.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_str", [">u2", ">f4"])
@pytest.mark.parametrize("version", VERSION_PARAMS)
def test_big_endian_downsampling_roundtrip(tmp_path, dtype_str, version):
    """Multiscale generation from big-endian input round-trips every scale."""
    big = _ramp(dtype_str, shape=(16, 16))
    image = to_ngff_image(big, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2])
    assert len(multiscales.images) == 2

    store_path = tmp_path / f"ms_{version}.zarr"
    to_ngff_zarr(str(store_path), multiscales, version=version)

    back = from_ngff_zarr(str(store_path))
    assert len(back.images) == len(multiscales.images)
    for written, read in zip(multiscales.images, back.images):
        written_native = np.asarray(written.data).astype(_native(dtype_str))
        read_arr = np.asarray(read.data)
        assert read_arr.shape == written_native.shape
        np.testing.assert_allclose(read_arr, written_native, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# TensorStore write path.
#
# The TensorStore Zarr v3 writer maps the array dtype to a Zarr v3 core dtype
# name via ``_numpy_to_zarr_dtype``.  That mapping must canonicalize byte order:
# a big-endian ``>u2`` has to resolve to ``"uint16"``, not fail.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype_str,expected",
    [
        (">u2", "uint16"),
        (">i2", "int16"),
        (">i4", "int32"),
        (">u4", "uint32"),
        (">f4", "float32"),
        (">f8", "float64"),
        ("<u2", "uint16"),
        ("uint16", "uint16"),
    ],
)
def test_numpy_to_zarr_dtype_canonicalizes_byteorder(dtype_str, expected):
    """The Zarr v3 dtype mapping is byte-order independent."""
    from ngff_zarr.to_ngff_zarr import _numpy_to_zarr_dtype

    assert _numpy_to_zarr_dtype(np.dtype(dtype_str)) == expected


def test_numpy_to_zarr_dtype_rejects_unsupported():
    """A dtype with no Zarr v3 core equivalent raises a clear error."""
    from ngff_zarr.to_ngff_zarr import _numpy_to_zarr_dtype

    with pytest.raises(ValueError, match="cannot be mapped to Zarr v3 core dtype"):
        _numpy_to_zarr_dtype(np.dtype("datetime64[s]"))


@pytest.mark.filterwarnings("ignore:use_tensorstore is deprecated:DeprecationWarning")
@pytest.mark.parametrize("dtype_str", [">u2", ">f4"])
@pytest.mark.parametrize("version", VERSION_PARAMS)
def test_big_endian_zarrista_roundtrip(tmp_path, version, dtype_str):
    """Writing big-endian input via the zarrista fast path round-trips."""
    pytest.importorskip("zarrista")

    expected = _ramp(dtype_str)
    image = to_ngff_image(expected, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[])

    store_path = tmp_path / f"zarrista_{version}.zarr"
    to_ngff_zarr(str(store_path), multiscales, version=version, use_tensorstore=True)

    back = np.asarray(from_ngff_zarr(str(store_path)).images[0].data)
    assert np.array_equal(back, expected.astype(_native(dtype_str)))
