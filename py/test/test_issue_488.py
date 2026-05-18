# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Regression tests for GH #488.

Issue: https://github.com/fideus-labs/ngff-zarr/issues/488

Two distinct bugs reported (after PR #503 / #505 fixed the chunk-alignment
warning and the ``KeyError: 'chunk_key_encoding'`` errors):

A. Integer-typed inputs were silently downsampled to all zeros (or grossly
   truncated values) by ``ITKWASM_GAUSSIAN`` because the upstream
   ``itkwasm_downsample`` Gaussian filter loses precision when operating on
   integer types.  The fix in ``_itkwasm_blur_and_downsample`` casts integer
   inputs to ``float32`` before downsampling and rounds back to the original
   dtype afterwards.

B. For OME-Zarr v0.5 writes that did not use sharding, the user-specified
   ``chunks=`` argument to ``to_multiscales`` was discarded for the
   highest-resolution level because ``zarr.create_array`` was called without
   ``chunks=``.  The fix in ``_write_array_direct`` forwards the dask array
   chunks when none are otherwise present.
"""

import tempfile
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import zarr
from ngff_zarr import (
    Methods,
    from_ngff_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
)

zarr_version_major = packaging.version.parse(zarr.__version__).major
ome_zarr_versions = ["0.4"] + (["0.5"] if zarr_version_major >= 3 else [])


# ---------------------------------------------------------------------------
# Bug A: ITKWASM_GAUSSIAN preserves integer-typed inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32])
def test_itkwasm_gaussian_integer_constant_preserved(dtype):
    """A constant integer image must remain constant after Gaussian downsample.

    Before the fix the downsampled scales were all zeros for ``np.ones`` input
    (uint16): the upstream ``itkwasm_downsample`` Gaussian filter does
    integer arithmetic that truncates 1 to 0.
    """
    value = 1
    data = np.full((1, 2, 32, 32, 32), value, dtype=dtype)
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image,
        chunks=(1, 1, 16, 16, 16),
        scale_factors=[2, 4],
        method=Methods.ITKWASM_GAUSSIAN,
    )
    for level, img in enumerate(multiscales.images):
        arr = np.asarray(img.data)
        assert arr.dtype == dtype, f"scale {level}: dtype changed to {arr.dtype}"
        assert arr.min() == value, (
            f"scale {level}: min={arr.min()} expected {value} (dtype={dtype})"
        )
        assert arr.max() == value, (
            f"scale {level}: max={arr.max()} expected {value} (dtype={dtype})"
        )


def test_itkwasm_gaussian_uint16_small_values_not_truncated():
    """Small uint16 values must survive Gaussian downsampling without becoming 0."""
    for value in (1, 2, 5, 10):
        data = np.full((1, 2, 32, 32, 32), value, dtype=np.uint16)
        image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
        multiscales = to_multiscales(
            image,
            scale_factors=[2],
            method=Methods.ITKWASM_GAUSSIAN,
        )
        scale1 = np.asarray(multiscales.images[1].data)
        # Within rounding of the Gaussian filter, the result should be
        # very close to ``value`` (within +/- 1).  Crucially it must not
        # be zero for nonzero input.
        assert scale1.min() > 0, (
            f"value={value}: scale1 contained zeros after downsample"
        )
        assert abs(int(scale1.mean()) - value) <= 1, (
            f"value={value}: mean drifted to {scale1.mean()}"
        )


def test_itkwasm_gaussian_float_unaffected_by_fix():
    """Float inputs must continue to work — the fix only intercepts integers."""
    data = np.ones((1, 2, 32, 32, 32), dtype=np.float32)
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image,
        scale_factors=[2],
        method=Methods.ITKWASM_GAUSSIAN,
    )
    for level, img in enumerate(multiscales.images):
        arr = np.asarray(img.data)
        assert arr.dtype == np.float32, f"scale {level}: dtype={arr.dtype}"
        # Float gaussian retains ~1.0 for an all-ones field
        assert np.isclose(arr.min(), 1.0, atol=1e-3)
        assert np.isclose(arr.max(), 1.0, atol=1e-3)


@pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
def test_issue_488_user_reproducer_roundtrip(ome_zarr_version):
    """End-to-end roundtrip mirroring the user's reproducer in #488.

    The original report uses ``numpy.ones((1, 2, 801, 1146, 1290), dtype=uint16)``
    which is ~4.7 GiB; we use a smaller but similarly-shaped array.  All scales
    must round-trip to all-ones with the user-requested chunk sizes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bug488.zarr"
        data = np.ones((1, 2, 80, 114, 129), dtype=np.uint16)
        image = to_ngff_image(
            data,
            dims=["t", "c", "z", "y", "x"],
            translation=dict.fromkeys("tczyx", 0),
            scale=dict.fromkeys("tczyx", 1),
        )
        multiscales = to_multiscales(
            image, chunks=(1, 1, 64, 64, 64), scale_factors=[2, 4]
        )
        to_ngff_zarr(str(out), multiscales, version=ome_zarr_version)

        read = from_ngff_zarr(str(out))
        assert len(read.images) == 3
        for level, img in enumerate(read.images):
            arr = np.asarray(img.data)
            assert arr.min() == 1, (
                f"scale {level}: min={arr.min()} (expected 1) "
                "— Gaussian downsample lost integer precision"
            )
            assert arr.max() == 1, f"scale {level}: max={arr.max()} (expected 1)"


# ---------------------------------------------------------------------------
# Bug B: chunks= parameter respected for v0.5 writes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ome_zarr_version", ome_zarr_versions)
def test_chunks_parameter_preserved_on_roundtrip(ome_zarr_version):
    """``chunks=(1,1,64,64,64)`` must round-trip through the OME-Zarr writer.

    Before the fix the scale-0 array written for v0.5 used zarr's heuristic
    default chunks instead of the user's request, producing surprising shapes
    like ``(1,1,40,57,129)`` for a ``(1,2,80,114,129)`` array.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "chunks.zarr"
        data = np.full((1, 2, 80, 114, 129), 100, dtype=np.uint16)
        image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
        multiscales = to_multiscales(
            image, chunks=(1, 1, 64, 64, 64), scale_factors=[2, 4]
        )
        to_ngff_zarr(str(out), multiscales, version=ome_zarr_version)

        read = from_ngff_zarr(str(out))
        # Scale 0 chunks are exactly what the user asked for.
        assert read.images[0].data.chunksize == (1, 1, 64, 64, 64), (
            f"scale 0 chunks={read.images[0].data.chunksize}"
        )
        # Lower scales clip to dim size where smaller than the requested chunk.
        assert read.images[1].data.chunksize == (1, 1, 40, 57, 64)
        assert read.images[2].data.chunksize == (1, 1, 20, 28, 32)


@pytest.mark.skipif(zarr_version_major < 3, reason="OME-Zarr 0.5 requires zarr v3")
def test_chunks_default_not_overridden_by_zarr_heuristic():
    """When the user does not pass chunks, the dask chunks (chosen by
    ``to_multiscales``) must still be the ones written, not zarr v3's
    heuristic.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "default_chunks.zarr"
        data = np.full((1, 2, 80, 114, 129), 100, dtype=np.uint16)
        image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
        multiscales = to_multiscales(image, scale_factors=[2])
        expected = multiscales.images[0].data.chunksize
        to_ngff_zarr(str(out), multiscales, version="0.5")
        read = from_ngff_zarr(str(out))
        assert read.images[0].data.chunksize == expected


@pytest.mark.skipif(zarr_version_major < 3, reason="OME-Zarr 0.5 requires zarr v3")
def test_chunks_robust_to_non_uniform_leading_chunk():
    """The forwarded chunk shape must use the array's canonical chunk size
    (``arr.chunksize``), not ``c[0]``.

    Slicing or other dask operations can produce a smaller leading chunk
    (e.g. ``arr[5:]`` yields a first chunk of size 5 while the rest are 10).
    Using ``arr.chunks[i][0]`` would forward 5 — the partial leading chunk —
    rather than the array's intended 10-element chunk size.  This test
    constructs that pathological input via ``rechunk`` semantics and asserts
    the on-disk chunks reflect the canonical size.
    """
    import dask.array

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nonuniform.zarr"
        # Construct an array whose leading chunk is smaller than the rest.
        # We do this by chunking the data with explicit per-dim block sizes
        # that produce a small first block on the z axis.
        np_data = np.full((1, 1, 25, 64, 64), 7, dtype=np.uint16)
        da_data = dask.array.from_array(
            np_data, chunks=((1,), (1,), (5, 10, 10), (64,), (64,))
        )
        image = to_ngff_image(da_data, dims=["t", "c", "z", "y", "x"])
        # Skip to_multiscales' own rechunk by writing a single-level
        # multiscales with the already-chunked dask array.
        multiscales = to_multiscales(image, scale_factors=[2])
        # Re-stitch the base level with the non-uniform chunking so we are
        # exercising the ``arr.chunksize`` path (to_multiscales would have
        # normalised the chunks).
        multiscales.images[0].data = da_data
        to_ngff_zarr(str(out), multiscales, version="0.5")
        read = from_ngff_zarr(str(out))
        # ``c[0]`` for z would be 5; ``arr.chunksize`` for z is 10.  The
        # written chunks must reflect 10.
        assert read.images[0].data.chunksize[2] == 10, (
            f"z chunk size was {read.images[0].data.chunksize[2]} — "
            "leading partial chunk leaked into the on-disk chunk size"
        )


@pytest.mark.skipif(zarr_version_major < 3, reason="Sharding requires zarr v3")
def test_chunks_per_shard_path_still_works():
    """The Bug B fix is gated on ``chunks`` not already being set in
    ``to_zarr_kwargs``.  When sharding is configured ``cleaned_sharding_kwargs``
    sets ``chunks`` so the new branch becomes a no-op — confirm the sharding
    path still round-trips correctly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sharded.zarr"
        data = np.full((1, 2, 64, 64, 64), 100, dtype=np.uint16)
        image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
        multiscales = to_multiscales(
            image, chunks=(1, 1, 32, 32, 32), scale_factors=[2]
        )
        to_ngff_zarr(str(out), multiscales, version="0.5", chunks_per_shard=2)
        read = from_ngff_zarr(str(out))
        for level, img in enumerate(read.images):
            arr = np.asarray(img.data)
            assert arr.min() == 100, f"scale {level}: data corrupted"
            assert arr.max() == 100, f"scale {level}: data corrupted"


# ---------------------------------------------------------------------------
# Bug A edge cases — extreme values, signed ints, and the unaffected
# label_image path
# ---------------------------------------------------------------------------


def test_itkwasm_gaussian_int16_negative_values_preserved():
    """Signed integers with negative values must survive the float workaround.

    The ``np.clip`` in the fix uses ``np.iinfo`` bounds, so this exercises the
    ``signed`` branch (clip min = -32768, not 0).
    """
    data = np.full((1, 1, 32, 32, 32), -1000, dtype=np.int16)
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image, scale_factors=[2], method=Methods.ITKWASM_GAUSSIAN
    )
    scale1 = np.asarray(multiscales.images[1].data)
    assert scale1.dtype == np.int16
    assert scale1.min() == -1000
    assert scale1.max() == -1000


def test_itkwasm_gaussian_uint8_near_max_no_overflow():
    """Values near a dtype's upper bound must not overflow during round-trip.

    Tests the ``np.clip`` upper bound — without it a float result slightly
    above 255 (e.g. 255.4 from kernel arithmetic) could wrap to 0 when cast
    to uint8.
    """
    data = np.full((1, 1, 32, 32, 32), 255, dtype=np.uint8)
    image = to_ngff_image(data, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image, scale_factors=[2], method=Methods.ITKWASM_GAUSSIAN
    )
    scale1 = np.asarray(multiscales.images[1].data)
    assert scale1.dtype == np.uint8
    assert scale1.min() == 255
    assert scale1.max() == 255


def test_itkwasm_label_image_integer_unchanged_by_fix():
    """``label_image`` downsampling already worked for integer dtypes — confirm
    the Gaussian-only workaround did not regress it.
    """
    labels = np.zeros((1, 1, 32, 32, 32), dtype=np.uint16)
    labels[..., :16, :, :] = 5
    labels[..., 16:, :, :] = 7
    image = to_ngff_image(labels, dims=["t", "c", "z", "y", "x"])
    multiscales = to_multiscales(
        image, scale_factors=[2], method=Methods.ITKWASM_LABEL_IMAGE
    )
    scale1 = np.asarray(multiscales.images[1].data)
    assert scale1.dtype == np.uint16
    # Label image downsampling assigns each output voxel a label from the
    # input — the only values that may appear are the originals.
    unique = set(np.unique(scale1).tolist())
    assert unique.issubset({5, 7}), f"unexpected labels: {unique}"
