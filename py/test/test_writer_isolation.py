# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""ngff-zarr should not inherit backend defaults, nor leak into its caller.

The baseline comparison works on decompressed values, so neither of these
shows up there. They need their own tests.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from ngff_zarr import config, to_multiscales, to_ngff_image, to_ngff_zarr
from packaging import version

zarr_version = version.parse(zarr.__version__)

pytestmark = [
    pytest.mark.skipif(
        zarr_version < version.parse("3.0.0b1"), reason="zarr version < 3.0.0b1"
    ),
    pytest.mark.filterwarnings(
        "ignore:use_tensorstore is deprecated:DeprecationWarning"
    ),
]


def _multiscales(shape=(128, 256, 256), dtype=np.uint16):
    image = to_ngff_image(np.zeros(shape, dtype=dtype), dims=("z", "y", "x"))
    return to_multiscales(image, [2])


def _scale0_metadata(store_path):
    return json.loads((Path(store_path) / "scale0" / "image" / "zarr.json").read_text())


def test_zarrista_matches_zarr_python_by_default():
    """Left unset, the fast-path backend must land on zarr-python's defaults,
    not its own (the tensorstore backend once wrote sharded, uncompressed)."""
    pytest.importorskip("zarrista")
    ms = _multiscales()

    written = {}
    for use_fast_path in (False, True):
        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(tmpdir, ms, version="0.5", use_tensorstore=use_fast_path)
            meta = _scale0_metadata(tmpdir)
            written[use_fast_path] = (
                meta["chunk_grid"]["configuration"]["chunk_shape"],
                [codec["name"] for codec in meta["codecs"]],
            )

    assert written[True] == written[False], (
        f"backends disagree: zarr-python wrote {written[False]}, "
        f"zarrista wrote {written[True]}"
    )
    assert "sharding_indexed" not in written[True][1], (
        "sharding was never requested via chunks_per_shard"
    )


@pytest.mark.parametrize(
    # No shuffle/typesize given: zarr-python evolves them from the dtype,
    # and the TensorStore path must land on the same configuration.
    "explicit_shuffle",
    [True, False],
    ids=["explicit-shuffle", "evolved-from-dtype"],
)
def test_zarrista_honours_requested_compressor(explicit_shuffle):
    """A supplied codec must reach the store with the same configuration
    zarr-python would write, not just the same name."""
    pytest.importorskip("zarrista")
    # Import from the submodule: recent zarr no longer re-exports the codec
    # classes from zarr.codecs.
    from zarr.codecs.blosc import BloscCodec, BloscShuffle

    kwargs = {"cname": "zlib", "clevel": 5}
    if explicit_shuffle:
        kwargs["shuffle"] = BloscShuffle.shuffle
    compressors = BloscCodec(**kwargs)
    ms = _multiscales()

    written = {}
    for use_fast_path in (False, True):
        with tempfile.TemporaryDirectory() as tmpdir:
            to_ngff_zarr(
                tmpdir,
                ms,
                version="0.5",
                use_tensorstore=use_fast_path,
                compressors=compressors,
            )
            codecs = _scale0_metadata(tmpdir)["codecs"]
            written[use_fast_path] = next(
                (codec for codec in codecs if codec["name"] == "blosc"), None
            )

    assert written[False] is not None, "zarr-python did not write blosc"
    assert written[True] == written[False], (
        f"backends disagree: zarr-python wrote {written[False]}, "
        f"zarrista wrote {written[True]}"
    )


def test_to_multiscales_leaves_its_input_alone():
    """Callers reuse images across calls; results must not depend on history."""
    image = to_ngff_image(np.zeros((512, 512), dtype=np.uint8), dims=("y", "x"))
    data_before = image.data
    callbacks_before = list(image.computed_callbacks)

    to_multiscales(image, [2, 4], chunks=(64, 64))

    assert image.data is data_before, "to_multiscales replaced the caller's data"
    assert image.computed_callbacks == callbacks_before, (
        "to_multiscales appended to the caller's computed_callbacks"
    )


def test_to_multiscales_does_not_leak_cache_cleanup():
    """The large-image path registers a cleanup callback; keep it local."""
    original_target = config.memory_target
    config.memory_target = int(1e5)
    try:
        image = to_ngff_image(np.zeros((512, 512), dtype=np.uint8), dims=("y", "x"))
        before = len(image.computed_callbacks)
        to_multiscales(image, [2])
        assert len(image.computed_callbacks) == before, (
            "cache cleanup piled up on the caller's image"
        )
    finally:
        config.memory_target = original_target


def test_to_ngff_zarr_leaves_multiscales_alone():
    """The write loop swaps data for on-disk views and regenerates scales;
    none of that may reach the caller's multiscales."""
    image = to_ngff_image(
        np.zeros((120, 120), dtype=np.uint8), dims=("y", "x"), scale={"y": 1, "x": 1}
    )
    ms = to_multiscales(image, [2, 3, 4])
    data_before = [img.data for img in ms.images]
    shapes_before = [img.data.shape for img in ms.images]

    with tempfile.TemporaryDirectory() as tmpdir:
        to_ngff_zarr(tmpdir, ms, version="0.5")

    for i, img in enumerate(ms.images):
        assert img.data is data_before[i], (
            f"to_ngff_zarr replaced the caller's data at scale {i}"
        )
        assert img.data.shape == shapes_before[i], (
            f"to_ngff_zarr reshaped the caller's scale {i}"
        )
