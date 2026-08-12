# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A missing baseline must never read as a passing comparison."""

import pytest
from ngff_zarr import Methods, to_multiscales, to_ngff_image

from ._data import store_equals, verify_against_baseline


def _tiny_multiscales():
    import numpy as np

    image = to_ngff_image(np.zeros((32, 32), dtype=np.uint8), dims=("y", "x"))
    return to_multiscales(image, [2], method=Methods.ITKWASM_GAUSSIAN)


def test_missing_baseline_skips_rather_than_passes():
    with pytest.raises(BaseException) as excinfo:
        verify_against_baseline(
            "does-not-exist", "no/such/baseline.zarr", _tiny_multiscales()
        )
    assert excinfo.typename == "Skipped", (
        f"a missing baseline must skip, got {excinfo.typename}"
    )


def test_store_equals_requires_identical_key_sets():
    """An empty baseline, or an extra key on either side, must not pass."""
    import zarr
    from ngff_zarr import to_ngff_zarr
    from packaging import version
    from zarr.storage import MemoryStore

    multiscales = _tiny_multiscales()
    populated = MemoryStore()
    to_ngff_zarr(populated, multiscales, version="0.4")

    assert not store_equals(MemoryStore(), populated), (
        "an empty baseline compared equal to a populated store"
    )

    test_store = MemoryStore()
    to_ngff_zarr(test_store, multiscales, version="0.4")
    assert store_equals(populated, test_store), (
        "two identical writes should compare equal"
    )

    # zarr 2 has no zarr_format kwarg and writes format 2 anyway.
    extra_kwargs = (
        {"zarr_format": 2}
        if version.parse(zarr.__version__) >= version.parse("3.0.0b1")
        else {}
    )
    zarr.create(shape=(1,), store=test_store, path="extra", **extra_kwargs)
    assert not store_equals(populated, test_store), (
        "an extra array in the test store went unnoticed"
    )


def test_provenance_version_is_ignored_only_in_place():
    """Only the recorded downsampler version may drift, where it belongs.

    The multiscales metadata block records the version of the tool that
    generated the pyramid; upgrades change it while the pixels stay
    identical, so store_equals ignores it in .zattrs and in consolidated
    .zmetadata. A look-alike path nested deeper must still be compared.
    """
    import asyncio
    import json

    import zarr
    from ngff_zarr import to_ngff_zarr
    from packaging import version
    from zarr.storage import MemoryStore

    is_zarr3 = version.parse(zarr.__version__) >= version.parse("3.0.0b1")

    def read(store, key):
        if is_zarr3:
            from zarr.core.buffer import default_buffer_prototype

            return asyncio.run(
                store.get(key, default_buffer_prototype())
            ).to_bytes()
        return store[key]

    def write(store, key, data):
        if is_zarr3:
            from zarr.core.buffer import default_buffer_prototype

            buffer = default_buffer_prototype().buffer.from_bytes(data)
            asyncio.run(store.set(key, buffer))
        else:
            store[key] = data

    def edit_attrs(store, mutate):
        for key in (".zattrs", ".zmetadata"):
            document = json.loads(read(store, key))
            attrs = (
                document["metadata"][".zattrs"] if key == ".zmetadata" else document
            )
            mutate(attrs)
            write(store, key, json.dumps(document).encode())

    multiscales = _tiny_multiscales()
    baseline = MemoryStore()
    test_store = MemoryStore()
    to_ngff_zarr(baseline, multiscales, version="0.4")
    to_ngff_zarr(test_store, multiscales, version="0.4")

    def bump_version(attrs):
        attrs["multiscales"][0]["metadata"]["version"] = "0.0.0-provenance"

    edit_attrs(test_store, bump_version)
    assert store_equals(baseline, test_store), (
        "a changed downsampler version in the provenance metadata must be ignored"
    )

    def nest_lookalike(value):
        def mutate(attrs):
            attrs["nested"] = {
                "multiscales": [{"metadata": {"version": value}}]
            }

        return mutate

    edit_attrs(baseline, nest_lookalike("a"))
    edit_attrs(test_store, nest_lookalike("b"))
    assert not store_equals(baseline, test_store), (
        "a version change on a nested look-alike path went unnoticed"
    )
