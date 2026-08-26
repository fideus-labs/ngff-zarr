# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Pin the breaking-change contracts quoted in the v0.43 -> v0.44 guide.

``docs/migration_guides/v0-43_to_v0-44.md`` quotes verbatim error messages
and ships a repair script for stores carrying a non-spec ``nthreads`` blosc
key. Users match on those strings to find the right section, so a reworded
message silently invalidates the guide -- and the FAQ error table in
``docs/faq.md`` that links into it.

These tests assert the substrings the guide leads with. If one fails, update
the guide, the FAQ table, and this module together.

The read-side store-object rejection and the ``use_tensorstore`` deprecation
warning are already pinned by ``test_from_ngff_zarr.py`` and
``test_to_ngff_zarr_zarrista.py``; the consolidated-metadata tolerance for
the legacy blosc keys is pinned by
``test_from_ngff_zarr_reads_with_nthreads``. This module covers the
documented behaviors those leave untested.
"""

import json

import numpy as np
import pytest
from ngff_zarr import config, from_ome_zarr, to_multiscales, to_ome_zarr

BLOSC_COMPRESSOR = {
    "name": "blosc",
    "configuration": {
        "cname": "zstd",
        "clevel": 5,
        "shuffle": "shuffle",
        "typesize": 1,
        "blocksize": 0,
    },
}


def _multiscales():
    """A single-scale 8x8 image, the smallest valid write payload."""
    return to_multiscales(np.zeros((8, 8), dtype=np.uint8), scale_factors=[])


def _write_blosc_store(path):
    """Write a v0.5 store whose arrays use the blosc codec."""
    to_ome_zarr(
        str(path),
        _multiscales(),
        version="0.5",
        compressors=[BLOSC_COMPRESSOR],
    )


def _inject_array_level_nthreads(root):
    """Add the non-spec ``nthreads`` key to every array-level blosc codec."""
    injected = False
    for doc_path in root.rglob("zarr.json"):
        doc = json.loads(doc_path.read_text())
        changed = False
        for codec in doc.get("codecs") or []:
            if codec.get("name") == "blosc":
                codec["configuration"]["nthreads"] = 4
                changed = True
        if changed:
            doc_path.write_text(json.dumps(doc))
            injected = True
    return injected


class TestWriteTargetRejection:
    """`to_ome_zarr` writes to local directory and .ozx paths only."""

    def test_rejects_zarr_python_store_object(self, tmp_path):
        """A zarr-python store object names itself in the write TypeError."""
        zarr_storage = pytest.importorskip("zarr.storage")
        if not hasattr(zarr_storage, "LocalStore"):
            pytest.skip("zarr-python 2.x stores are MutableMappings")

        with pytest.raises(
            TypeError, match="ngff-zarr writes to local directory paths"
        ) as exc_info:
            to_ome_zarr(
                zarr_storage.LocalStore(str(tmp_path / "out.ome.zarr")),
                _multiscales(),
            )

        message = str(exc_info.value)
        assert "zarr-python store objects are no longer accepted" in message

    def test_rejects_remote_url(self):
        """Remote writes are unsupported; the guide says upload afterwards."""
        with pytest.raises(
            TypeError, match="ngff-zarr writes to local directory paths"
        ) as exc_info:
            to_ome_zarr("s3://my-bucket/image.ome.zarr", _multiscales())

        assert "upload afterwards" in str(exc_info.value)

    def test_rejects_in_memory_mapping(self):
        """Mappings are a read-only input form; writes reject them."""
        with pytest.raises(
            TypeError, match="ngff-zarr writes to local directory paths"
        ) as exc_info:
            to_ome_zarr({}, _multiscales())

        assert "In-memory mappings" in str(exc_info.value)

    def test_rejects_zip_target(self, tmp_path):
        """.zip is read-only; .ozx goes through the RFC-9 staging path."""
        with pytest.raises(
            ValueError, match="cannot write into zip archives"
        ) as exc_info:
            to_ome_zarr(str(tmp_path / "image.zip"), _multiscales())

        assert "rfc9_zip" in str(exc_info.value)

    def test_accepts_ozx_target(self, tmp_path):
        """The .ozx counterexample the guide gives must keep working."""
        ozx_path = tmp_path / "image.ozx"
        to_ome_zarr(str(ozx_path), _multiscales(), version="0.5")

        assert ozx_path.is_file()


def test_chunk_store_rejected(tmp_path):
    """`chunk_store` is rejected explicitly, not silently ignored."""
    with pytest.raises(
        TypeError, match="chunk_store is no longer supported"
    ) as exc_info:
        to_ome_zarr(
            str(tmp_path / "out.ome.zarr"),
            _multiscales(),
            chunk_store=str(tmp_path / "chunks"),
        )

    assert "always written to the same local directory store" in str(exc_info.value)


def test_cache_store_must_be_a_directory_path(tmp_path):
    """`config.cache_store` is a directory path, not a store object."""
    zarr_storage = pytest.importorskip("zarr.storage")
    if not hasattr(zarr_storage, "LocalStore"):
        pytest.skip("zarr-python 2.x stores are MutableMappings")

    original = config.cache_store
    config.cache_store = zarr_storage.LocalStore(str(tmp_path / "cache"))
    try:
        from ngff_zarr.to_multiscales import _CacheTarget

        with pytest.raises(
            TypeError, match="config.cache_store must be a local directory path"
        ):
            _CacheTarget.from_config()
    finally:
        config.cache_store = original


class TestArrayLevelNthreads:
    """Array-level `nthreads` is rejected, and the guide's script repairs it."""

    def test_array_level_nthreads_rejected(self, tmp_path):
        """An array-level `nthreads` key fails the read the guide describes."""
        store_path = tmp_path / "nthreads.ome.zarr"
        _write_blosc_store(store_path)
        assert _inject_array_level_nthreads(store_path)

        # zarrista raises ArrayCreateError from the Rust binding; it is not
        # exported from the package namespace, so match on the message the
        # guide and the FAQ table quote instead of the type.
        with pytest.raises(Exception) as exc_info:
            from_ome_zarr(str(store_path))

        assert "BloscCodecConfiguration" in str(exc_info.value)

    def test_blocksize_alone_is_accepted(self, tmp_path):
        """`blocksize` is in the spec; only `nthreads` breaks the read.

        The guide says so explicitly, so an over-broad repair that also
        stripped `blocksize` would be unnecessary.
        """
        store_path = tmp_path / "blocksize.ome.zarr"
        _write_blosc_store(store_path)
        for doc_path in store_path.rglob("zarr.json"):
            doc = json.loads(doc_path.read_text())
            for codec in doc.get("codecs") or []:
                if codec.get("name") == "blosc":
                    codec["configuration"]["blocksize"] = 0
            doc_path.write_text(json.dumps(doc))

        assert from_ome_zarr(str(store_path)) is not None

    def test_documented_repair_script_restores_readability(self, tmp_path):
        """Run the guide's repair snippet verbatim and read the store back."""
        store_path = tmp_path / "repair.ome.zarr"
        _write_blosc_store(store_path)
        assert _inject_array_level_nthreads(store_path)

        # --- as published in the migration guide ---
        for doc_path in store_path.rglob("zarr.json"):
            doc = json.loads(doc_path.read_text())
            changed = False
            for codec in doc.get("codecs") or []:
                if codec.get("name") == "blosc":
                    configuration = codec.get("configuration", {})
                    if "nthreads" in configuration:
                        del configuration["nthreads"]
                        changed = True
            if changed:
                doc_path.write_text(json.dumps(doc))
        # --- end of published snippet ---

        multiscales = from_ome_zarr(str(store_path))
        np.testing.assert_array_equal(
            multiscales.images[0].data.compute(),
            np.zeros((8, 8), dtype=np.uint8),
        )
