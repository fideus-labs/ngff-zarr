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


def _codec_multiscales():
    """Two scale levels, so a chunk layout the migration must carry over."""
    rng = np.random.default_rng(7)
    data = rng.integers(0, 4096, size=(32, 32), dtype=np.uint16)
    return to_multiscales(data, scale_factors=[2], chunks=8)


def _array_codecs(doc):
    """The codec chain, reaching inside the sharding codec when there is one."""
    codecs = doc["codecs"]
    if codecs[0]["name"] == "sharding_indexed":
        return codecs[0]["configuration"]["codecs"]
    return codecs


class TestForeignCodecMigration:
    """The guide's two-writer recipe for a codec chain zarrista cannot encode."""

    def test_codec_arguments_are_rejected_with_metadata_only(self, tmp_path):
        """`metadata_only=True` validates codecs exactly like a full write.

        The guide tells callers to leave the codec arguments off that call
        because of this; passing them along with it raises the same error the
        migration is trying to get out of.
        """
        numcodecs = pytest.importorskip("numcodecs")

        with pytest.raises(ValueError, match="filters are not supported"):
            to_ome_zarr(
                str(tmp_path / "skeleton.ome.zarr"),
                _multiscales(),
                version="0.5",
                metadata_only=True,
                filters=[numcodecs.Delta(dtype="uint8")],
            )

    @pytest.mark.parametrize("chunks_per_shard", [None, 2])
    def test_documented_recipe_writes_the_foreign_chain(
        self, tmp_path, chunks_per_shard
    ):
        """Run the guide's migration snippet and check what it produced.

        The chain is a transpose filter: an array-to-array codec zarr-python
        writes and this writer rejects. The assertions cover what the snippet
        is easy to get wrong -- dropping the skeleton's chunk and shard
        layout, and leaving the consolidated metadata describing the codec
        chain of the skeletons rather than of the arrays.

        The guide writes the skeleton without sharding, which is the
        ``chunks_per_shard=None`` case. The sharded one runs too because
        ``shards=skeleton.shards`` is the line of the snippet most easily
        dropped, and it does nothing unless the skeleton has shards.
        """
        zarr = pytest.importorskip("zarr")
        da = pytest.importorskip("dask.array")
        if not hasattr(zarr, "create_array"):
            pytest.skip("the recipe uses the zarr-python 3 array API")
        from zarr.codecs import TransposeCodec

        store_path = tmp_path / "foreign_codec.ome.zarr"
        multiscales = _codec_multiscales()
        expected = [level.data.compute() for level in multiscales.images]

        # The guide's skeleton call, with this test's sharding parameter added.
        to_ome_zarr(
            str(store_path),
            multiscales,
            version="0.5",
            metadata_only=True,
            chunks_per_shard=chunks_per_shard,
        )

        # --- as published in the migration guide, with the store path and the
        # placeholder codec chain filled in ---
        for dataset, level in zip(multiscales.metadata.datasets, multiscales.images):
            skeleton = zarr.open_array(str(store_path), path=dataset.path, mode="r")
            array = zarr.create_array(
                str(store_path),
                name=dataset.path,
                shape=skeleton.shape,
                chunks=skeleton.chunks,
                shards=skeleton.shards,
                dtype=skeleton.dtype,
                filters=[TransposeCodec(order=(1, 0))],
                fill_value=0,
                dimension_names=list(level.dims),
                overwrite=True,
            )
            write_shape = skeleton.shards or skeleton.chunks
            da.store(level.data.rechunk(write_shape), array, lock=False)
        zarr.consolidate_metadata(str(store_path))
        # --- end of published snippet ---

        # An untouched skeleton is the layout the migration had to carry over.
        reference_path = tmp_path / "reference.ome.zarr"
        to_ome_zarr(
            str(reference_path),
            multiscales,
            version="0.5",
            metadata_only=True,
            chunks_per_shard=chunks_per_shard,
        )

        root = json.loads((store_path / "zarr.json").read_text())
        consolidated = root["consolidated_metadata"]["metadata"]
        for index, dataset in enumerate(multiscales.metadata.datasets):
            doc = json.loads((store_path / dataset.path / "zarr.json").read_text())
            is_sharded = doc["codecs"][0]["name"] == "sharding_indexed"
            assert is_sharded == (chunks_per_shard is not None), (
                "the skeleton's sharding must survive the re-creation"
            )
            assert _array_codecs(doc)[0]["name"] == "transpose"

            written = zarr.open_array(str(store_path), path=dataset.path, mode="r")
            reference = zarr.open_array(
                str(reference_path), path=dataset.path, mode="r"
            )
            assert (written.chunks, written.shards) == (
                reference.chunks,
                reference.shards,
            ), "the skeleton's chunk and shard layout must survive the re-creation"

            entry = consolidated[dataset.path]
            assert entry["codecs"] == doc["codecs"], (
                "a reader trusting the consolidated document would otherwise "
                "see the skeleton's codec chain"
            )

            np.testing.assert_array_equal(np.asarray(written), expected[index])
