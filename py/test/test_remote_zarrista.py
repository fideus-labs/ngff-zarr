# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Remote reads through zarrista's async API over obstore.

A localhost HTTP server with Range support (object stores serve ranged
GETs; Python's ``SimpleHTTPRequestHandler`` alone does not) stands in for
remote storage, so these tests exercise the real obstore/zarrista network
stack without leaving the machine.
"""

import http.server
import io
import json
import os
import re
import socketserver
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import dask.array
import numpy as np
import pytest
from ngff_zarr import from_ngff_zarr, to_multiscales, to_ngff_image, to_ngff_zarr

# zarrista requires Python >= 3.11; the py310 environment runs without it,
# and obstore ships via the `remote` extra.
pytest.importorskip("zarrista")
pytest.importorskip("obstore")

from ngff_zarr._remote_reader import (
    RemoteZarrStore,
    _translate_storage_options,
)


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with the Range support obstore requires."""

    def log_message(self, *args):
        pass

    def send_head(self):
        path = self.translate_path(self.path)
        range_header = self.headers.get("Range")
        if range_header is None or not os.path.isfile(path):
            return super().send_head()
        match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
        if match is None:
            return super().send_head()
        size = os.path.getsize(path)
        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
        else:
            start, end = size - int(end_text), size - 1
        end = min(end, size - 1)
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return io.BytesIO(body)


@contextmanager
def _serve(directory: Path):
    """Serve *directory* over localhost HTTP, yielding the base URL."""

    class Handler(_RangeHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()
            thread.join()


def _forbid_zarr_python():
    """Patches that make any zarr-python read explode."""
    return (
        patch("zarr.open_group", side_effect=AssertionError("zarr-python used")),
        patch("dask.array.from_zarr", side_effect=AssertionError("zarr-python used")),
    )


def _write_ramp_store(store_path, version: str, **to_zarr_kwargs):
    """A small two-scale OME-Zarr store with deterministic values."""
    data = np.arange(16 * 16, dtype="uint16").reshape(16, 16)
    image = to_ngff_image(data, dims=["y", "x"])
    multiscales = to_multiscales(image, scale_factors=[2], chunks=8)
    to_ngff_zarr(str(store_path), multiscales, version=version, **to_zarr_kwargs)
    return data


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_remote_http_read_bypasses_zarr_python(tmp_path, version):
    """Remote URL stores read through zarrista/obstore, not zarr-python."""
    data = _write_ramp_store(tmp_path / "image.zarr", version)

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with _serve(tmp_path) as base_url, open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(f"{base_url}/image.zarr")
        scale0 = multiscales.images[0].data
        # The read stays lazy with the on-disk chunking.
        assert isinstance(scale0, dask.array.Array)
        assert scale0.chunksize == (8, 8)
        assert np.array_equal(np.asarray(scale0), data)
    assert len(multiscales.images) == 2


def test_remote_http_read_sharded(tmp_path):
    """Sharded zarr v3 stores decode remotely on the inner-chunk grid."""
    data = _write_ramp_store(tmp_path / "image.zarr", "0.5", chunks_per_shard=2)

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with _serve(tmp_path) as base_url, open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(f"{base_url}/image.zarr")
        scale0 = multiscales.images[0].data
        assert scale0.chunksize == (8, 8)
        assert np.array_equal(np.asarray(scale0), data)


def test_remote_http_read_validate(tmp_path):
    """validate=True works on remote reads (attrs-based validation)."""
    _write_ramp_store(tmp_path / "image.zarr", "0.4")

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with _serve(tmp_path) as base_url, open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(f"{base_url}/image.zarr", validate=True)
    assert len(multiscales.images) == 2


def test_remote_missing_store_error(tmp_path):
    """A URL with no zarr store yields the shared group-not-found error."""
    with _serve(tmp_path) as base_url:
        with pytest.raises(ValueError, match="No valid Zarr group found"):
            from_ngff_zarr(f"{base_url}/nope.zarr")


def test_remote_array_at_root_error(tmp_path):
    """An array at the store root yields the shared array-at-root error."""
    import zarr

    zarr.save(str(tmp_path / "array.zarr"), np.arange(16, dtype="uint8"))

    with _serve(tmp_path) as base_url:
        with pytest.raises(ValueError, match="contains an array at the root level"):
            from_ngff_zarr(f"{base_url}/array.zarr")


def test_remote_spurious_v3_root_falls_back_to_v2_attrs(tmp_path):
    """A spurious empty zarr.json must not hide a v2 store's attributes."""
    data = _write_ramp_store(tmp_path / "image.zarr", "0.4")
    (tmp_path / "image.zarr" / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with _serve(tmp_path) as base_url, open_group_patch, from_zarr_patch:
        multiscales = from_ngff_zarr(f"{base_url}/image.zarr")
        assert np.array_equal(np.asarray(multiscales.images[0].data), data)


def test_remote_untranslatable_storage_options_warn(tmp_path):
    """Options with no obstore equivalent warn and are ignored, not fatal."""
    data = _write_ramp_store(tmp_path / "image.zarr", "0.4")

    with _serve(tmp_path) as base_url:
        with pytest.warns(UserWarning, match="bogus_option"):
            multiscales = from_ngff_zarr(
                f"{base_url}/image.zarr",
                storage_options={"headers": {"x-test": "1"}, "bogus_option": True},
            )
        assert np.array_equal(np.asarray(multiscales.images[0].data), data)


def test_remote_s3_storage_options_construct_store():
    """fsspec-style S3 options translate into a working obstore S3Store.

    Construction is metadata-only (no network), so this verifies the full
    translation + ``obstore.store.from_url`` path for the s3 scheme.
    """
    handle = RemoteZarrStore(
        "s3://bucket/data.zarr",
        storage_options={
            "anon": True,
            "client_kwargs": {"region_name": "us-west-2"},
        },
    )
    assert type(handle._store).__name__ == "S3Store"
    assert str(handle) == "s3://bucket/data.zarr"


def test_translate_storage_options_s3_aliases():
    translated, passthrough = _translate_storage_options(
        "s3",
        {
            "anon": True,
            "key": "AKIA...",
            "secret": "shhh",
            "token": "sess",
            "endpoint_url": "https://minio.example.com",
            "client_kwargs": {"region_name": "eu-central-1", "verify": False},
            "unknown_thing": 1,
        },
    )
    assert translated == {
        "skip_signature": True,
        "access_key_id": "AKIA...",
        "secret_access_key": "shhh",
        "session_token": "sess",
        "endpoint": "https://minio.example.com",
        "region": "eu-central-1",
    }
    assert passthrough == {"client_kwargs.verify": False, "unknown_thing": 1}


def test_translate_storage_options_gcs_anon_token():
    translated, passthrough = _translate_storage_options("gs", {"token": "anon"})
    assert translated == {"skip_signature": True}
    assert passthrough == {}
    translated, passthrough = _translate_storage_options(
        "gs", {"token": "/path/creds.json"}
    )
    assert translated == {}
    assert passthrough == {"token": "/path/creds.json"}


def test_remote_hcs_plate_read(tmp_path):
    """HCS plates read remotely: plate metadata, well navigation, pixels."""
    from ngff_zarr.hcs import from_hcs_zarr

    fixture = Path(__file__).parent / "data" / "input" / "hcs.ome.zarr"
    if not fixture.exists():
        pytest.skip("hcs.ome.zarr test fixture not downloaded")

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with _serve(fixture.parent) as base_url, open_group_patch, from_zarr_patch:
        plate = from_hcs_zarr(f"{base_url}/hcs.ome.zarr")
        assert len(plate.wells) == 4
        well = plate.get_well("A", "1")
        assert well is not None
        image = well.get_image(0)
        assert image is not None
        assert isinstance(image.images[0].data, dask.array.Array)
        # Pixel data resolves through the shared obstore client.
        np.asarray(image.images[0].data)


def test_open_array_reads_regions_from_a_remote_store(tmp_path):
    """``open_array`` serves a remote node the way it serves a local one.

    The handle is what a consumer reads regions through, and it was local-only:
    a caller holding a URL had to fall back to another Zarr library for the one
    thing this package could otherwise do end to end.
    """
    from ngff_zarr import from_ome_zarr, open_array

    data = _write_ramp_store(tmp_path / "image.zarr", "0.5")

    open_group_patch, from_zarr_patch = _forbid_zarr_python()
    with _serve(tmp_path) as base_url, open_group_patch, from_zarr_patch:
        store = f"{base_url}/image.zarr"
        path = from_ome_zarr(store).metadata.datasets[0].path
        array = open_array(store, path)

        assert array.shape == data.shape
        assert array.dtype == data.dtype
        assert array.chunks == (8, 8)
        # A window, and the whole array: the same values the store holds.
        assert np.array_equal(np.asarray(array[4:12, 2:6]), data[4:12, 2:6])
        assert np.array_equal(np.asarray(array[...]), data)


def test_a_remote_array_refuses_a_region_write(tmp_path):
    """Read-only, and said so: the write path needs a local directory store,
    and a silent no-op would leave a producer believing its region landed."""
    from ngff_zarr import from_ome_zarr, open_array

    _write_ramp_store(tmp_path / "image.zarr", "0.5")
    with _serve(tmp_path) as base_url:
        store = f"{base_url}/image.zarr"
        array = open_array(store, from_ome_zarr(store).metadata.datasets[0].path)
        with pytest.raises(TypeError, match="read-only"):
            array[0:2, 0:2] = np.zeros((2, 2), dtype="uint16")


def test_open_array_on_a_remote_group_is_refused(tmp_path):
    from ngff_zarr import open_array

    _write_ramp_store(tmp_path / "image.zarr", "0.5")
    with _serve(tmp_path) as base_url:
        with pytest.raises(ValueError, match="is a group, not an array"):
            open_array(f"{base_url}/image.zarr")


def test_a_store_object_is_not_opened_as_a_url():
    """A wrapper around a URL is remote, but it is not a URL.

    ``_is_remote_store`` calls a zarr-python ``FsspecStore`` remote by its
    filesystem protocol, while its ``str()`` is a repr rather than the URL
    obstore would be handed. ``from_ome_zarr`` does not route one through the
    remote engine either, so neither does this: it keeps the path it has today.
    """
    from ngff_zarr._zarrista_utils import _remote_handle
    from ngff_zarr.from_ngff_zarr import _is_remote_store

    class _FsspecStoreLike:
        class fs:
            protocol = "s3"

        path = "bucket/image.zarr"

    store = _FsspecStoreLike()
    assert _is_remote_store(store)
    assert _remote_handle(store, None) is None


def test_a_url_without_the_remote_extra_names_the_extra(monkeypatch):
    """The install hint ``from_ome_zarr`` gives, given here too: reaching
    obstore's own constructor instead raises a bare ModuleNotFoundError."""
    import ngff_zarr._remote_reader as remote_reader
    from ngff_zarr._zarrista_utils import _remote_handle

    monkeypatch.setattr(remote_reader, "remote_read_available", lambda: False)
    with pytest.raises(ImportError, match=r"ngff-zarr\[remote\]"):
        _remote_handle("https://example.org/image.zarr", None)
