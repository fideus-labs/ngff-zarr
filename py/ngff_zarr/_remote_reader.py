# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Remote OME-Zarr reads through zarrista's async API over obstore.

zarrista's synchronous ``Array``/``Group`` only accept its closed store
union (``FilesystemStore``/``MemoryStore``/``ZipStore``); remote object
stores are reached through ``AsyncArray``/``AsyncGroup`` over any obstore
store (HTTP(S), S3, GCS, Azure). This module bridges that async surface to
ngff-zarr's synchronous read path:

- Every zarrista coroutine runs on a dedicated daemon event-loop thread.
  ``asyncio.run`` would fail inside environments that already run a loop
  (e.g. Jupyter), and zarrista's pyo3 futures require a running loop at
  *call* time, so calls are submitted as coroutine factories executed
  inside the loop.
- fsspec-style ``storage_options`` are translated to obstore configuration
  where a mapping exists; unrecognized keys are first passed through
  verbatim (so obstore-native option names keep working) and dropped with
  a warning only when obstore rejects them.
- :class:`RemoteZarrGroup`/:class:`RemoteZarrArray` provide the same read
  surface as the compat layer's ``LocalZarrGroup``/``LocalZarrArray``.

zarrista requires Python >= 3.11 and obstore is an optional dependency
(the ``remote`` extra), so both are imported lazily inside each function.
"""

import asyncio
import importlib.util
import os
import posixpath
import threading
import urllib.error
import urllib.request
import warnings
from urllib.parse import urlsplit

import dask.array
import numpy as np

from ._v2_store_reader import AttrsDict

__all__ = [
    "RemoteZarrArray",
    "RemoteZarrGroup",
    "RemoteZarrStore",
    "open_remote_lazy",
    "open_remote_node",
    "remote_read_available",
]

_REMOTE_READ_AVAILABLE: bool | None = None

# zarrista raises GroupCreateError/ArrayCreateError both when the node's
# metadata documents are absent and for transport failures; only the former
# carries this marker. Tests pin the message so upstream drift surfaces.
_MISSING_METADATA_MARKER = "metadata is missing"


def remote_read_available() -> bool:
    """Whether the zarrista/obstore remote read engine can be used."""
    global _REMOTE_READ_AVAILABLE
    if _REMOTE_READ_AVAILABLE is None:
        _REMOTE_READ_AVAILABLE = (
            importlib.util.find_spec("zarrista") is not None
            and importlib.util.find_spec("obstore") is not None
        )
    return _REMOTE_READ_AVAILABLE


# -- async-to-sync bridge ---------------------------------------------------

_IO_LOOP: asyncio.AbstractEventLoop | None = None
_IO_LOOP_LOCK = threading.Lock()


def _io_loop() -> asyncio.AbstractEventLoop:
    """The shared event loop running on a daemon thread, started on demand."""
    global _IO_LOOP
    with _IO_LOOP_LOCK:
        if _IO_LOOP is None or _IO_LOOP.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="ngff-zarr-remote-io", daemon=True
            ).start()
            _IO_LOOP = loop
    return _IO_LOOP


def _run(factory):
    """Execute the awaitable produced by *factory* on the IO loop.

    *factory* is a zero-argument callable evaluated inside the loop, not a
    coroutine object: zarrista's pyo3 methods create their futures at call
    time and require the running loop to exist then.
    """

    async def _invoke():
        return await factory()

    return asyncio.run_coroutine_threadsafe(_invoke(), _io_loop()).result()


# -- storage_options translation --------------------------------------------

# fsspec option names, per backend, mapped to their obstore equivalents.
# s3fs / gcsfs / adlfs / HTTPFileSystem spellings on the left.
_S3_ALIASES = {
    "anon": "skip_signature",
    "key": "access_key_id",
    "username": "access_key_id",
    "secret": "secret_access_key",
    "password": "secret_access_key",
    "token": "session_token",
    "endpoint_url": "endpoint",
    "region_name": "region",
    "requester_pays": "request_payer",
}
_GS_ALIASES = {
    "endpoint_url": "endpoint",
}
_AZURE_ALIASES = {
    "account_key": "access_key",
    "sas_token": "sas_key",
    "anon": "skip_signature",
}
_HTTP_ALIASES = {
    "headers": "default_headers",
}

_SCHEME_ALIASES = {
    "s3": _S3_ALIASES,
    "s3a": _S3_ALIASES,
    "gs": _GS_ALIASES,
    "gcs": _GS_ALIASES,
    "az": _AZURE_ALIASES,
    "azure": _AZURE_ALIASES,
    "abfs": _AZURE_ALIASES,
    "abfss": _AZURE_ALIASES,
    "http": _HTTP_ALIASES,
    "https": _HTTP_ALIASES,
}

# fsspec's nested ``client_kwargs`` (botocore/aiohttp session options) whose
# values map onto top-level obstore config.
_CLIENT_KWARGS_ALIASES = {
    "region_name": "region",
    "endpoint_url": "endpoint",
}


def _translate_storage_options(
    scheme: str, storage_options: dict | None
) -> tuple[dict, dict]:
    """Split fsspec-style *storage_options* into obstore option dicts.

    Returns ``(translated, passthrough)``: *translated* holds options with a
    known obstore equivalent, *passthrough* the remaining keys forwarded
    verbatim (they may already be obstore-native names). The caller drops
    the passthrough set with a warning when obstore rejects it.
    """
    aliases = _SCHEME_ALIASES.get(scheme, {})
    translated: dict = {}
    passthrough: dict = {}
    for key, value in (storage_options or {}).items():
        if key == "client_kwargs" and isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if nested_key in _CLIENT_KWARGS_ALIASES:
                    translated[_CLIENT_KWARGS_ALIASES[nested_key]] = nested_value
                else:
                    passthrough[f"client_kwargs.{nested_key}"] = nested_value
            continue
        if scheme in ("gs", "gcs") and key == "token":
            # gcsfs uses token="anon" for anonymous access; other token
            # forms (paths, dicts, google-auth objects) have no obstore
            # equivalent and fall through to the passthrough set.
            if value == "anon":
                translated["skip_signature"] = True
            else:
                passthrough[key] = value
            continue
        if key in aliases:
            translated[aliases[key]] = value
        else:
            passthrough[key] = value
    return translated, passthrough


#: Regions already resolved, keyed by bucket. A bucket does not move.
_BUCKET_REGIONS: dict[str, str] = {}

_REGION_TIMEOUT = 10.0


def _resolve_bucket_region(bucket: str) -> str | None:
    """The region AWS reports for *bucket*, or ``None`` if it does not say.

    obstore sends the request to ``us-east-1`` when no region is configured,
    and S3 answers for a bucket held elsewhere with a 301 that carries no
    ``Location``, so the read fails instead of being redirected. AWS names
    the region in a header on the bucket itself, and answers unsigned.
    """
    if bucket in _BUCKET_REGIONS:
        return _BUCKET_REGIONS[bucket]
    request = urllib.request.Request(
        f"https://{bucket}.s3.amazonaws.com/", method="HEAD"
    )
    try:
        with urllib.request.urlopen(request, timeout=_REGION_TIMEOUT) as response:
            region = response.headers.get("x-amz-bucket-region")
    except urllib.error.HTTPError as error:
        # A bucket that exists still names its region when it refuses the read.
        region = error.headers.get("x-amz-bucket-region")
    except OSError:
        return None
    if region:
        _BUCKET_REGIONS[bucket] = region
    return region


def _fill_bucket_region(url: str, translated: dict, passthrough: dict) -> None:
    """Add the bucket's region to *translated* when nothing else supplies one."""
    if "region" in translated or "region" in passthrough:
        return
    if "endpoint" in translated or "endpoint" in passthrough:
        # An endpoint names an S3-compatible service, not AWS.
        return
    if os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"):
        return
    bucket = urlsplit(url).netloc
    if not bucket:
        return
    region = _resolve_bucket_region(bucket)
    if region:
        translated["region"] = region


def _build_obstore(url: str, storage_options: dict | None):
    """Construct the obstore store for *url*, translating *storage_options*."""
    from obstore.store import from_url

    scheme = urlsplit(url).scheme.lower()
    translated, passthrough = _translate_storage_options(scheme, storage_options)
    if scheme in ("s3", "s3a"):
        _fill_bucket_region(url, translated, passthrough)

    def _construct(options: dict):
        if scheme in ("http", "https"):
            # HTTPStore accepts no bucket configuration; every option is a
            # client option (headers, timeouts, proxies, ...). A plain
            # http:// URL is an explicit opt-out of TLS, which obstore
            # otherwise rejects.
            if scheme == "http":
                options = {"allow_http": True, **options}
            return from_url(url, client_options=options or None)
        return from_url(url, config=options or None)

    try:
        return _construct({**translated, **passthrough})
    except Exception:
        if not passthrough:
            raise
    warnings.warn(
        "Ignoring storage_options with no obstore equivalent for "
        f"{scheme}:// stores: {sorted(passthrough)}. Remote reads go through "
        "obstore; see https://developmentseed.org/obstore/ for the supported "
        "configuration.",
        UserWarning,
        stacklevel=3,
    )
    return _construct(translated)


# -- store handle and node surface -------------------------------------------


class RemoteZarrStore:
    """Handle for a remote OME-Zarr store read via zarrista's async API.

    Wraps the obstore store for *url* plus an optional node *prefix*
    (HCS well/field views). ``str()`` is the full URL so shared error
    messages keep pointing at what the user passed in.
    """

    def __init__(self, url: str, storage_options: dict | None = None, prefix: str = ""):
        self.url = url.rstrip("/")
        self.storage_options = storage_options
        self.prefix = "/".join(
            part for part in str(prefix).split("/") if part not in ("", ".")
        )
        self._store = _build_obstore(self.url, storage_options)

    def with_prefix(self, prefix: str) -> "RemoteZarrStore":
        """A view of the same store narrowed to *prefix* (joined to any
        existing prefix). The underlying obstore client is shared."""
        view = object.__new__(RemoteZarrStore)
        view.url = self.url
        view.storage_options = self.storage_options
        view.prefix = posixpath.join(self.prefix, str(prefix).strip("/"))
        view._store = self._store
        return view

    def node_path(self, component: str | None = None) -> str:
        """The zarrista node path for *component* under this view's prefix."""
        parts = [
            part
            for source in (self.prefix, component)
            if source
            for part in str(source).split("/")
            if part not in ("", ".")
        ]
        return "/" + "/".join(parts)

    def __str__(self) -> str:
        if self.prefix:
            return f"{self.url}/{self.prefix}"
        return self.url


def _async_adapter(arr):
    """dask adapter over an ``AsyncArray``, fetching through the IO loop."""
    from ._zarrista_utils import _ZarristaArrayAdapter

    class _Adapter(_ZarristaArrayAdapter):
        def _fetch(self, selection) -> np.ndarray:
            # The pyo3 future must be created inside the running loop, so
            # the subscript itself happens in the factory.
            return np.asarray(_run(lambda: self._arr[selection]))

        def __setitem__(self, selection, value):
            raise TypeError("Remote zarr stores are read-only")

    return _Adapter(arr)


def _remote_array_to_dask(arr) -> dask.array.Array:
    """Wrap ``AsyncArray`` *arr* as a lazy dask array on its chunk grid.

    Mirrors the sync ``zarrista_array_to_dask``: sharded arrays chunk on
    the subchunk (inner chunk) shape so reads stay at the efficient
    granularity.
    """
    if arr.is_sharded:
        chunks = tuple(arr.subchunk_shape)
    else:
        chunks = tuple(arr.chunk_shape([0] * arr.ndim))
    return dask.array.from_array(_async_adapter(arr), chunks=chunks)


class RemoteZarrArray:
    """Read-only handle for an array node within a remote store.

    The read half of :class:`~ngff_zarr._zarrista_utils.LocalZarrArray`: the
    node's ``shape``, ``dtype``, ``chunks`` and ``attrs``, numpy style ``[]``
    reads, and :meth:`to_dask` for lazy pixel access. Writes are refused, a
    remote store being read-only here.
    """

    def __init__(self, store: RemoteZarrStore, component: str, arr):
        self._store = store
        self._arr = arr
        self.path = component
        self.shape = tuple(int(s) for s in arr.shape)
        self.ndim = arr.ndim
        self.attrs = AttrsDict(dict(arr.attrs))
        self._adapter = None

    def _array(self):
        """The synchronous adapter over the async node, made once."""
        if self._adapter is None:
            self._adapter = _async_adapter(self._arr)
        return self._adapter

    @property
    def dtype(self) -> np.dtype:
        return self._array().dtype

    @property
    def chunks(self) -> tuple[int, ...]:
        """The stored chunk grid, the inner chunk shape of a sharded array."""
        if self._arr.is_sharded:
            return tuple(int(c) for c in self._arr.subchunk_shape)
        return tuple(int(c) for c in self._arr.chunk_shape([0] * self._arr.ndim))

    def __getitem__(self, selection) -> np.ndarray:
        return self._array()[selection]

    def __setitem__(self, selection, value) -> None:
        raise TypeError("Remote zarr stores are read-only")

    def to_dask(self) -> dask.array.Array:
        return _remote_array_to_dask(self._arr)


class RemoteZarrGroup:
    """Read-only handle for a group node within a remote store.

    Provides the slice of the group surface the read path uses --
    ``attrs`` (a dict with ``asdict()``), ``__contains__``/``__getitem__``
    path navigation, and ``keys()`` -- backed by zarrista ``AsyncGroup``
    metadata reads. Child arrays open their pixel data lazily.
    """

    def __init__(self, store: RemoteZarrStore, component: str, group):
        self._store = store
        self._group = group
        self.path = component
        self.attrs = AttrsDict(dict(group.attrs))

    def _child_component(self, name) -> str:
        child = "/".join(p for p in str(name).split("/") if p not in ("", "."))
        return f"{self.path}/{child}" if self.path else child

    def __getitem__(self, name):
        node = open_remote_node(self._store, self._child_component(name))
        if node is None:
            raise KeyError(name)
        return node

    def __contains__(self, name) -> bool:
        try:
            self[name]
        except KeyError:
            return False
        return True

    def keys(self) -> list[str]:
        """Names of all immediate child nodes, sorted.

        Requires a store that supports listing (S3/GCS/Azure); plain HTTP
        servers generally do not, in which case this returns an empty list
        with a warning -- remote navigation should rely on metadata-declared
        paths instead.
        """
        try:
            child_paths = _run(self._group.child_paths)
        except Exception as error:
            warnings.warn(
                f"Listing children of the remote store '{self._store}' failed "
                f"({error}); the store may not support listing (plain HTTP).",
                UserWarning,
                stacklevel=2,
            )
            return []
        return sorted(posixpath.basename(path) for path in child_paths)


def _is_missing_metadata(error: Exception) -> bool:
    return _MISSING_METADATA_MARKER in str(error)


def _fetch_v2_group_attrs(store: RemoteZarrStore, component: str | None) -> dict | None:
    """The ``.zattrs`` of the zarr format 2 group at *component*, if one exists.

    Used when auto-detection opened a format 3 group with empty attributes:
    like the local read path, a spurious ``zarr.json`` sitting on top of a
    format 2 store must not hide the real ``.zgroup``/``.zattrs`` documents.
    Returns ``None`` when there is no format 2 group here.
    """
    import json

    key_prefix = store.node_path(component).strip("/")

    async def _get(name: str):
        key = f"{key_prefix}/{name}" if key_prefix else name
        result = await store._store.get_async(key)
        return bytes(await result.bytes_async())

    try:
        _run(lambda: _get(".zgroup"))
    except Exception:
        return None
    try:
        loaded = json.loads(_run(lambda: _get(".zattrs")))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def open_remote_node(store: RemoteZarrStore, component: str | None = None):
    """Open the node at *component* within remote *store*.

    Returns a :class:`RemoteZarrGroup` or :class:`RemoteZarrArray`, or
    ``None`` when neither node type exists there (missing metadata).
    zarrista auto-detects the zarr format, preferring format 3; transport
    and store errors propagate unchanged.
    """
    from zarrista import AsyncArray, AsyncGroup
    from zarrista.exceptions import ArrayCreateError, GroupCreateError

    path = store.node_path(component)
    normalized = path.strip("/")
    try:
        group = _run(lambda: AsyncGroup.open(store._store, path))
    except GroupCreateError as group_error:
        # Either the node is an array (missing or non-group metadata) or the
        # store itself is unreachable; the array attempt distinguishes them.
        try:
            arr = _run(lambda: AsyncArray.open(store._store, path))
        except ArrayCreateError as array_error:
            if _is_missing_metadata(group_error) and _is_missing_metadata(array_error):
                return None
            # Transport/store errors repeat identically for both node types;
            # propagate the group attempt's, which callers opened first.
            raise group_error from array_error
        return RemoteZarrArray(store, normalized, arr)
    node = RemoteZarrGroup(store, normalized, group)
    if not node.attrs:
        v2_attrs = _fetch_v2_group_attrs(store, component)
        if v2_attrs is not None:
            node.attrs = AttrsDict(v2_attrs)
    return node


def open_remote_lazy(
    store: RemoteZarrStore, component: str | None = None
) -> dask.array.Array:
    """Open the array at *component* within *store* as a lazy dask array."""
    from zarrista import AsyncArray

    path = store.node_path(component)
    arr = _run(lambda: AsyncArray.open(store._store, path))
    return _remote_array_to_dask(arr)
