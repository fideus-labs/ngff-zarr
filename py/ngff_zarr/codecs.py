# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Compression codec utilities for ngff-zarr.

Provides a shared interface for translating human-readable codec names
(e.g. ``"gzip"``, ``"blosc:zstd"``) into *numcodecs* compressor objects
that can be passed to :func:`~ngff_zarr.to_ngff_zarr` via its
``compressor`` keyword argument.

The same helpers are used by the CLI (``--codec``), the MCP server, and
can be called directly by library consumers.
"""

from __future__ import annotations

from typing import Any


def get_available_codecs() -> list[str]:
    """Return the list of supported compression codec names.

    Always includes ``"none"``, ``"gzip"``, ``"lz4"``, and ``"zstd"``.
    When *numcodecs* is installed the blosc family is appended:
    ``"blosc"``, ``"blosc:blosclz"``, ``"blosc:lz4"``, ``"blosc:lz4hc"``,
    ``"blosc:snappy"``, ``"blosc:zlib"``, ``"blosc:zstd"``.
    """
    codecs: list[str] = ["none", "gzip", "lz4", "zstd"]

    try:
        import numcodecs  # noqa: F401

        codecs.append("blosc")
        blosc_variants = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
        codecs.extend([f"blosc:{v}" for v in blosc_variants])
    except ImportError:
        pass

    return codecs


def codec_from_name(name: str, level: int | None = None) -> Any:
    """Convert a codec name string to a *numcodecs* compressor object.

    Parameters
    ----------
    name : str
        One of the names returned by :func:`get_available_codecs`.
        ``"none"`` explicitly disables compression (returns ``None``).
    level : int, optional
        Compression level.  When omitted a sensible default is used for
        the chosen codec (gzip=6, zstd=3, blosc=5).

    Returns
    -------
    compressor
        A *numcodecs* compressor instance, or ``None`` when *name* is
        ``"none"``.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    # Allow the "none" codec to work even when numcodecs is not installed.
    if name == "none":
        return None

    # All other codecs require numcodecs; provide a clear error if missing.
    try:
        import numcodecs
    except ImportError as exc:
        raise ImportError(
            f"Compression codec {name!r} requires the 'numcodecs' package, which "
            "is not installed. Install 'numcodecs' to use compressed codecs, or "
            "use 'none' to disable compression."
        ) from exc

    if name == "gzip":
        return numcodecs.GZip(level=level if level is not None else 6)

    if name == "lz4":
        return numcodecs.LZ4()

    if name == "zstd":
        return numcodecs.Zstd(level=level if level is not None else 3)

    if name == "blosc" or name.startswith("blosc:"):
        parts = name.split(":", 1)
        cname = parts[1] if len(parts) == 2 else "lz4"
        return numcodecs.Blosc(cname=cname, clevel=level if level is not None else 5)

    available = get_available_codecs()
    raise ValueError(
        f"Unknown codec: {name!r}. Available codecs: {', '.join(available)}"
    )
