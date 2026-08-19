#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT

if __name__ == "__main__" and __package__ is None:
    __package__ = "ngff_zarr"

import argparse
import atexit
import re
import signal
import sys
from pathlib import Path

import dask.utils
import zarr
import zarr.storage
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    MofNCompleteColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)
from rich.progress import (
    Progress as RichProgress,
)
from rich.spinner import Spinner
from rich_argparse import RichHelpFormatter

if hasattr(zarr.storage, "DirectoryStore"):
    LocalStore = zarr.storage.DirectoryStore
else:
    LocalStore = zarr.storage.LocalStore

from ._store_types import StoreLike
from .cli_input_to_ngff_image import cli_input_to_ngff_image
from .compute_omero import compute_omero_from_multiscales
from .config import config
from .detect_cli_io_backend import (
    ConversionBackend,
    conversion_backends_values,
    detect_cli_io_backend,
)
from .from_ngff_zarr import from_ome_zarr
from .methods import Methods, methods_values
from .ngff_image_to_itk_image import ngff_image_to_itk_image
from .rfc4 import (
    AnatomicalOrientation,
    AnatomicalOrientationValues,
    orientation_from_name,
)
from .rich_dask_progress import NgffProgress, NgffProgressCallback
from .to_multiscales import to_multiscales
from .to_ngff_zarr import to_ome_zarr
from .v04.zarr_metadata import Omero, OmeroChannel, OmeroWindow, is_unit_supported


class _ReplacingTextIO:
    """Text-stream proxy whose ``write`` never raises ``UnicodeEncodeError``.

    A last resort for streams that cannot be reconfigured in place (for
    example an ``ipykernel`` ``OutStream``, which has no ``reconfigure``).
    Characters the underlying stream's encoding cannot represent are replaced
    rather than raising, so Rich can render without crashing (see issue #37).
    All other stream behavior is delegated to the wrapped stream.
    """

    def __init__(self, stream):
        self._stream = stream
        self.encoding = getattr(stream, "encoding", None) or "utf-8"

    def write(self, text):
        try:
            return self._stream.write(text)
        except UnicodeEncodeError:
            safe = text.encode(self.encoding, errors="replace").decode(
                self.encoding, errors="replace"
            )
            return self._stream.write(safe)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        # Delegate everything else (isatty, fileno, close, ...) to the stream.
        return getattr(self._stream, name)


def _build_console() -> Console:
    """Create a Rich ``Console`` resilient to legacy Windows encodings.

    In some Windows environments the standard output stream uses a legacy
    code page such as ``cp1252`` -- for example a Jupyter/IPython ``!``
    shell-out, or git-bash. Rich emits Unicode box-drawing and spinner
    glyphs that those code pages cannot encode, so the underlying
    ``str.encode`` raises ``UnicodeEncodeError`` and the CLI crashes before
    doing any work (see issue #37).

    Reconfigure the stream to UTF-8 so the glyphs encode cleanly. If the
    stream cannot be reconfigured in place, bind the console to a proxy that
    replaces any un-encodable characters at write time, so rendering degrades
    gracefully instead of raising.
    """
    stream = sys.stdout
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    # ``startswith`` keeps capable UTF variants (utf-8-sig, utf-16-le, ...) as-is
    # while still handling legacy code pages (cp1252, latin-1, ascii, ...).
    if encoding.startswith(("utf8", "utf16", "utf32")):
        return Console()

    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8")
            return Console()
        except (ValueError, OSError):
            # Could not switch to UTF-8 (e.g. a detached/redirected stream);
            # keep the encoding but replace un-encodable characters in place.
            try:
                reconfigure(errors="replace")
                return Console()
            except (ValueError, OSError):
                pass

    # The stream cannot be reconfigured in place; wrap it so Rich's writes
    # never raise UnicodeEncodeError at render time.
    return Console(file=_ReplacingTextIO(stream))


def _apply_omero_metadata(live, args, multiscales):
    """Apply OMERO metadata to multiscales based on CLI arguments.

    If --no-omero is specified, skip OMERO computation.
    If --omero-window is specified, use manual values.
    Otherwise, compute OMERO metadata from the image data.
    """
    # Skip if --no-omero is specified
    if args.no_omero:
        return

    # Validate quantiles early
    if args.omero_quantiles:
        low, high = args.omero_quantiles
        if not (0.0 <= low <= 1.0):
            raise ValueError(
                f"Low quantile must be between 0 and 1, got {low}. "
                "Use --omero-quantiles LOW HIGH (e.g., --omero-quantiles 0.02 0.98)"
            )
        if not (0.0 <= high <= 1.0):
            raise ValueError(
                f"High quantile must be between 0 and 1, got {high}. "
                "Use --omero-quantiles LOW HIGH (e.g., --omero-quantiles 0.02 0.98)"
            )
        if low >= high:
            raise ValueError(
                f"Low quantile must be less than high quantile, got ({low}, {high}). "
                "Use --omero-quantiles LOW HIGH (e.g., --omero-quantiles 0.02 0.98)"
            )

    # Validate colors early if provided
    if args.omero_colors:
        for color in args.omero_colors:
            if not re.fullmatch(r"[0-9A-Fa-f]{6}", color):
                raise ValueError(
                    f"Color must be a 6-digit hexadecimal string without # prefix, got '{color}'. "
                    "Use --omero-colors RRGGBB RRGGBB (e.g., --omero-colors FF0000 00FF00)"
                )

    # If multiscales already has OMERO metadata and no manual override, keep it
    if multiscales.metadata.omero is not None and args.omero_window is None:
        return

    # If manual OMERO windows are specified, use them
    if args.omero_window is not None:
        channels = []
        n_windows = len(args.omero_window)

        # Get colors (default to white for single, glasbey for multiple)
        if args.omero_colors:
            # Validate we have enough colors
            if len(args.omero_colors) < n_windows:
                raise ValueError(
                    f"Not enough colors provided for {n_windows} windows. "
                    f"Got {len(args.omero_colors)} colors. "
                    "Provide at least one color per --omero-window."
                )
            colors = args.omero_colors
        else:
            from .compute_omero import get_default_colors

            colors = get_default_colors(n_windows)

        # Get labels (default to empty)
        if args.omero_labels:
            # Validate we have enough labels
            if len(args.omero_labels) < n_windows:
                raise ValueError(
                    f"Not enough labels provided for {n_windows} windows. "
                    f"Got {len(args.omero_labels)} labels. "
                    "Provide at least one label per --omero-window."
                )
            labels = args.omero_labels
        else:
            labels = [""] * n_windows

        for i, window_values in enumerate(args.omero_window):
            min_val, max_val, start_val, end_val = window_values
            window = OmeroWindow(min=min_val, max=max_val, start=start_val, end=end_val)
            color = colors[i]
            label = labels[i]
            channel = OmeroChannel(color=color, window=window, label=label)
            channels.append(channel)

        multiscales.metadata.omero = Omero(channels=channels)
        return

    # Compute OMERO metadata from image data
    if not args.quiet:
        live.console.log("[yellow]Computing OMERO visualization metadata...")

    quantiles = tuple(args.omero_quantiles)
    colors = args.omero_colors
    labels = args.omero_labels

    try:
        omero = compute_omero_from_multiscales(
            multiscales,
            quantiles=quantiles,
            colors=colors,
            labels=labels,
            dense=args.omero_dense,
        )
        multiscales.metadata.omero = omero
        if not args.quiet:
            live.console.log(
                f"[green]Computed OMERO metadata for {len(omero.channels)} channel(s)"
            )
    except Exception as e:
        if not args.quiet:
            live.console.log(f"[yellow]Warning: Could not compute OMERO metadata: {e}")


def _series_output_target(
    args: argparse.Namespace,
    output_store: StoreLike | None,
    series_name: str,
    series_index: int,
    n_series: int,
) -> tuple[StoreLike | None, str | None]:
    """Resolve the output store and display path for one series of a file.

    Returns a ``(store, display_path)`` tuple.

    With a single requested output path but several series in the source
    (every Aperio ``.svs`` whole-slide file has Baseline + Thumbnail + Label +
    Macro series, for example), the destination depends on the requested
    format:

    - For a single-file ``.ozx`` output, the primary (first) series is written
      to the exact requested path so the file the user asked for is actually
      created, and additional series are written alongside it as
      ``<base>_<series_name>.ozx`` (preserving the zip format). Previously the
      ``.ozx`` request was silently discarded and replaced with
      ``<base>_<series_name>.ome.zarr`` directories, so the requested ``.ozx``
      never appeared.
    - For directory-style ``.zarr`` / ``.ome.zarr`` output, each series is
      written to its own ``<base>_<series_name>.ome.zarr`` directory, the
      long-standing documented behavior.
    """
    if not args.output:
        return None, None
    if n_series <= 1:
        return output_store, args.output

    # ``.ozx`` detection is case-sensitive to match the rest of the codebase
    # (main()'s output_store selection, rfc9_zip.is_ozx_path, to_ngff_zarr).
    if args.output.endswith(".ozx"):
        if series_index == 0:
            return output_store, args.output
        base = args.output[: -len(".ozx")]
        derived = f"{base}_{series_name}.ozx"
        # to_ngff_zarr handles a string ``.ozx`` path directly.
        return derived, derived

    # Directory-style outputs: one .ome.zarr per series (documented behavior).
    # Strip the full ``.ome.zarr`` / ``.zarr`` suffix so that an ``out.ome.zarr``
    # base does not produce a doubled ``out.ome_<name>.ome.zarr``.
    if args.output.endswith(".ome.zarr"):
        base = args.output[: -len(".ome.zarr")]
    elif args.output.endswith(".zarr"):
        base = args.output[: -len(".zarr")]
    else:
        output_path = Path(args.output)
        base = str(output_path.parent / output_path.stem)
    derived = f"{base}_{series_name}.ome.zarr"
    return LocalStore(derived), derived


def _multiscales_to_ngff_zarr(
    live, args, output_store, rich_dask_progress, multiscales, chunks_per_shard=None
):
    # Apply OMERO metadata before displaying or writing
    _apply_omero_metadata(live, args, multiscales)

    if not args.output:
        if args.quiet:
            live.update(Pretty(multiscales))
        else:
            live.update(
                Panel(
                    Pretty(multiscales),
                    title="[red]NGFF OME-Zarr",
                    subtitle="[red]information",
                    style="magenta",
                )
            )
        return
    if args.use_tensorstore:
        live.console.print(
            "[yellow]Warning: --use-tensorstore is deprecated; "
            "using the zarrista backend."
        )
        # The zarrista writer requires a path-like store
        if hasattr(output_store, "root"):
            output_store = output_store.root
        else:
            output_store = output_store.path

    codec_kwargs = {}
    if args.compression_level is not None and args.codec is None:
        live.console.print(
            "[red]Error: --compression-level requires --codec to be specified."
        )
        sys.exit(1)
    if args.codec is not None:
        from .codecs import codec_from_name

        codec_kwargs["compressor"] = codec_from_name(args.codec, args.compression_level)

    to_ome_zarr(
        output_store,
        multiscales,
        chunks_per_shard=chunks_per_shard,
        progress=rich_dask_progress,
        use_tensorstore=args.use_tensorstore,
        version=args.ome_zarr_version,
        **codec_kwargs,
    )


def _apply_cli_metadata_overrides(ngff_image, args, live):
    """Apply user CLI metadata overrides (dims, scale, translation, units, name) to an NgffImage."""
    if args.dims:
        if len(args.dims) != len(ngff_image.dims):
            live.console.print(
                f"[red]Provided number of dims do not match expected: {len(ngff_image.dims)}"
            )
            sys.exit(1)
        ngff_image.dims = args.dims
    if args.scale:
        if len(args.scale) % 2 != 0:
            live.console.print(
                "[red]Provided scales are expected to be dim value pairs"
            )
            sys.exit(1)
        n_scale_args = len(args.scale) // 2
        for scale in range(n_scale_args):
            dim = args.scale[scale * 2]
            value = float(args.scale[scale * 2 + 1])
            ngff_image.scale[dim] = value
    if args.translation:
        if len(args.translation) % 2 != 0:
            live.console.print(
                "[red]Provided translations are expected to be dim value pairs"
            )
            sys.exit(1)
        n_translation_args = len(args.translation) // 2
        for translation in range(n_translation_args):
            dim = args.translation[translation * 2]
            value = float(args.translation[translation * 2 + 1])
            ngff_image.translation[dim] = value
    if args.units:
        if len(args.units) % 2 != 0:
            live.console.print(
                '[red]Provided units are expected to be dim value pairs, i.e. "x" "meter" ...'
            )
            sys.exit(1)
        unit_pairs = {
            str(args.units[unit * 2]).lower(): str(args.units[unit * 2 + 1]).lower()
            for unit in range(len(args.units) // 2)
        }
        unsupported_units = [
            value for value in unit_pairs.values() if not is_unit_supported(value)
        ]
        if any(unsupported_units):
            live.console.print(
                f"[red]The following unit(s) were requested but are not supported: {unsupported_units}"
            )
            sys.exit(1)
        ngff_image.axes_units = unit_pairs
    if args.name:
        ngff_image.name = args.name
    if args.orientation:
        ngff_image.axes_orientations = _parse_orientation_arg(args.orientation, live)


def _parse_orientation_arg(values, live):
    """Parse the CLI ``--orientation`` argument into an axes_orientations mapping.

    Accepts either a single preset name (``"LPS"``/``"RAS"``) or an even number
    of ordered dim/value pairs, e.g.
    ``x right-to-left y anterior-to-posterior z inferior-to-superior``.
    """
    if len(values) == 1:
        try:
            return orientation_from_name(values[0])
        except ValueError as exc:
            live.console.print(f"[red]{exc}")
            sys.exit(1)

    if len(values) % 2 != 0:
        live.console.print(
            "[red]--orientation expects either a preset name (LPS, RAS) or "
            "dim/value pairs, e.g. x right-to-left y anterior-to-posterior"
        )
        sys.exit(1)

    orientations: dict[str, AnatomicalOrientation] = {}
    for pair in range(len(values) // 2):
        dim = values[pair * 2]
        value = values[pair * 2 + 1]
        try:
            orientation_value = AnatomicalOrientationValues(value)
        except ValueError:
            supported = ", ".join(v.value for v in AnatomicalOrientationValues)
            live.console.print(
                f"[red]Unsupported anatomical orientation value: {value!r}.\n"
                f"[yellow]Supported values: {supported}"
            )
            sys.exit(1)
        orientations[dim] = AnatomicalOrientation(value=orientation_value)
    return orientations


def _ngff_image_to_multiscales(
    live, ngff_image, args, progress, rich_dask_progress, subtitle, method
):
    data = ngff_image.data
    _apply_cli_metadata_overrides(ngff_image, args, live)

    # Generate NgffMultiscales
    cache = data.nbytes > config.memory_target
    if not args.output:
        cache = False
    if not args.quiet:
        live.update(
            Panel(
                progress, title="[red]NGFF OME-Zarr", subtitle=subtitle, style="magenta"
            )
        )
    chunks = args.chunks
    if chunks is not None:
        chunks = chunks[0] if len(chunks) == 1 else tuple(chunks)
    return to_multiscales(
        ngff_image,
        method=method,
        progress=rich_dask_progress,
        chunks=chunks,
        cache=cache,
    )


def _convert_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ngff-zarr",
        description="Convert datasets to and from the OME-Zarr Next Generation File Format.",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", nargs="+", help="Input image(s)", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output image. If not specified just print information to stdout.",
    )

    metadata_group = parser.add_argument_group("metadata", "Specify output metadata")
    metadata_group.add_argument(
        "-d",
        "--dims",
        nargs="+",
        help='Ordered OME-Zarr NGFF dimensions from {"t", "z", "y", "x", "c"}',
        metavar="DIM",
    )
    metadata_group.add_argument(
        "-u",
        "--units",
        nargs="+",
        help="Ordered OME-Zarr NGFF axes spatial or temporal units",
        metavar="UNITS",
    )
    metadata_group.add_argument(
        "-s",
        "--scale",
        nargs="+",
        help="Override scale / spacing for each dimension, e.g. z 4.0 y 1.0 x 1.0",
        metavar="SCALE",
    )
    metadata_group.add_argument(
        "-t",
        "--translation",
        nargs="+",
        help="Override translation / origin for each dimension, e.g. z 0.0 y 50.0 x 40.0",
        metavar="TRANSLATION",
    )
    metadata_group.add_argument("-n", "--name", help="Image name")
    metadata_group.add_argument(
        "--output-scale",
        help="Scale to pick from multiscale input for a single-scale output format",
        type=int,
        default=0,
    )
    metadata_group.add_argument(
        "--ome-zarr-version",
        help="OME-Zarr version",
        default="0.5",
        choices=["0.4", "0.5"],
    )
    metadata_group.add_argument(
        "--orientation",
        nargs="+",
        help="Anatomical orientation (RFC 4) for the spatial axes. Either a preset "
        'coordinate system ("LPS" or "RAS") or ordered dim/value pairs, e.g. '
        '"x" "right-to-left" "y" "anterior-to-posterior" "z" "inferior-to-superior". '
        "Orientation is written to the output automatically when present.",
        metavar="ORIENTATION",
    )
    metadata_group.add_argument(
        "--series",
        help="Series to convert from multi-series files (e.g., LIF, TIFF). "
        "Can be: index (0, 1, 2), name pattern ('*area_1*'), or 'all' (default: all)",
        default=None,
    )

    # OMERO metadata options
    omero_group = parser.add_argument_group(
        "omero", "OMERO visualization metadata options"
    )
    omero_group.add_argument(
        "--no-omero",
        action="store_true",
        help="Disable automatic OMERO metadata computation",
    )
    omero_group.add_argument(
        "--omero-quantiles",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=[0.02, 0.98],
        help="Quantiles for OMERO window start/end (default: 0.02 0.98)",
    )
    omero_group.add_argument(
        "--omero-window",
        nargs=4,
        type=float,
        action="append",
        metavar=("MIN", "MAX", "START", "END"),
        help="Manually specify OMERO window for a channel. Can be used multiple times for multi-channel images.",
    )
    omero_group.add_argument(
        "--omero-colors",
        nargs="+",
        metavar="COLOR",
        help="Hex colors for channels without # prefix (e.g., FF0000 00FF00 0000FF)",
    )
    omero_group.add_argument(
        "--omero-labels",
        nargs="+",
        metavar="LABEL",
        help="Labels for channels (e.g., DAPI GFP RFP)",
    )
    omero_group.add_argument(
        "--omero-dense",
        action="store_true",
        help="Use histogram-based dense sampling for exact OMERO quantile computation. "
        "More accurate than the default approximate method for large datasets.",
    )

    processing_group = parser.add_argument_group("processing", "Processing options")
    processing_group.add_argument(
        "-c",
        "--chunks",
        nargs="+",
        type=int,
        help="Dask array chunking specification, either a single integer or integer per dimension, e.g. 64 or 8 16 32",
        metavar="CHUNKS",
    )
    processing_group.add_argument(
        "--chunks-per-shard",
        nargs="+",
        type=int,
        help="Number of chunks along each axis in a shard. If not set, no sharding. Either a single integer or integer per dimension, e.g. 4 or 2 4 8",
        metavar="CHUNKS_PER_SHARD",
    )
    processing_group.add_argument(
        "-m",
        "--method",
        default="itkwasm_gaussian",
        choices=methods_values,
        help="Downsampling method",
    )
    processing_group.add_argument(
        "-q", "--quiet", action="store_true", help="Do not display progress information"
    )
    processing_group.add_argument(
        "-l",
        "--local-cluster",
        action="store_true",
        help="Create a Dask Distributed LocalCluster. Better for large datasets.",
    )
    processing_group.add_argument(
        "--input-backend",
        choices=conversion_backends_values,
        help="Input conversion backend",
    )
    processing_group.add_argument("--memory-target", help="Memory limit, e.g. 4GB")
    processing_group.add_argument(
        "--cache-dir", help="Directory to use for caching with large datasets"
    )
    processing_group.add_argument(
        "--use-tensorstore",
        action="store_true",
        help="Deprecated: now uses the zarrista backend for I/O",
    )
    processing_group.add_argument(
        "--codec",
        help="Compression codec: blosc:zstd, blosc:lz4, gzip, zstd, lz4, none. "
        "Default: zarr's built-in default.",
        default=None,
    )
    processing_group.add_argument(
        "--compression-level",
        type=int,
        help="Compression level (codec-dependent, e.g. 1-9 for gzip, 1-22 for zstd)",
        default=None,
    )

    args = parser.parse_args(argv)

    _REMOTE_SCHEMES = ("s3://", "gs://", "az://", "azure://", "http://", "https://")

    def _is_remote(path_str: str) -> bool:
        lower = path_str.lower()
        return any(lower.startswith(scheme) for scheme in _REMOTE_SCHEMES)

    def _maybe_resolve(path_str: str) -> str:
        if _is_remote(path_str):
            return path_str
        return str(Path(path_str).resolve())

    # Check that input and output are not the same.
    # Resolve local paths to absolute paths for consistent handling across platforms.
    # Remote URLs (s3://, gs://, http://, etc.) are left unchanged.
    if args.output:
        output_resolved = _maybe_resolve(args.output)
        input_resolved = [_maybe_resolve(inp) for inp in args.input]
        if not _is_remote(output_resolved):
            output_path = Path(output_resolved)
            if any(
                not _is_remote(inp) and output_path == Path(inp)
                for inp in input_resolved
            ):
                parser.error("Input and output file/directory must not be the same.")

        # Use resolved paths for all subsequent operations
        args.output = output_resolved
        args.input = input_resolved

        # Set default OME-Zarr version to 0.5 for .ozx output files
        if args.output.endswith(".ozx") and args.ome_zarr_version == "0.4":
            args.ome_zarr_version = "0.5"
    else:
        # Resolve input paths even when no output is specified
        args.input = [_maybe_resolve(inp) for inp in args.input]

    if args.memory_target:
        config.memory_target = dask.utils.parse_bytes(args.memory_target)

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
        if not cache_dir.exists():
            Path.makedirs(cache_dir, parents=True)
        config.cache_store = cache_dir

    console = _build_console()
    progress = RichProgress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        *RichProgress.get_default_columns(),
        transient=False,
        console=console,
    )
    rich_dask_progress = None

    # Setup LocalCluster
    if args.local_cluster:
        from dask.distributed import Client, LocalCluster

        n_workers = 4
        worker_memory_target = config.memory_target // n_workers
        try:
            import psutil

            n_workers = psutil.cpu_count(False) // 2
            worker_memory_target = config.memory_target // n_workers
        except ImportError:
            pass

        cluster = LocalCluster(
            n_workers=n_workers,
            memory_limit=worker_memory_target,
            processes=True,
            threads_per_worker=2,
        )
        client = Client(cluster)

        def shutdown_client(sig_id, frame):
            client.shutdown()

        atexit.register(shutdown_client, None, None)
        signal.signal(signal.SIGTERM, shutdown_client)
        signal.signal(signal.SIGINT, shutdown_client)

        if not args.quiet:
            console.log(f"[yellow]Dashboard: [cyan]{client.dashboard_link}")

        if not args.quiet:
            rich_dask_progress = NgffProgress(progress)
    else:
        if not args.quiet:
            rich_dask_progress = NgffProgressCallback(progress)
            rich_dask_progress.register()

    # Parse conversion options
    if args.input_backend is None:
        input_backend = detect_cli_io_backend(args.input)
    else:
        input_backend = ConversionBackend(args.input_backend)
    method = Methods.ITKWASM_GAUSSIAN if args.method is None else Methods(args.method)

    if args.output:
        output_backend = detect_cli_io_backend(
            [
                args.output,
            ]
        )
    output_store = None
    if args.output and output_backend is ConversionBackend.NGFF_ZARR:
        # Handle .ozx files - just pass the path, to_ngff_zarr will handle it
        if args.output.endswith(".ozx"):
            output_store = args.output
        else:
            output_store = LocalStore(args.output)

    subtitle = "[red]generation"
    if not args.output:
        subtitle = "[red]information"
    initial = Panel(
        Spinner("point", text="Loading input..."),
        title="[red]NGFF OME-Zarr",
        subtitle=subtitle,
        style="magenta",
    )
    if args.quiet:
        initial = None
    with Live(initial, console=console) as live:
        chunks_per_shard = None
        if args.chunks_per_shard:
            if args.ome_zarr_version == "0.4":
                live.console.print(
                    "[red]Sharding is only supported for OME-Zarr version 0.5 and greater"
                )
                sys.exit(1)
            if len(args.chunks_per_shard) == 1:
                chunks_per_shard = args.chunks_per_shard[0]
            else:
                chunks_per_shard = tuple(args.chunks_per_shard)
        if args.output and output_backend is ConversionBackend.TIFFFILE:
            import numpy as np
            import tifffile

            ngff_image = cli_input_to_ngff_image(
                input_backend, args.input, args.output_scale
            )
            if isinstance(rich_dask_progress, NgffProgressCallback):
                rich_dask_progress.add_callback_task("[green]Converting to TIFF")

            from .methods._support import _channel_dim_last

            ngff_image = _channel_dim_last(ngff_image)

            data = np.asarray(ngff_image.data)

            # Map ngff dims to TIFF/OME axes string
            axes_map = {"t": "T", "c": "C", "z": "Z", "y": "Y", "x": "X"}
            axes = "".join(axes_map.get(d, d.upper()) for d in ngff_image.dims)

            # Determine photometric interpretation
            if "c" in ngff_image.dims:
                c_idx = ngff_image.dims.index("c")
                c_size = data.shape[c_idx]
                if c_size in (3, 4) and data.dtype == np.uint8:
                    photometric = "rgb"
                else:
                    photometric = "minisblack"
            else:
                photometric = "minisblack"

            # Build OME-TIFF resolution metadata
            metadata = {"axes": axes}
            resolution = None
            resolutionunit = None

            if "x" in ngff_image.scale and "y" in ngff_image.scale:
                x_scale = ngff_image.scale["x"]
                y_scale = ngff_image.scale["y"]
                if x_scale > 0 and y_scale > 0:
                    resolution = (1e4 / x_scale, 1e4 / y_scale)
                    resolutionunit = "CENTIMETER"
                    metadata["PhysicalSizeX"] = x_scale
                    metadata["PhysicalSizeXUnit"] = "µm"
                    metadata["PhysicalSizeY"] = y_scale
                    metadata["PhysicalSizeYUnit"] = "µm"

            if "z" in ngff_image.scale:
                z_scale = ngff_image.scale["z"]
                if z_scale > 0:
                    metadata["PhysicalSizeZ"] = z_scale
                    metadata["PhysicalSizeZUnit"] = "µm"

            tifffile.imwrite(
                args.output,
                data,
                photometric=photometric,
                metadata=metadata,
                resolution=resolution,
                resolutionunit=resolutionunit,
            )
            return

        if args.output and output_backend is ConversionBackend.ITKWASM:
            import itkwasm_image_io

            ngff_image = cli_input_to_ngff_image(
                input_backend, args.input, args.output_scale
            )
            if isinstance(rich_dask_progress, NgffProgressCallback):
                rich_dask_progress.add_callback_task("[green]Converting to image")
            itkwasm_image = ngff_image_to_itk_image(ngff_image, wasm=True)
            itkwasm_image_io.imwrite(itkwasm_image, args.output)
            return

        if args.output and output_backend is ConversionBackend.ITK:
            import itk

            ngff_image = cli_input_to_ngff_image(
                input_backend, args.input, args.output_scale
            )
            if isinstance(rich_dask_progress, NgffProgressCallback):
                rich_dask_progress.add_callback_task(
                    "[green]Converting Zarr Array to NumPy Array"
                )
            itk_image = ngff_image_to_itk_image(ngff_image, wasm=False)
            itk.imwrite(itk_image, args.output)
            return

        if args.output and output_backend is ConversionBackend.IMAGEIO:
            import imageio.v3 as iio
            import numpy as np

            ngff_image = cli_input_to_ngff_image(
                input_backend, args.input, args.output_scale
            )
            if isinstance(rich_dask_progress, NgffProgressCallback):
                rich_dask_progress.add_callback_task("[green]Converting to image")

            from .methods._support import _channel_dim_last

            ngff_image = _channel_dim_last(ngff_image)

            data = np.asarray(ngff_image.data)
            iio.imwrite(args.output, data)
            return

        if args.output and output_backend is ConversionBackend.NIBABEL:
            import nibabel as nib
            import numpy as np

            ngff_image = cli_input_to_ngff_image(
                input_backend, args.input, args.output_scale
            )
            if isinstance(rich_dask_progress, NgffProgressCallback):
                rich_dask_progress.add_callback_task("[green]Converting to NIfTI")

            data = np.asarray(ngff_image.data)

            # Build affine matrix from scale and translation
            affine = np.eye(4)
            for i, dim in enumerate(["x", "y", "z"]):
                if dim in ngff_image.scale:
                    affine[i, i] = ngff_image.scale[dim]
                if dim in ngff_image.translation:
                    affine[i, 3] = ngff_image.translation[dim]

            # Apply orientation sign from axes_orientations
            if ngff_image.axes_orientations:
                from .rfc4 import AnatomicalOrientationValues

                # In NIfTI RAS+ convention, positive direction is:
                #   x: left-to-right, y: posterior-to-anterior,
                #   z: inferior-to-superior
                # Flip the sign for the opposite direction.
                negative_orientations = {
                    "x": AnatomicalOrientationValues.right_to_left,
                    "y": AnatomicalOrientationValues.anterior_to_posterior,
                    "z": AnatomicalOrientationValues.superior_to_inferior,
                }
                for i, dim in enumerate(["x", "y", "z"]):
                    ornt = ngff_image.axes_orientations.get(dim)
                    if ornt is not None:
                        ornt_val = ornt.value if hasattr(ornt, "value") else ornt
                        if ornt_val == negative_orientations.get(dim):
                            affine[i, i] = -abs(affine[i, i])

            nii_img = nib.Nifti1Image(data, affine)
            nib.save(nii_img, args.output)
            return

        if args.output and output_backend in (
            ConversionBackend.LIFFILE,
            ConversionBackend.ZARR_ARRAY,
        ):
            live.console.print(
                f"[red]✗ {args.output}: unsupported output format {output_backend.name}"
            )
            live.console.print(
                "[yellow]→ Supported output formats:\n"
                "  • OME-Zarr: .zarr, .ome.zarr, .ozx\n"
                "  • NIfTI: .nii, .nii.gz\n"
                "  • TIFF: .tif, .tiff, .ome.tif, .lsm, .stk, .svs,"
                " .ndpi, etc.\n"
                "  • ITK: .png, .jpg, .bmp, .nrrd, .mha, .mhd, .dcm,"
                " .vtk, etc.\n"
                "  • Other: via imageio (format must be supported)"
            )
            sys.exit(1)

        if input_backend is ConversionBackend.NGFF_ZARR:
            # Pass the path directly to from_ome_zarr to let it handle .ozx files
            multiscales = from_ome_zarr(args.input[0])
            _multiscales_to_ngff_zarr(
                live,
                args,
                output_store,
                rich_dask_progress,
                multiscales,
                chunks_per_shard=chunks_per_shard,
            )
        elif input_backend is ConversionBackend.TIFFFILE:
            try:
                import importlib.util

                if importlib.util.find_spec("tifffile") is None:
                    raise ImportError("tifffile not found")

                from .multiscales import NgffMultiscales as MultiscalesType
                from .tiff_to_ngff_image import tiff_file_to_ngff_images

                # Get series to convert based on --series argument
                series_spec = args.series
                if series_spec is not None:
                    # Try to parse as integer
                    try:
                        series_spec = int(series_spec)
                    except ValueError:
                        pass  # Keep as string (pattern or "all")

                series_list = tiff_file_to_ngff_images(
                    args.input[0],
                    series=series_spec,
                    reuse_existing_pyramids=True,
                )

                if len(series_list) == 0:
                    live.console.print("[red]No matching series found in TIFF file.")
                    sys.exit(1)

                for series_index, (series_name, result) in enumerate(series_list):
                    # Determine output store/path for this series
                    series_store, series_path = _series_output_target(
                        args,
                        output_store,
                        series_name,
                        series_index,
                        len(series_list),
                    )

                    if isinstance(result, MultiscalesType):
                        # Pyramidal TIFF: reuse existing pyramid levels
                        multiscales = result
                        # Apply CLI metadata overrides to the base image
                        _apply_cli_metadata_overrides(multiscales.images[0], args, live)
                        if not args.quiet:
                            n_levels = len(multiscales.images)
                            live.console.log(
                                f"[green]Reusing {n_levels} existing pyramid levels"
                            )
                    else:
                        # Single-level: generate multiscales via downsampling
                        multiscales = _ngff_image_to_multiscales(
                            live,
                            result,
                            args,
                            progress,
                            rich_dask_progress,
                            subtitle,
                            method,
                        )
                    _multiscales_to_ngff_zarr(
                        live,
                        args,
                        series_store,
                        rich_dask_progress,
                        multiscales,
                        chunks_per_shard=chunks_per_shard,
                    )

                    if len(series_list) > 1 and args.output and not args.quiet:
                        live.console.print(
                            f"[green]Written series '{series_name}' -> {series_path}"
                        )

            except ImportError:
                sys.stdout.write("[red]Please install the [i]tifffile[/i] package.\n")
                sys.exit(1)
        elif input_backend is ConversionBackend.LIFFILE:
            try:
                from liffile import LifFile

                from .hcs import HCSPlateWriter
                from .lif_to_ngff_image import (
                    lif_file_to_ngff_images,
                    lif_to_hcs_plate,
                )

                with LifFile(args.input[0]) as lif:
                    # Get series to convert based on --series argument
                    series_spec = args.series
                    if series_spec is not None:
                        # Try to parse as integer
                        try:
                            series_spec = int(series_spec)
                        except ValueError:
                            pass  # Keep as string (pattern or "all")

                    series_list = lif_file_to_ngff_images(
                        args.input[0], series=series_spec
                    )

                    if len(series_list) == 0:
                        live.console.print("[red]No matching series found in LIF file.")
                        sys.exit(1)

                    for series_index, (series_name, ngff_image) in enumerate(
                        series_list
                    ):
                        # Check for mosaic dimension (M > 1) - requires original lif_image
                        # For now, check if 'm' dimension exists in ngff_image
                        has_mosaic = (
                            "m" in ngff_image.dims
                            and ngff_image.data.shape[list(ngff_image.dims).index("m")]
                            > 1
                        )

                        if has_mosaic and args.output:
                            # HCS output for mosaic data
                            # Find original LIF image to get mosaic info
                            lif_image = None
                            for img in lif.images:
                                if img.path.replace("/", "_") == series_name:
                                    lif_image = img
                                    break

                            if lif_image is not None:
                                plate, well_images = lif_to_hcs_plate(
                                    lif_image,
                                    plate_name=series_name,
                                    version=args.ome_zarr_version,
                                )

                                # Determine output path
                                if len(series_list) > 1:
                                    output_base = Path(args.output).stem
                                    output_path = (
                                        Path(args.output).parent
                                        / f"{output_base}_{series_name}.ome.zarr"
                                    )
                                else:
                                    output_path = Path(args.output)

                                # Write HCS plate
                                with HCSPlateWriter(
                                    str(output_path),
                                    plate,
                                    version=args.ome_zarr_version,
                                ) as writer:
                                    for (
                                        row_name,
                                        column_name,
                                        field_index,
                                        field_image,
                                    ) in well_images:
                                        # Generate multiscales for this field
                                        field_multiscales = _ngff_image_to_multiscales(
                                            live,
                                            field_image,
                                            args,
                                            progress,
                                            rich_dask_progress,
                                            subtitle,
                                            method,
                                        )
                                        writer.write_well_image(
                                            multiscales=field_multiscales,
                                            row_name=row_name,
                                            column_name=column_name,
                                            field_index=field_index,
                                        )
                                if not args.quiet:
                                    live.console.print(
                                        f"[green]Written HCS plate: {output_path}"
                                    )
                                continue

                        # Standard image output
                        # Determine output store/path for this series
                        series_store, series_path = _series_output_target(
                            args,
                            output_store,
                            series_name,
                            series_index,
                            len(series_list),
                        )

                        multiscales = _ngff_image_to_multiscales(
                            live,
                            ngff_image,
                            args,
                            progress,
                            rich_dask_progress,
                            subtitle,
                            method,
                        )
                        _multiscales_to_ngff_zarr(
                            live,
                            args,
                            series_store,
                            rich_dask_progress,
                            multiscales,
                            chunks_per_shard=chunks_per_shard,
                        )

                        if len(series_list) > 1 and args.output and not args.quiet:
                            live.console.print(
                                f"[green]Written series '{series_name}' -> {series_path}"
                            )

            except ImportError:
                sys.stdout.write("[red]Please install the [i]liffile[/i] package.\n")
                sys.exit(1)
        else:
            # Generate NgffImage
            ngff_image = cli_input_to_ngff_image(input_backend, args.input)
            multiscales = _ngff_image_to_multiscales(
                live, ngff_image, args, progress, rich_dask_progress, subtitle, method
            )
            _multiscales_to_ngff_zarr(
                live,
                args,
                output_store,
                rich_dask_progress,
                multiscales,
                chunks_per_shard=chunks_per_shard,
            )


def _upgrade_main(argv: list[str] | None = None) -> None:
    """Handle the ``ngff-zarr upgrade`` subcommand.

    Upgrades an OME-Zarr store to a different specification version. With no
    ``--output`` the upgrade is in-place and metadata-only -- every array chunk
    on disk is left untouched. With an ``--output`` an upgraded copy is written
    to a new store through the standard write pipeline, leaving the source
    intact.
    """
    parser = argparse.ArgumentParser(
        prog="ngff-zarr upgrade",
        description=(
            "Upgrade an OME-Zarr store to a different specification version. "
            "Omit --output for an in-place, metadata-only upgrade (array chunks "
            "are left untouched); pass --output to write an upgraded copy to a "
            "new store, leaving the source intact."
        ),
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path or URL to the source OME-Zarr store.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination store. If omitted, the store is upgraded in place "
        "(metadata-only; array chunks are left untouched).",
    )
    parser.add_argument(
        "--to",
        "--version",
        dest="version",
        choices=["0.4", "0.5", "0.6"],
        default="0.6",
        help="Target OME-Zarr version (default: 0.6).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the source metadata against the NGFF schema on read.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="When writing to a new --output store, overwrite existing data (default).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="When writing to a new --output store, do not overwrite existing data.",
    )
    parser.set_defaults(overwrite=True)

    args = parser.parse_args(argv)

    from .upgrade_ome_zarr import (
        _read_source_version,
        _stores_are_same,
        upgrade_ome_zarr,
    )

    console = _build_console()

    # Detect the on-disk source version for an informative summary. A detection
    # failure here is non-fatal: upgrade_ome_zarr re-reads the store below and
    # will surface a precise error, rendered friendlily, if the input is bad.
    try:
        source_version = _read_source_version(args.input, None)
    except Exception:
        source_version = None

    in_place = _stores_are_same(args.input, args.output)
    mode = "in-place metadata-only upgrade" if in_place else "write to a new store"

    if source_version is not None:
        console.print(f"[cyan]Detected source OME-Zarr version:[/] {source_version}")
    console.print(f"[cyan]Target OME-Zarr version:[/] {args.version}")
    console.print(f"[cyan]Mode:[/] {mode}")
    if not in_place:
        console.print(f"[cyan]Output store:[/] {args.output}")

    try:
        upgrade_ome_zarr(
            args.input,
            args.output,
            version=args.version,
            validate=args.validate,
            overwrite=args.overwrite,
        )
    except (ValueError, NotImplementedError) as exc:
        console.print(f"[red]✗ Upgrade failed:[/] {exc}")
        sys.exit(1)

    if in_place:
        console.print(
            f"[green]✓ Upgraded [/]{args.input}[green] in place to "
            f"OME-Zarr {args.version} (array chunks left untouched).[/]"
        )
    else:
        console.print(
            f"[green]✓ Wrote upgraded OME-Zarr {args.version} store to [/]{args.output}"
        )


def _conformance_main(argv: list[str] | None = None) -> None:
    """Handle the ``ngff-zarr conformance`` subcommand.

    Prints one canonical JSON RFC 4 verdict (rfc4_valid, axes, violations,
    warnings) per input, for a tool-agnostic conformance driver to diff against
    its manifest.
    """
    import json

    from .rfc4_conformance import conformance_report

    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("usage: ngff-zarr conformance <input>", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(conformance_report(argv[0]), indent=2))


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``ngff-zarr`` command.

    Dispatches to the ``upgrade`` or ``conformance`` subcommand when the first
    token matches; otherwise runs the flat conversion command unchanged so
    existing invocations behave exactly as before.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "upgrade":
        return _upgrade_main(argv[1:])
    if argv and argv[0] == "conformance":
        return _conformance_main(argv[1:])
    return _convert_main(argv)


if __name__ == "__main__":
    main()
