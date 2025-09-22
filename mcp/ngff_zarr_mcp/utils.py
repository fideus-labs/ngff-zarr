# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Utility functions for ngff-zarr MCP server."""

import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
import httpx
import aiofiles

import zarr
from ngff_zarr import (  # type: ignore[import-untyped]
    from_ngff_zarr,
    config,
)

# Import unit validation function
try:
    from ngff_zarr.v04.zarr_metadata import is_unit_supported  # type: ignore[import-untyped]
except ImportError:
    # Fallback if not available
    def is_unit_supported(unit):
        common_units = [
            "meter",
            "millimeter",
            "micrometer",
            "nanometer",
            "second",
            "millisecond",
        ]
        return unit.lower() in common_units


from .models import StoreInfo, SupportedFormats


async def download_file(url: str, destination: Path) -> None:
    """Download a file from URL to destination."""
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async with aiofiles.open(destination, "wb") as f:
                async for chunk in response.aiter_bytes():
                    await f.write(chunk)


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_s3_url(path: str) -> bool:
    """Check if path is an S3 URL."""
    return path.startswith("s3://")


def is_zarr_store(path: str) -> bool:
    """Check if path points to a Zarr store."""
    if is_url(path):
        # For URLs, we'd need to check remotely - simplified for now
        return path.endswith(".zarr") or path.endswith(".ome.zarr")

    path_obj = Path(path)
    return (path_obj / ".zgroup").exists() or (path_obj / ".zarray").exists()


def get_store_size_and_files(store_path: str) -> tuple[int, int]:
    """Get total size and number of files in a store."""
    if is_url(store_path):
        # For remote stores, this would need special handling
        return 0, 0

    path = Path(store_path)
    if not path.exists():
        return 0, 0

    total_size = 0
    file_count = 0

    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
            file_count += 1

    return total_size, file_count


def analyze_zarr_store(store_path: str) -> StoreInfo:
    """Analyze a Zarr store and return information."""
    try:
        if is_url(store_path):
            # Handle remote stores
            if is_s3_url(store_path):
                try:
                    import s3fs  # type: ignore[import-not-found]

                    fs = s3fs.S3FileSystem()
                    store = s3fs.S3Map(store_path, s3=fs)
                except ImportError:
                    raise ImportError("s3fs required for S3 URLs")
            else:
                # HTTP/HTTPS URLs
                import fsspec  # type: ignore[import-untyped]

                store = fsspec.get_mapper(store_path)
        else:
            # For local files, just use the path directly
            store = store_path

        # Load multiscales
        try:
            multiscales = from_ngff_zarr(store)
        except ValueError as e:
            if "path" in str(e) and "FSMap" in str(e):
                # Handle zarr v3 FSMap issue by using zarr v2 for remote stores
                import zarr  # type: ignore[import-untyped]

                if store_path.startswith(("http://", "https://")):
                    try:
                        store_v2 = zarr.open(store, mode="r")
                        multiscales = from_ngff_zarr(store_v2.store)
                    except KeyError as ke:
                        if str(ke) == "'start'":
                            # Handle OMERO metadata compatibility issue for remote stores
                            raise ValueError(
                                "Remote store has incompatible OMERO metadata format: missing 'start' key in channel window. This is a known compatibility issue with some OME-Zarr stores that use 'min'/'max' instead of 'start'/'end'."
                            ) from ke
                        else:
                            raise ke
                else:
                    raise e
            else:
                raise e
        except KeyError as e:
            if str(e) == "'start'" and store_path.startswith(("http://", "https://")):
                # Handle OMERO metadata compatibility issue for remote stores
                # Some stores have 'min'/'max' instead of 'start'/'end' in channel window
                raise ValueError(
                    "Remote store has incompatible OMERO metadata format: missing 'start' key in channel window. This is a known compatibility issue with some OME-Zarr stores that use 'min'/'max' instead of 'start'/'end'."
                ) from e
            else:
                raise e
        first_image = multiscales.images[0]

        # Get store size and file count
        size_bytes, num_files = get_store_size_and_files(store_path)

        # Extract compression info if available
        compression = None
        if hasattr(first_image.data, "chunks"):
            # Get compression from first chunk
            try:
                chunk_info = first_image.data._meta
                if hasattr(chunk_info, "compressor") and chunk_info.compressor:
                    compression = str(chunk_info.compressor)
            except Exception:
                pass

        # Determine OME-Zarr version from metadata
        version = "0.4"  # Default
        try:
            import zarr as zarr_module  # Use a different name to avoid conflicts

            root = zarr_module.open(store, mode="r")

            # Check for v0.5 format first (ome.version)
            if "ome" in root.attrs:
                ome_attrs = root.attrs["ome"]
                if isinstance(ome_attrs, dict) and "version" in ome_attrs:
                    version_attr = ome_attrs["version"]
                    if isinstance(version_attr, str):
                        version = version_attr
            # Check for v0.4 format (multiscales[0].version)
            elif "multiscales" in root.attrs:
                multiscales_attr = root.attrs["multiscales"]
                if isinstance(multiscales_attr, list) and len(multiscales_attr) > 0:
                    ms_attrs = multiscales_attr[0]
                    if isinstance(ms_attrs, dict) and "version" in ms_attrs:
                        version_attr = ms_attrs["version"]
                        if isinstance(version_attr, str):
                            version = version_attr
        except Exception:
            pass

        return StoreInfo(
            path=store_path,
            version=version,
            size_bytes=size_bytes,
            num_files=num_files,
            num_scales=len(multiscales.images),
            dimensions=list(first_image.dims),
            shape=list(first_image.data.shape),
            dtype=str(first_image.data.dtype),
            chunks=(
                list(first_image.data.chunks[0])
                if hasattr(first_image.data, "chunks")
                else []
            ),
            compression=compression,
            scale_info=dict(first_image.scale) if first_image.scale else {},
            translation_info=(
                dict(first_image.translation) if first_image.translation else {}
            ),
            method_type=getattr(multiscales, "method", None),
            method_metadata=getattr(multiscales, "method_metadata", None),
            anatomical_orientation=None,  # Will be populated if available
            rfc_support=None,  # Will be populated if available
        )

    except Exception as e:
        raise ValueError(f"Failed to analyze store {store_path}: {str(e)}")


# Common file extensions supported by both ITK and ITKWasm
COMMON_EXTENSIONS = [
    ".bmp",
    ".dcm",
    ".gipl",
    ".gipl.gz",
    ".hdf5",
    ".jpg",
    ".jpeg",
    ".iwi",
    ".iwi.cbor",
    ".iwi.cbor.zst",
    ".lsm",
    ".mnc",
    ".mnc.gz",
    ".mnc2",
    ".mgh",
    ".mhz",
    ".mha",
    ".mhd",
    ".mrc",
    ".nia",
    ".nii",
    ".nii.gz",
    ".hdr",
    ".nrrd",
    ".nhdr",
    ".png",
    ".pic",
    ".vtk",
]


def get_supported_formats() -> SupportedFormats:
    """Get information about supported formats."""

    input_formats = {
        "ngff_zarr": [".zarr", ".ome.zarr"],
        "zarr": [".zarr"],  # Raw zarr arrays
        "itkwasm": COMMON_EXTENSIONS
        + [".aim", ".isq", ".fdf"],  # ITKWasm-specific extensions
        "itk": COMMON_EXTENSIONS + [],  # ITK-specific extensions
        "tifffile": [
            ".tif",
            ".tiff",  # Standard TIFF
            ".ome.tiff",  # OME-TIFF microscopy format
            ".stk",  # MetaMorph STK format
            ".gel",  # Molecular Dynamics GEL format
            ".seq",  # Media Cybernetics SEQ format
            ".zif",  # Zoomable Image File Format
            ".qptiff",  # PerkinElmer QPTIFF format
            ".bif",  # Roche BIF format
            ".avs",  # Argos AVS format
            ".dp",  # Philips DP format
            # Digital pathology formats
            ".svs",  # Aperio SVS format
            ".ndpi",  # Hamamatsu NDPI format
            ".scn",  # Leica SCN format
            # Note: .lsm is already covered by ITK/ITKWasm in COMMON_EXTENSIONS
        ],
        "imageio": [
            # Video formats (FFMPEG plugin)
            ".webm",
            ".mp4",
            ".avi",
            ".mov",
            ".gif",  # Already present
            ".mkv",
            ".mpeg",
            ".mpg",
            ".wmv",
            ".flv",
            ".3gp",
            ".3g2",
            ".m4v",
            ".f4v",
            ".m2ts",
            ".vob",
            ".ogv",
            ".dv",
            ".asf",
            ".rm",
            ".rmvb",
            ".ts",
            ".mts",
            ".divx",
            ".xvid",
            # Image formats not well-supported by ITK/tifffile
            ".psd",  # Photoshop Document
            ".pcx",  # ZSoft Paintbrush
            ".sgi",  # Silicon Graphics Image
            ".ras",  # Sun Raster
            ".xbm",
            ".xpm",  # X Window System
            ".dds",  # DirectDraw Surface
            ".cut",  # Dr. Halo
            ".exr",  # OpenEXR
            ".ppm",
            ".pgm",
            ".pbm",  # Portable Pixmap formats
            ".webp",  # WebP format
            ".icns",  # Apple Icon format
            ".ico",  # Windows Icon format
            ".tga",  # Targa format
            ".pict",  # Macintosh PICT
            ".pcd",  # Kodak PhotoCD
            # Camera raw formats (FreeImage plugin)
            ".3fr",
            ".arw",
            ".cr2",
            ".crw",
            ".nef",
            ".orf",
            ".raf",
            ".rw2",
            ".dng",
            ".raw",
            ".bay",
            ".cap",
            ".dcs",
            ".dcr",
            ".drf",
            ".erf",
            ".fff",
            ".iiq",
            ".k25",
            ".kdc",
            ".mdc",
            ".mef",
            ".mos",
            ".mrw",
            ".nrw",
            ".pef",
            ".ptx",
            ".pxn",
            ".r3d",
            ".rwl",
            ".sr2",
            ".srf",
            ".srw",
            ".x3f",
        ],
    }

    output_formats = [".zarr", ".ome.zarr"]

    backends = ["ngff_zarr", "zarr", "itkwasm", "itk", "tifffile", "imageio"]

    return SupportedFormats(
        input_formats=input_formats, output_formats=output_formats, backends=backends
    )


def get_available_methods() -> List[str]:
    """Get available downsampling methods."""
    from ngff_zarr.methods import methods_values  # type: ignore[import-untyped]

    return methods_values


def get_available_compression_codecs() -> List[str]:
    """Get available compression codecs."""
    codecs = ["gzip", "lz4", "zstd"]

    # Check for blosc variants
    try:
        import numcodecs  # type: ignore[import-untyped]  # noqa: F401

        blosc_codecs = ["blosc", "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
        codecs.extend([f"blosc:{codec}" for codec in blosc_codecs])
    except ImportError:
        pass

    return codecs


def setup_dask_config(
    use_local_cluster: bool = False,
    cache_dir: Optional[str] = None,
) -> Optional[Any]:
    """Setup Dask configuration and return client if using LocalCluster."""

    if cache_dir:
        cache_path = Path(cache_dir).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)
        if hasattr(zarr.storage, "DirectoryStore"):  # type: ignore[attr-defined]
            LocalStore = zarr.storage.DirectoryStore  # type: ignore[attr-defined]
        else:
            LocalStore = zarr.storage.LocalStore  # type: ignore[attr-defined]
        config.cache_store = LocalStore(cache_path)

    client = None
    if use_local_cluster:
        try:
            from dask.distributed import Client, LocalCluster
            import psutil

            n_workers = 4

            try:
                cpu_count = psutil.cpu_count(False)
                if cpu_count is not None:
                    n_workers = cpu_count // 2
            except Exception:
                pass

            cluster = LocalCluster(
                n_workers=n_workers,
                processes=True,
                threads_per_worker=2,
            )
            client = Client(cluster)
        except ImportError:
            raise ImportError("dask[distributed] required for LocalCluster")

    return client


def validate_conversion_options(options: Dict[str, Any]) -> List[str]:
    """Validate conversion options and return list of errors."""
    errors = []

    # Validate dimensions
    if "dims" in options and options["dims"]:
        valid_dims = {"t", "z", "y", "x", "c"}
        invalid_dims = set(options["dims"]) - valid_dims
        if invalid_dims:
            errors.append(
                f"Invalid dimensions: {invalid_dims}. Must be from {valid_dims}"
            )

    # Validate units
    if "units" in options and options["units"]:
        for dim, unit in options["units"].items():
            if not is_unit_supported(unit.lower()):
                errors.append(f"Unsupported unit '{unit}' for dimension '{dim}'")

    # Validate method
    if "method" in options:
        available_methods = get_available_methods()
        if options["method"] not in available_methods:
            errors.append(
                f"Invalid method '{options['method']}'. Available: {available_methods}"
            )

    # Validate OME-Zarr version
    if "ome_zarr_version" in options:
        if options["ome_zarr_version"] not in ["0.4", "0.5"]:
            errors.append("OME-Zarr version must be '0.4' or '0.5'")

    return errors


async def prepare_input_files(
    input_paths: List[str], temp_dir: Optional[str] = None
) -> List[str]:
    """Prepare input files, downloading remote files if necessary."""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    prepared_paths: list[str] = []

    for input_path in input_paths:
        if is_url(input_path) and not is_s3_url(input_path):
            # Download HTTP/HTTPS files
            filename = Path(urlparse(input_path).path).name
            if not filename:
                filename = f"downloaded_file_{len(prepared_paths)}"

            local_path = temp_path / filename
            await download_file(input_path, local_path)
            prepared_paths.append(str(local_path))
        else:
            # Local files or S3 URLs (handled by fsspec)
            prepared_paths.append(input_path)

    return prepared_paths
