# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

from dask.array.image import imread as daimread
from rich import print

from ._zarrista_utils import open_local_node
from .detect_cli_io_backend import ConversionBackend
from .from_ngff_zarr import from_ome_zarr
from .itk_image_to_ngff_image import itk_image_to_ngff_image
from .ngff_image import NgffImage
from .nibabel_image_to_ngff_image import nibabel_image_to_ngff_image
from .to_ngff_image import to_ngff_image


def cli_input_to_ngff_image(
    backend: ConversionBackend, input, output_scale: int = 0
) -> NgffImage:
    if backend is ConversionBackend.NGFF_ZARR:
        multiscales = from_ome_zarr(input[0])
        return multiscales.images[output_scale]
    if backend is ConversionBackend.ZARR_ARRAY:
        # Backend detection fires on a .zarray document, so this is a local
        # zarr format 2 array directory: read it through the compat layer.
        arr = open_local_node(input[0], zarr_format=2)
        return to_ngff_image(arr)
    if backend is ConversionBackend.NIBABEL:
        try:
            import nibabel as nib
        except ImportError:
            print("[red]Please install the [i]nibabel[/i] package.")
            sys.exit(1)
        image = nib.load(input[0])
        return nibabel_image_to_ngff_image(image)
    if backend is ConversionBackend.ITKWASM:
        try:
            import itkwasm_image_io
        except ImportError:
            print("[red]Please install the [i]itkwasm-image-io[/i] package.")
            sys.exit(1)
        # This will fail on windows systems if Path is not used and string is passed as input.
        image = itkwasm_image_io.imread(Path(input[0]))
        return itk_image_to_ngff_image(image)
    if backend is ConversionBackend.ITK:
        try:
            import itk
        except ImportError:
            print("[red]Please install the [i]itk-io[/i] package.")
            sys.exit(1)
        if len(input) == 1:
            if "*" in str(input[0]):

                def imread(filename):
                    image = itk.imread(filename)
                    return itk.array_from_image(image)

                da = daimread(str(input[0]), imread=imread)
                return to_ngff_image(da)
            image = itk.imread(input[0])
            return itk_image_to_ngff_image(image)
        image = itk.imread(input)
        return itk_image_to_ngff_image(image)
    if backend is ConversionBackend.LIFFILE:
        try:
            from liffile import LifFile
        except ImportError:
            print("[red]Please install the [i]liffile[/i] package.")
            sys.exit(1)
        from .lif_to_ngff_image import lif_to_ngff_image

        with LifFile(input[0]) as lif:
            # Default to first series for simple cli_input_to_ngff_image usage
            # Multi-series handling is done in cli.py
            if len(lif.images) == 0:
                print("[red]No images found in LIF file.")
                sys.exit(1)
            return lif_to_ngff_image(lif.images[0])
    if backend is ConversionBackend.TIFFFILE:
        try:
            import tifffile
        except ImportError:
            print("[red]Please install the [i]tifffile[/i] package.")
            sys.exit(1)
        try:
            # tifffile's aszarr stores import zarr-python; build the lazy
            # view from TIFF page primitives instead so the minimal install
            # can convert TIFFs.
            from .tiff_to_ngff_image import _series_level_to_dask

            if len(input) == 1:
                tif = tifffile.TiffFile(input[0])
                data = _series_level_to_dask(tif.series[0])
            else:
                # Stack each file's first series along a new leading axis,
                # mirroring tifffile's file-sequence stores.
                import dask.array as da

                data = da.stack(
                    [
                        _series_level_to_dask(tifffile.TiffFile(path).series[0])
                        for path in input
                    ]
                )
            return to_ngff_image(data)
        except (OSError, ValueError, tifffile.TiffFileError) as e:
            path_repr = input[0] if len(input) == 1 else input
            if isinstance(e, (FileNotFoundError, PermissionError, IsADirectoryError)):
                # Clear path/access error; don't suggest corruption.
                print(
                    f"[red]Error: Could not read TIFF file {str(path_repr)!r}. "
                    f"Details: {type(e).__name__}: {e}"
                )
            else:
                print(
                    f"[red]Error: Failed to read TIFF file {str(path_repr)!r}. "
                    f"The file may be corrupted, truncated, or have invalid page "
                    f"offsets. Details: {type(e).__name__}: {e}"
                )
            sys.exit(1)
    if backend is ConversionBackend.IMAGEIO:
        try:
            import imageio.v3 as iio
        except ImportError:
            print("[red]Please install the [i]imageio[/i] package.")
            sys.exit(1)

        image = iio.imread(str(input[0]))

        ngff_image = to_ngff_image(image)

        props = iio.improps(str(input[0]))
        if props.spacing is not None:
            # Only apply spacing to spatial dimensions (x, y, z)
            spatial_dims = [d for d in ngff_image.dims if d in {"x", "y", "z"}]
            if len(props.spacing) == 1:
                scale = dict.fromkeys(spatial_dims, props.spacing[0])
            else:
                # Match spacing values to spatial dims
                # Spacing from imageio may have fewer values than spatial dims
                n_spatial = min(len(props.spacing), len(spatial_dims))
                scale = {spatial_dims[i]: props.spacing[i] for i in range(n_spatial)}
            ngff_image.scale = scale

        return ngff_image
    return None
