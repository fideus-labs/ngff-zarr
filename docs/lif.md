---
orphan: true
---
<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🔬 Leica Image Format (LIF) Support

NGFF-Zarr provides comprehensive support for converting Leica Image Format (LIF)
files to OME-Zarr. LIF is a proprietary format used by Leica microscopy systems
to store multi-dimensional imaging data, including confocal, widefield, and
super-resolution microscopy images.

## Background

Leica Image Format (LIF) files are commonly produced by Leica microscopy
systems, including:

- Leica SP (confocal) series microscopes
- Leica THUNDER imaging systems
- Leica STELLARIS systems
- Leica DMi8 and other widefield systems

LIF files can contain:

- **Multiple series**: Several independent image datasets in a single file
- **Multi-dimensional data**: X, Y, Z (depth), T (time), C (channels), and M (mosaic/tiles)
- **Channel information**: Emission (λ), excitation (Λ), and RGB (S) wavelengths
- **Calibration data**: Pixel sizes, timestamps, and coordinate transformations
- **Mosaic/tile data**: Large area scans composed of multiple tiles

## Installation

To enable LIF support, install ngff-zarr with the CLI optional dependencies:

```shell
pip install 'ngff-zarr[cli]'
```

This installs the [liffile](https://pypi.org/project/liffile/) library for
reading LIF files.

## Supported File Extensions

- `.lif` - Leica Image Format
- `.lof` - Leica Object Format
- `.xlif` - Leica XML Image Format
- `.xlef` - Leica XML Experiment Format
- `.xlcf` - Leica XML Configuration Format

## Command Line Usage

### Convert a LIF file

Convert all series from a LIF file:

```shell
ngff-zarr -i microscopy_data.lif -o output/
```

Each series in the LIF file will be converted to a separate `.ome.zarr`
directory within the output folder.

### Convert a specific series by index

```shell
ngff-zarr -i microscopy_data.lif -o output.ome.zarr --series 0
```

### Convert series matching a pattern

Use glob patterns to select series by name:

```shell
# Convert all series containing "GFP"
ngff-zarr -i microscopy_data.lif -o output/ --series "*GFP*"

# Convert series starting with "Experiment"
ngff-zarr -i microscopy_data.lif -o output/ --series "Experiment*"
```

### Inspect LIF file contents

To see information about the series in a LIF file without converting:

```shell
ngff-zarr -i microscopy_data.lif
```

### Mosaic/Tile Data (HCS Output)

When a LIF series contains mosaic data (M dimension > 1), ngff-zarr
automatically converts it to an HCS (High Content Screening) plate structure:

```shell
ngff-zarr -i tile_scan.lif -o tile_scan.ome.zarr
```

The output will have the standard HCS plate/well/field hierarchy, making it
compatible with plate viewers and analysis tools.

## Python API

### Convert a single LIF image

```python
from liffile import LifFile
from ngff_zarr import lif_to_ngff_image, to_multiscales, to_ngff_zarr

# Open LIF file and convert first series
with LifFile("microscopy_data.lif") as lif:
    # Convert the first image to NgffImage
    ngff_image = lif_to_ngff_image(lif.images[0])

    print(f"Dimensions: {ngff_image.dims}")
    print(f"Shape: {ngff_image.data.shape}")
    print(f"Scale: {ngff_image.scale}")

    # Generate multiscales and write to OME-Zarr
    multiscales = to_multiscales(ngff_image, scale_factors=[2, 4])
    to_ngff_zarr("output.ome.zarr", multiscales)
```

### Convert multiple series with filtering

```python
from ngff_zarr import lif_file_to_ngff_images, to_multiscales, to_ngff_zarr

# Convert all series
images = lif_file_to_ngff_images("microscopy_data.lif")
for name, ngff_image in images:
    print(f"Series: {name}, Shape: {ngff_image.data.shape}")

# Convert specific series by index
images = lif_file_to_ngff_images("microscopy_data.lif", series=0)

# Convert series matching a pattern
images = lif_file_to_ngff_images("microscopy_data.lif", series="*GFP*")

# Process and write each series
for name, ngff_image in images:
    multiscales = to_multiscales(ngff_image, scale_factors=[2, 4])
    to_ngff_zarr(f"{name}.ome.zarr", multiscales)
```

### Convert mosaic data to HCS plate

```python
from liffile import LifFile
from ngff_zarr import (
    lif_to_ngff_image,
    lif_to_hcs_plate,
    has_mosaic_dimension,
    to_multiscales,
)
from ngff_zarr.hcs import HCSPlateWriter

with LifFile("tile_scan.lif") as lif:
    lif_image = lif.images[0]

    # Check if image has mosaic dimension
    if has_mosaic_dimension(lif_image):
        # Convert to HCS plate structure
        plate_metadata, well_images = lif_to_hcs_plate(lif_image, plate_name="TileScan")

        # Write as HCS plate
        with HCSPlateWriter("tile_scan.ome.zarr", plate_metadata) as writer:
            for row_name, column_name, field_index, field_image in well_images:
                multiscales = to_multiscales(field_image, scale_factors=[2, 4])
                writer.write_well_image(
                    multiscales=multiscales,
                    row_name=row_name,
                    column_name=column_name,
                    field_index=field_index,
                )
    else:
        # Standard image conversion
        ngff_image = lif_to_ngff_image(lif_image)
        multiscales = to_multiscales(ngff_image, scale_factors=[2, 4])
        to_ngff_zarr("output.ome.zarr", multiscales)
```

### Access metadata

```python
from liffile import LifFile
from ngff_zarr import lif_to_ngff_image

with LifFile("microscopy_data.lif") as lif:
    for i, lif_image in enumerate(lif.images):
        ngff_image = lif_to_ngff_image(lif_image)

        print(f"\nSeries {i}: {lif_image.path}")
        print(f"  Dimensions: {ngff_image.dims}")
        print(f"  Shape: {ngff_image.data.shape}")
        print(f"  Scale (pixel size): {ngff_image.scale}")
        print(f"  Translation (origin): {ngff_image.translation}")

        # Access timestamps if available
        if ngff_image.attrs and "timestamps" in ngff_image.attrs:
            print(f"  Timestamps: {ngff_image.attrs['timestamps']}")
```

## Dimension Mapping

LIF files use uppercase dimension names, which are mapped to OME-Zarr's
lowercase convention:

| LIF Dimension | OME-Zarr Dimension | Description |
|---------------|-------------------|-------------|
| X | x | Horizontal spatial axis |
| Y | y | Vertical spatial axis |
| Z | z | Depth/focus axis |
| T | t | Time axis |
| C | c | Channel axis |
| λ (lambda) | c | Emission wavelength (flattened to channel) |
| Λ (Lambda) | c | Excitation wavelength (flattened to channel) |
| S | c | RGB/spectral (flattened to channel) |
| M | m | Mosaic/tile index |

**Note:** Multiple channel-like dimensions (C, λ, Λ, S) are automatically
flattened into a single 'c' dimension for compatibility with OME-Zarr viewers.

## Scale and Translation

NGFF-Zarr automatically extracts calibration information from LIF files:

- **Scale**: Pixel sizes in physical units (typically micrometers)
- **Translation**: Coordinate offsets for the origin of each dimension

This metadata is preserved in the OME-Zarr output, ensuring accurate
visualization and measurement in compatible viewers.

## API Reference

### `lif_to_ngff_image(lif_image)`

Convert a single LifImage to an NgffImage.

**Parameters:**
- `lif_image`: A `liffile.LifImage` object

**Returns:**
- `NgffImage`: The converted image with proper dimensions, scale, and translation

### `lif_file_to_ngff_images(input_file, series=None)`

Convert a LIF file to a list of NgffImage objects.

**Parameters:**
- `input_file`: Path to the LIF file
- `series`: Optional series filter:
  - `None` or `"all"`: Convert all series
  - `int`: Convert only the series at this index
  - `str`: Convert series matching this glob pattern

**Returns:**
- `list[tuple[str, NgffImage]]`: List of (series_name, ngff_image) pairs

### `lif_to_hcs_plate(lif_image, plate_name=None)`

Convert a mosaic LIF image to an HCS plate structure.

**Parameters:**
- `lif_image`: A `liffile.LifImage` object with mosaic dimension (M > 1)
- `plate_name`: Optional name for the plate (defaults to image path)

**Returns:**
- `tuple[Plate, list]`: Plate metadata and list of (row, column, field_index, NgffImage) tuples

### `has_mosaic_dimension(lif_image)`

Check if a LIF image has a mosaic dimension with size > 1.

**Parameters:**
- `lif_image`: A `liffile.LifImage` object

**Returns:**
- `bool`: True if the image has M dimension > 1

## See Also

- [Command Line Interface](./cli.md)
- [High Content Screening (HCS) Support](./hcs.md)
- [Python API](./python.md)
- [liffile documentation](https://pypi.org/project/liffile/)
