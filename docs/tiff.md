<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# TIFF and OME-TIFF Support

NGFF-Zarr provides comprehensive support for converting TIFF and OME-TIFF files
to OME-Zarr. When reading OME-TIFF files, physical size metadata is automatically
extracted and applied to the output.

## Background

TIFF (Tagged Image File Format) is one of the most common formats for storing
scientific and microscopy images. OME-TIFF extends the TIFF format with
standardized metadata in OME-XML format, enabling interoperability between
different microscopy software tools.

TIFF files can contain:

- **Multiple series**: Several independent image datasets in a single file
- **Multi-dimensional data**: X, Y, Z (depth), T (time), and C (channels)
- **Pyramid levels**: Multiple resolutions for large images
- **OME-XML metadata**: Physical sizes, channel information, and more

## Installation

To enable TIFF support, install ngff-zarr with the CLI optional dependencies:

```shell
pip install 'ngff-zarr[cli]'
```

This installs the [tifffile](https://pypi.org/project/tifffile/) library for
reading TIFF files.

## Supported File Extensions

- `.tiff`, `.tif` - Standard TIFF files
- `.ome.tiff`, `.ome.tif` - OME-TIFF files with embedded OME-XML metadata

## Command Line Usage

### Convert a TIFF file

```shell
ngff-zarr -i microscopy.tiff -o output.ome.zarr
```

### Convert all series from a multi-series TIFF

When converting a multi-series TIFF file, each series will be written to a
separate `.ome.zarr` directory:

```shell
ngff-zarr -i multi_series.ome.tiff -o output/
```

This creates one OME-Zarr store per series. For unnamed series, the default
filenames will be `output_series_0.ome.zarr`, `output_series_1.ome.zarr`, etc.
If the TIFF series have names (common for OME-TIFF), their names are used
instead of the index, for example `output_GFP.ome.zarr`, `output_DAPI.ome.zarr`,
and so on. Series names are automatically sanitized to ensure filesystem safety.

### Convert a specific series by index

```shell
ngff-zarr -i multi_series.ome.tiff -o output.ome.zarr --series 0
```

### Convert series matching a pattern

Use glob patterns to select series by name:

```shell
# Convert all series containing "GFP"
ngff-zarr -i multi_series.ome.tiff -o output/ --series "*GFP*"

# Convert series starting with "Channel"
ngff-zarr -i multi_series.ome.tiff -o output/ --series "Channel*"
```

### Inspect TIFF file contents

To see information about the series in a TIFF file without converting:

```shell
ngff-zarr -i microscopy.ome.tiff
```

## Python API

### Convert all series

```python
from ngff_zarr import tiff_file_to_ngff_images, to_multiscales, to_ngff_zarr

# Convert all series from a TIFF file
images = tiff_file_to_ngff_images("multi_series.ome.tiff")

for name, ngff_image in images:
    print(f"Series: {name}")
    print(f"  Shape: {ngff_image.data.shape}")
    print(f"  Dims: {ngff_image.dims}")
    print(f"  Scale: {ngff_image.scale}")
    print(f"  Units: {ngff_image.axes_units}")

    # Generate multiscales and write to OME-Zarr
    # Note: series names are automatically sanitized for filesystem safety
    multiscales = to_multiscales(ngff_image, scale_factors=[2, 4])
    to_ngff_zarr(f"{name}.ome.zarr", multiscales)
```

### Convert specific series by index

```python
from ngff_zarr import tiff_file_to_ngff_images

# Convert only series 0
images = tiff_file_to_ngff_images("multi_series.ome.tiff", series=0)
name, ngff_image = images[0]
print(f"Converted: {name}, shape: {ngff_image.data.shape}")
```

### Convert series matching a pattern

```python
from ngff_zarr import tiff_file_to_ngff_images

# Convert series matching a glob pattern
images = tiff_file_to_ngff_images("multi_series.ome.tiff", series="*GFP*")

for name, ngff_image in images:
    print(f"Matched series: {name}")
```

### Access OME metadata

When reading OME-TIFF files, physical size metadata is automatically extracted:

```python
from ngff_zarr import tiff_file_to_ngff_images

images = tiff_file_to_ngff_images("sample.ome.tiff")
name, ngff_image = images[0]

# Physical sizes from OME-XML
print(f"Scale (pixel sizes): {ngff_image.scale}")
# Example: {'x': 0.5, 'y': 0.5, 'z': 2.0}

# Units normalized to NGFF format
print(f"Units: {ngff_image.axes_units}")
# Example: {'x': 'micrometer', 'y': 'micrometer', 'z': 'micrometer'}
```

## OME Metadata Extraction

When reading OME-TIFF files, NGFF-Zarr automatically extracts and applies:

### Physical Sizes

The following OME-XML attributes are extracted:

| OME-XML Attribute | NGFF Scale Dimension |
|-------------------|---------------------|
| PhysicalSizeX | x |
| PhysicalSizeY | y |
| PhysicalSizeZ | z |

### Unit Mapping

OME unit symbols are normalized to NGFF-compatible (UDUNITS-2) unit names:

| OME Unit | NGFF Unit |
|----------|-----------|
| µm, um, micron | micrometer |
| nm | nanometer |
| mm | millimeter |
| cm | centimeter |
| m | meter |
| Å, A | angstrom |
| pm | picometer |

### Example OME-TIFF Metadata

For an OME-TIFF with this metadata:

```xml
<Pixels PhysicalSizeX="0.5" PhysicalSizeXUnit="µm"
        PhysicalSizeY="0.5" PhysicalSizeYUnit="µm"
        PhysicalSizeZ="2.0" PhysicalSizeZUnit="µm">
```

The resulting NgffImage will have:

```python
ngff_image.scale = {'x': 0.5, 'y': 0.5, 'z': 2.0}
ngff_image.axes_units = {'x': 'micrometer', 'y': 'micrometer', 'z': 'micrometer'}
```

## Pyramidal TIFFs

For pyramidal TIFFs (containing multiple resolution levels), NGFF-Zarr extracts
only the base resolution (level 0) from the source file. Use `to_multiscales()`
to regenerate the pyramid with consistent settings for your OME-Zarr output.

This approach ensures:

- Consistent downsampling method across all scales
- Optimal chunk sizes for the output format
- Compatibility with OME-Zarr viewers

## API Reference

### `tiff_file_to_ngff_images(tiff_path, series=None)`

Convert a TIFF file to a list of NgffImage objects.

**Parameters:**

- `tiff_path` (str or Path): Path to the TIFF file
- `series` (int, str, or list, optional): Series filter:
  - `None` or `"all"`: Convert all series (default)
  - `int`: Convert only the series at this index
  - `str`: Convert series matching this glob pattern (use `*` as wildcard)
  - `list`: Convert multiple series by index or pattern

**Returns:**

- `list[tuple[str, NgffImage]]`: List of (series_name, ngff_image) pairs

**Raises:**

- `ImportError`: If tifffile is not installed
- `IndexError`: If a specified series index is out of bounds
- `ValueError`: If the series specification is invalid

**Example:**

```python
from ngff_zarr import tiff_file_to_ngff_images

# All series
images = tiff_file_to_ngff_images("file.ome.tiff")

# Single series by index
images = tiff_file_to_ngff_images("file.ome.tiff", series=0)

# Series matching pattern
images = tiff_file_to_ngff_images("file.ome.tiff", series="*region_1*")

# Multiple specific series
images = tiff_file_to_ngff_images("file.ome.tiff", series=[0, 2, "*GFP*"])
```

## See Also

- [Command Line Interface](./cli.md)
- [Python API](./python.md)
- [Leica LIF Support](./lif.md)
- [tifffile documentation](https://pypi.org/project/tifffile/)
- [OME-TIFF specification](https://docs.openmicroscopy.org/ome-model/latest/ome-tiff/)
