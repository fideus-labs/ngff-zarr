<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 👨‍💻 Command Line Interface

`ngff-zarr` provides a command line interface to convert a variety of scientific
file formats to ome-zarr and inspect and ome-zarr store's contents.

## Installation

To install the command line interface (CLI):

```shell
pip install 'ngff-zarr[cli]'
```

## Usage

### Convert an image file

Convert any scientific image file format supported by either
[itk](https://wasm.itk.org/docs/image_formats),
[tifffile](https://pypi.org/project/tifffile/) (TIFF/OME-TIFF with metadata extraction),
[liffile](https://pypi.org/project/liffile/) (Leica LIF), or
[imageio](https://imageio.readthedocs.io/en/stable/formats/index.html).

Example:

```shell
ngff-zarr -i ./MR-head.nrrd -o ./MR-head.ome.zarr
```

![ngff-zarr convert](https://i.imgur.com/I7gTG52.png)

### Convert a TIFF file

TIFF files, including multi-series OME-TIFF files, can be converted with automatic
extraction of physical size metadata when available.

Convert a single TIFF file:

```shell
ngff-zarr -i microscopy.ome.tiff -o output.ome.zarr
```

Convert all series from a multi-series TIFF:

```shell
ngff-zarr -i multi_series.ome.tiff -o output/
```

Convert a specific series by index:

```shell
ngff-zarr -i multi_series.ome.tiff -o output.ome.zarr --series 0
```

Convert series matching a pattern:

```shell
ngff-zarr -i multi_series.ome.tiff -o output/ --series "*channel_1*"
```

When converting OME-TIFF files, physical size metadata (PhysicalSizeX/Y/Z and units)
is automatically extracted and applied to the output OME-Zarr.

For more details on TIFF conversion, see [TIFF Support](./tiff.md).

### Convert a Leica LIF file

Convert all series from a LIF file:

```shell
ngff-zarr -i microscopy.lif -o output/
```

Convert a specific series by index:

```shell
ngff-zarr -i microscopy.lif -o output.ome.zarr --series 0
```

Convert series matching a pattern:

```shell
ngff-zarr -i microscopy.lif -o output/ --series "*GFP*"
```

For more details on LIF conversion, see [Leica LIF Support](./lif.md).

### Convert an image volume slice series

Note the quotes:

```shell
ngff-zarr -i "series/*.tif" -o ome-ngff.ome.zarr
```

### Print information about generated multiscales

To print information about the input, omit the output argument.

```shell
ngff-zarr -i ./MR-head.nrrd
```

![ngff-zarr information](https://i.imgur.com/25RhzG2.png)

### Specify output chunks

```shell
ngff-zarr -c 64 -i ./MR-head.nrrd
```

![ngff-zarr output chunks](https://i.imgur.com/OGHyGQe.png)

### Specify metadata

```shell
ngff-zarr --dims "z" "y" "x" --scale x 1.4 y 1.4 z 2.5 --translation x 6.24 y 360.0 z 332.5 --name LIDC2 -i "series/*.tif"
```

![ngff-zarr metadata](https://i.imgur.com/AecFANr.png)

### Limit memory consumption

Limit memory consumption by passing a rough memory limit in human-readable
units, e.g. _8GB_ with the `--memory-target` option.

```shell
ngff-zarr --memory-target 50M -i ./LIDCFull.vtk -o ./LIDCFull.ome.zarr
```

![ngff-zarr memory-target](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmQ2NzVmMzU0NDA5ZDcyNzczNTU3MWE2YjczZjY5YmJkNWE4OTRhZSZjdD1n/ODobGeUYQr9wrE9J2s/giphy.gif)

### OME-Zarr Zip (.ozx) files

[RFC-9](https://ngff.openmicroscopy.org/rfc/9/index.html) introduces support for OME-Zarr Zip (.ozx) files, which package an entire OME-Zarr hierarchy into a single ZIP archive. This format enables:

- **Single-file distribution**: Share complete multiscale datasets as one file
- **Version metadata**: Embedded NGFF version in ZIP comment

#### Write .ozx files

```shell
ngff-zarr -i MR-head.nrrd -o MR-head.ozx
```

#### Inspect .ozx files

```shell
ngff-zarr -i ./MR-head.ozx
```

### High Content Screening (HCS) plate data

HCS plates contain multiple wells and images. To convert a specific well and field from an HCS plate, use sub-path syntax:

```shell
# Convert well A/1, field 0
ngff-zarr -i plate.ome.zarr/A/1/0 -o well_A1_field0.ome.zarr

# Convert well B/2, field 1
ngff-zarr -i plate.ome.zarr/B/2/1 -o well_B2_field1.ome.zarr
```

Attempting to convert an entire plate will display available wells:

```shell
$ ngff-zarr -i plate.ome.zarr -o output.ome.zarr
Error: The input appears to be an HCS plate structure with 96 wells.
Provide the full path: 'plate.ome.zarr/A/1/0', 'plate.ome.zarr/A/2/0', ...
```

For more details on HCS support, see [HCS Support](./hcs.md).

### More options

```shell
ngff-zarr --help
```
