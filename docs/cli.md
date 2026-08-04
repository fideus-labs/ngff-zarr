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

### Upgrade an OME-Zarr store to a new version

The `ngff-zarr upgrade` subcommand changes the OME-Zarr specification version
recorded in an existing store. It has two modes.

Omitting `-o`/`--output` performs an **in-place, metadata-only** upgrade: only
the store's group metadata is rewritten and every array chunk on disk is left
untouched.

```shell
ngff-zarr upgrade path/to/image.zarr --to 0.6
```

Passing `-o`/`--output` writes an **upgraded copy** to a new store through the
standard write pipeline, leaving the source store intact. This is the mode to
use for a transition across the Zarr v2/v3 boundary that the in-place mode
cannot do safely (for example a 0.5/0.6 downgrade back to 0.4).

```shell
ngff-zarr upgrade src.zarr -o dst.zarr --to 0.5
```

The target version is selected with `--to` (alias `--version`), one of `0.4`,
`0.5`, or `0.6` (default `0.6`). Add `--validate` to validate the source
metadata against the NGFF schema while reading. For the write-to-new-store mode,
`--overwrite` (the default) replaces any pre-existing data at the output store,
while `--no-overwrite` refuses to; both flags are ignored for an in-place
upgrade, which never overwrites array data.

```shell
# Validate the source, and refuse to overwrite an existing destination store.
ngff-zarr upgrade src.zarr -o dst.zarr --to 0.6 --validate --no-overwrite
```

The command prints the detected source version, the target version, and which
mode it ran. `ngff-zarr upgrade --help` lists every option.

### Check RFC-4 anatomical orientation conformance

The `ngff-zarr conformance` subcommand reads one store and prints a single JSON
verdict on its [RFC-4](./rfc4.md) anatomical orientation metadata:

```shell
ngff-zarr conformance path/to/image.ome.zarr
```

```json
{
  "input": "path/to/image.ome.zarr",
  "format": "ome-zarr",
  "rfc4_valid": true,
  "axes": {
    "z": "inferior-to-superior",
    "y": "anterior-to-posterior",
    "x": "right-to-left"
  },
  "violations": [],
  "warnings": []
}
```

Directory stores and `.zip` archives are both accepted; `.ozx` archives are not
supported by this subcommand. The output is the canonical contract used by the
[RFC-4 conformance suite](https://github.com/fideus-labs/ome-zarr-rfc4-validation),
so the suite's driver can be pointed straight at this command. The
[RFC-4 documentation](./rfc4.md) lists the violation and warning codes.

### More options

```shell
ngff-zarr --help
```
