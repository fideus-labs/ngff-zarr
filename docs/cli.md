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
[tifffile](https://pypi.org/project/tifffile/), or
[imageio](https://imageio.readthedocs.io/en/stable/formats/index.html).

Example:

```shell
ngff-zarr -i ./MR-head.nrrd -o ./MR-head.ome.zarr
```

![ngff-zarr convert](https://i.imgur.com/I7gTG52.png)

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

### More options

```shell
ngff-zarr --help
```
