<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🐍 Python Interface

NGFF-Zarr is a Python library that provides a simple, natural interface for
working with OME-Zarr data structures, creating chunked, multiscale OME-Zarr
image pyramids, and reading and writing OME-Zarr multiscale image files.

NGFF-Zarr's interface, which reflects the OME-Zarr data model, is built on
Python's built-in [dataclasses] and [Dask arrays]. It is designed to be simple,
flexible, and easy to use.

## Array to NGFF Image

NGFF-Zarr supports conversion of any NumPy array-like object that follows the
[Python Array API Standard] into the OME-Zarr data model. This includes such
objects an NumPy `ndarray`'s, Dask Arrays, PyTorch Tensors, CuPy arrays, Zarr
array, etc.

Convert the array to an [`NgffImage`], which is a standard Python [dataclass]
that represents an OME-Zarr image for a single scale.

When creating the image from the array, you can specify

- names of the `dims` from `{‘t’, ‘z’, ‘y’, ‘x’, ‘c’}`
- the `scale`, the pixel spacing for the spatial dims
- the `translation`, the origin or offset of the center of the first pixel
- a `name` for the image
- and `axes_units` with [UDUNITS-2 identifiers]

```python
>>> # Load an image as a NumPy array
>>> from imageio.v3 import imread
>>> data = imread('cthead1.png')
>>> print(type(data))
<class 'numpy.ndarray'>
```

Specify optional additional metadata with [`to_ngff_image`].

```python
>>> import ngff_zarr as nz
>>> image = nz.to_ngff_image(data,
                             dims=['y', 'x'],
                             scale={'y': 1.0, 'x': 1.0},
                             translation={'y': 0.0, 'x': 0.0})
>>> print(image)
NgffImage(
    data=dask.array<array, shape=(256, 256),
    dtype=uint8,
chunksize=(256, 256), chunktype=numpy.ndarray>,
    dims=['y', 'x'],
    scale={'y': 1.0, 'x': 1.0},
    translation={'y': 0.0, 'x': 0.0},
    name='image',
    axes_units=None,
    computed_callbacks=[]
)
```

The image data is nested in a lazy `dask.Array` and chucked.

If `dims`, `scale`, or `translation` are not specified, NumPy-compatible
defaults are used.

## Generate Multiscales

OME-Zarr represents images in a chunked, multiscale data structure. Use
[`to_multiscales`] to build a task graph that will produce a chunked, multiscale
image pyramid. [`to_multiscales`] has optional `scale_factors` and `chunks`
parameters. An [antialiasing method](./methods.md) can also be prescribed.

```python
>>> multiscales = nz.to_multiscales(image,
                                    scale_factors=[2,4],
                                    chunks=64)
>>> print(multiscales)
Multiscales(
    images=[
        NgffImage(
            data=dask.array<rechunk-merge, shape=(256, 256), dtype=uint8,chunksize=(64, 64), chunktype=numpy.ndarray>,
            dims=['y', 'x'],
            scale={'y': 1.0, 'x': 1.0},
            translation={'y': 0.0, 'x': 0.0},
            name='image',
            axes_units=None,
            computed_callbacks=[]
        ),
        NgffImage(
            data=dask.array<rechunk-merge, shape=(128, 128), dtype=uint8,
chunksize=(64, 64), chunktype=numpy.ndarray>,
            dims=['y', 'x'],
            scale={'x': 2.0, 'y': 2.0},
            translation={'x': 0.5, 'y': 0.5},
            name='image',
            axes_units=None,
            computed_callbacks=[]
        ),
        NgffImage(
            data=dask.array<rechunk-merge, shape=(64, 64), dtype=uint8,
chunksize=(64, 64), chunktype=numpy.ndarray>,
            dims=['y', 'x'],
            scale={'x': 4.0, 'y': 4.0},
            translation={'x': 1.5, 'y': 1.5},
            name='image',
            axes_units=None,
            computed_callbacks=[]
        )
    ],
    metadata=Metadata(
        axes=[
            Axis(name='y', type='space', unit=None),
            Axis(name='x', type='space', unit=None)
        ],
        datasets=[
            Dataset(
                path='scale0/image',
                coordinateTransformations=[
                    Scale(scale=[1.0, 1.0], type='scale'),
                    Translation(
                        translation=[0.0, 0.0],
                        type='translation'
                    )
                ]
            ),
            Dataset(
                path='scale1/image',
                coordinateTransformations=[
                    Scale(scale=[2.0, 2.0], type='scale'),
                    Translation(
                        translation=[0.5, 0.5],
                        type='translation'
                    )
                ]
            ),
            Dataset(
                path='scale2/image',
                coordinateTransformations=[
                    Scale(scale=[4.0, 4.0], type='scale'),
                    Translation(
                        translation=[1.5, 1.5],
                        type='translation'
                    )
                ]
            )
        ],
        coordinateTransformations=None,
        name='image',
        version='0.4'
    ),
    scale_factors=[2, 4],
    method=<Methods.ITKWASM_GAUSSIAN: 'itkwasm_gaussian'>,
    chunks={'y': 64, 'x': 64}
)
```

The [`Multiscales`] dataclass stores all the images and their metadata for each
scale according the OME-Zarr data model. Note that the correct `scale` and
`translation` for each scale are automatically computed.

## Read an OME-Zarr

To read an OME-Zarr file, use [`from_ngff_zarr`], which returns the
[`Multiscales`] dataclass.

```python
>>> multiscales = nz.from_ngff_zarr('cthead1.ome.zarr')
```

OME-Zarr version 0.1 to 0.5 is supported.

## Validate OME-Zarr metadata

To validate that an OME-Zarr's metadata following the specification's data
model, which is used by all the programming languages in the community, use the
`validate` optional dependency and kwarg to [`from_ngff_zarr`].

```shell
pip install "ngff-zarr[validate]"
```

```python
>>> multiscales = nz.from_ngff_zarr('cthead1.ome.zarr', validate=True)
```

If the metadata does not follow the data model, an error will be raised.

Metadata validation is supported for OME-Zarr version 0.1 to 0.5.

## Write an OME-Zarr

To write the multiscales to OME-Zarr, use [`to_ngff_zarr`].

```python
nz.to_ngff_zarr('cthead1.ome.zarr', multiscales)
```

Use the `.ome.zarr` extension for local directory stores by convention.

Any other
[Zarr store type](https://zarr.readthedocs.io/en/stable/api/storage.html) can
also be used.

The multiscales will be computed and written out-of-core, limiting memory usage.

### Writing with Tensorstore

To write with [tensorstore], which may provide better performance, use the
`tensorstore` optional dependency.

```shell
pip install "ngff-zarr[tensorstore]"
```

```python
nz.to_ngff_zarr('cthead1.ome.zarr', multiscales, use_tensorstore=True)
```

### Write a sharded OME-Zarr store

[Sharded Zarr] stores save multiple compressed chunks in a single file or blob.
This can be useful for large datasets, as it can reduce the number of files in a
directory.

To generate a sharded OME-Zarr store, pass the `chunks_per_shard` kwarg to
`to_ngff_zarr`. Sharding requires OME-Zarr version 0.5, which uses the Zarr
Format Specification 3.

This can be a single integer,

```python
version = '0.5'
nz.to_ngff_zarr('lightsheet.ome.zarr',
                multiscales,
                chunks_per_shard=2,
                version=version)
```

This will use 2 chunks per shard for all dimensions.

Or, specify a tuple of integers for each dimension.

```python
nz.to_ngff_zarr('lightsheet.ome.zarr',
                multiscales,
                chunks_per_shard=(2, 2, 4),
                version=version)
```

Or, specify a dictionary of integers for each dimension.

```python
nz.to_ngff_zarr('lightsheet.ome.zarr',
                multiscales,
                chunks_per_shard={'z':4, 'y':2, 'x':2},
                version=version)
```

The resulting shard shape will be the product of the chunk shape and the
`chunks_per_shard` shape. In this case the shard shape will be `(256, 128, 128)`
for a chunk shape of `(64, 64, 64)`.

Tensorstore can also be used with sharded OME-Zarr stores.

```python
nz.to_ngff_zarr('lightsheet.ome.zarr',
                multiscales,
                chunks_per_shard={'z':4, 'y':2, 'x':2},
                use_tensorstore=True,
                version=version)
```

## High Content Screening (HCS)

NGFF-Zarr provides full support for High Content Screening data, implementing
the plate and well metadata structures defined in the OME-Zarr specification.
This enables working with multi-well plate data commonly used in drug discovery
and high-throughput imaging.

### Reading HCS Data

Use [`from_hcs_zarr`] to load HCS plate data:

```python
# Load an HCS plate
plate = nz.from_hcs_zarr('screening_plate.ome.zarr')

print(f"Plate: {plate.metadata.name}")
print(f"Wells: {len(plate.metadata.wells)}")

# Access a specific well
well = plate.get_well("A", "1")  # Row A, Column 1
if well:
    print(f"Well A/1 has {len(well.images)} field(s)")

    # Get the first field image
    image = well.get_image(0)
    if image:
        print(f"Image shape: {image.images[0].data.shape}")
```

### Working with Multi-field Wells

Each well can contain multiple fields of view:

```python
well = plate.get_well("B", "2")
for field_idx in range(len(well.images)):
    image = well.get_image(field_idx)
    if image:
        # Each field is a standard multiscale image
        ngff_image = image.images[0]  # First scale level
        print(f"Field {field_idx}: {ngff_image.data.shape}")
```

### Time Series and Acquisitions

For plates with multiple acquisitions (time points or conditions):

```python
if plate.metadata.acquisitions:
    for acq in plate.metadata.acquisitions:
        print(f"Acquisition {acq.id}: {acq.name}")

    # Get image from specific acquisition
    well = plate.get_well("A", "1")
    image = well.get_image_by_acquisition(acquisition_id=0, field_index=0)
```

### HCS Validation

Validate HCS metadata during loading:

```python
# Validate against HCS schema
plate = nz.from_hcs_zarr('plate.ome.zarr', validate=True)
```

For more detailed examples and advanced usage, see the
[HCS documentation](./hcs.md).

## Convert OME-Zarr versions

To convert from OME-Zarr version 0.4, which uses the Zarr Format Specification
2, to 0.5, which uses the Zarr Format Specification 3, or vice version, specify
the desired version when writing.

```python
# Convert from 0.4 to 0.5
multiscales = from_ngff_zarr('cthead1.ome.zarr')
to_ngff_zarr('cthead1_zarr3.ome.zarr', multiscales, version='0.5')
```

```python
# Convert from 0.5 to 0.4
multiscales = from_ngff_zarr('cthead1.ome.zarr')
to_ngff_zarr('cthead1_zarr2.ome.zarr', multiscales, version='0.4')
```

[dataclass]: https://docs.python.org/3/library/dataclasses.html
[dataclasses]: https://docs.python.org/3/library/dataclasses.html
[Dask arrays]: https://docs.dask.org/en/stable/array.html
[Python Array API Standard]: https://data-apis.org/array-api/latest/
[UDUNITS-2 identifiers]: https://ngff.openmicroscopy.org/latest/#axes-md
[`Multiscales`]: ./apidocs/ngff_zarr/ngff_zarr.multiscales.md
[`NgffImage`]: ./apidocs/ngff_zarr/ngff_zarr.ngff_image.md
[`to_ngff_zarr`]: ./apidocs/ngff_zarr/ngff_zarr.to_ngff_zarr.md
[`to_ngff_image`]: ./apidocs/ngff_zarr/ngff_zarr.to_ngff_image.md
[`to_multiscales`]: ./apidocs/ngff_zarr/ngff_zarr.to_multiscales.md
[`from_ngff_zarr`]: ./apidocs/ngff_zarr/ngff_zarr.from_ngff_zarr.md
[`from_hcs_zarr`]: ./apidocs/ngff_zarr/ngff_zarr.hcs.md
[Sharded Zarr]: https://zarr.dev/zeps/accepted/ZEP0002.html
[tensorstore]: https://google.github.io/tensorstore/
