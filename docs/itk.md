<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ⚕️ Insight Toolkit (ITK)

Interoperability is available with the [Insight Toolkit (ITK)](https://itk.org).

Bidirectional type conversion that preserves spatial metadata is available with
`itk_image_to_ngff_image` and `ngff_image_to_itk_image`.

Once represented as an `NgffImage`, a multiscale representation can be generated
with `to_multiscales`. And an OME-Zarr can be generated from the multiscales
with `to_ome_zarr`. For more information, see the
[Python interface documentation](./python.md).

## ITK Python

An example with
[ITK Python](https://docs.itk.org/en/latest/learn/python_quick_start.html):

```python
>>> import itk
>>> import ngff_zarr as nz
>>>
>>> itk_image = itk.imread('cthead1.png')
>>>
>>> ngff_image = nz.itk_image_to_ngff_image(itk_image)
>>>
>>> # Back again
>>> itk_image = nz.ngff_image_to_itk_image(ngff_image)
```

## ITK-Wasm Python

An example with [ITK-Wasm](https://wasm.itk.org). ITK-Wasm's `Image` is a simple
Python dataclass like `NgffImage`.

```python
>>> from itkwasm_image_io import imread
>>> import ngff_zarr as nz
>>>
>>> itk_wasm_image = imread('cthead1.png')
>>>
>>> ngff_image = nz.itk_image_to_ngff_image(itk_wasm_image)
>>>
>>> # Back again
>>> itk_wasm_image = nz.ngff_image_to_itk_image(ngff_image, wasm=True)
```

## Out-of-core resampling

Resampling a fixed grid through a transform only ever reads moving-image
samples inside the transformed footprint of that grid. When the moving image is
large, remote, or chunked, materializing all of it to resample a small
overlapping region is wasteful.

`resample_bounding_box` answers *which moving-image indices will
the resample actually read?* from image geometry alone. The pixel buffers are
never touched and the Dask graphs are never computed, so both images can be
described by a handful of numbers:

```python
>>> import dask.array as da
>>> import numpy as np
>>> import ngff_zarr as nz
>>> from ngff_zarr.v06.zarr_metadata import Affine
>>>
>>> fixed = nz.NgffImage(
...     data=da.zeros((64, 64), dtype=np.uint8),
...     dims=['y', 'x'],
...     scale={'y': 1.0, 'x': 1.0},
...     translation={'y': 0.0, 'x': 0.0})
>>> moving = nz.NgffImage(
...     data=da.zeros((256, 256), dtype=np.uint8),
...     dims=['y', 'x'],
...     scale={'y': 1.0, 'x': 1.0},
...     translation={'y': 0.0, 'x': 0.0})
>>>
>>> # An RFC-5 affine mapping fixed points into moving space. Its parameters
>>> # are in Zarr axis order, and the translation is the last column, so this
>>> # shifts y by +12 and x by -4.
>>> transform = Affine(affine=[[1.0, 0.0, 12.0],
...                            [0.0, 1.0, -4.0]])
>>>
>>> region = nz.resample_bounding_box(transform, fixed, moving)
>>> region.start_index
{'y': 11, 'x': -5}
>>> region.size
{'y': 66, 'x': 66}
```

The result is keyed by dimension name, so there is no ambiguity about axis
order. `region.crop(moving)` returns a lazily sliced `NgffImage` whose
`translation` has been shifted to match, ready to hand to
`ngff_image_to_itk_image`:

```python
>>> block = region.crop(moving)   # still lazy; nothing read yet
>>> block.data.shape              # clamped into the moving image bounds
(66, 61)
>>> block.translation
{'y': 11.0, 'x': 0.0}
>>> moving_itk = nz.ngff_image_to_itk_image(block, wasm=False)  # doctest: +SKIP
```

Only this block's chunks are read when `moving_itk` is finally built. `crop`
returns `None` when the transformed grid does not overlap the moving image at
all, so a tiling loop can skip that block instead of resampling nothing. Start
indices may be negative when the grid extends past the moving origin; `crop`,
`slices` and `clamped` all clamp into bounds rather than letting a negative
index wrap around.

Use `padding` to cover the interpolator's support. The default of `1` covers
linear interpolation, which reads one neighbor beyond the continuous index
bound; pass `0` for the tight region or a larger value for wider kernels.

### Resampling the whole grid

`resample` does the loop for you: it returns a lazy `NgffImage`
on the grid of `fixed`, where every block reads only the chunks of `moving`
inside its own region and resamples that crop. The regions are computed when
the graph is built, and the blocks are tasks of one Dask graph that reference
the moving chunks directly, so a chunk that several blocks need is read and
decoded once. The full moving image is never loaded, and nothing runs until
the result is computed.

`resample` takes the same two kinds of transform as
`resample_bounding_box`, the RFC-5 `Affine` above included:

```python
>>> resampled = nz.resample(transform, fixed, moving)   # doctest: +SKIP
>>> nz.to_ome_zarr("resampled.zarr",                    # doctest: +SKIP
...     nz.to_multiscales(resampled))
```

An RFC-5 transformation is read on the intrinsic coordinate systems in both
functions, so anatomical orientation does not enter and the region reported
for a transformation is the region resampling it reads. An ITK transform is
read on ITK physical space, direction matrix included. `fields=` carries the
field of a `displacements` or `coordinates` transformation, as it does for the
bounding box.

Resampling runs through `itkwasm-downsample`, so no native ITK build is needed
and the result does not depend on the platform.

What bounds memory is streaming the result, not building it. Writing the lazy
array straight to a store keeps only the blocks in flight resident, while
`np.asarray` on it holds the whole output at once and gives most of that back.
The number of blocks in flight is a Dask setting rather than an argument here,
so that is the knob to reach for when the peak is still too high:

```python
>>> import dask                                         # doctest: +SKIP
>>> with dask.config.set(num_workers=4):                # doctest: +SKIP
...     nz.to_ome_zarr("resampled.zarr",
...         nz.to_multiscales(resampled, scale_factors=[]))
```

Measured on the 1.2 billion voxel resample of the example notebook, the peak
follows `2.7 GB + 62 MB per block in flight` with 128³ blocks and
`2.8 GB + 100 MB` with 192³ ones, each to within 0.1 GB from 1 to 24 blocks in
flight. The two slopes decompose into about 45 MB of itkwasm instance per
worker thread, which the block size does not change, plus roughly four times the
block's own bytes: the crop read in, its copy inside the wasm heap, and the
resampled block coming out. The constant is the loaded modules plus the moving
chunks the graph shares between neighbouring blocks, held while the front of the
sweep passes; it therefore depends on what else the session has imported, and
grows with the image's cross-section rather than its volume. Nothing holds the
whole output.

Each worker thread that resamples holds an itkwasm instance of its own, and
Dask caches a thread pool per distinct `num_workers`, so a process that sweeps
that setting accumulates instances instead of replacing them: six pools and
ninety threads after twelve rounds, with the resident set doubling. Shut the
pools down between settings (`dask.threaded.pools`) if you sweep it in a
long-lived process such as a notebook kernel.

**The block size is the chunking that lands on disk.** It is `fixed.data.chunks`,
and `to_multiscales` carries it through to the store, so choosing it chooses the
shape of the result rather than tuning something internal. Inheriting the source
store's chunking is the option to avoid: on that resample, with 16 blocks in
flight, 128³ ran 9% faster than the store's own 256³ for 24% less memory, and
at 24 in flight the gap widens to 17%. 128³ is also what ngff-zarr picks for a
3D pyramid, so the conventional choice is the fast one here. Smaller does not
help: below 128³ the graph costs more than the finer decomposition returns.

Building the graph is where the regions are computed, one bounding box per
output block, so the call itself is not instant and its cost grows with the
number of blocks rather than with their size. On a 513 x 1331 x 1776 grid that
is about 0.2 s for the 126 blocks of 256³, 0.2 s for the 210 of 192³, 0.8 s
for the 770 of 128³ and 5.7 s for the 5292 of 64³. It is paid once, whether or
not the result is ever computed, so prefer chunks that are large enough to be
worth a resample call.

Writing is about a quarter of the wall clock: computing every block and dropping
it, with no store to write, runs at 366 Mvoxel/s against 271 end to end.

For a worked example, see
[`py/examples/itk_elastix_transform_resample_s3.ipynb`](https://github.com/fideus-labs/ngff-zarr/blob/main/py/examples/itk_elastix_transform_resample_s3.ipynb):
it registers two whole mouse brains streamed anonymously from S3 with
ITKElastix at the coarsest pyramid level, resamples one onto the other at full
resolution (1.2 billion voxels) into a local OME-Zarr, and benchmarks the
resample level by level from local disk and from S3.

`padding` defaults to what the chosen `interpolator` requires for the
block-wise result to match an undecomposed one exactly. Those defaults are
measured, not assumed, because too small a padding does not raise: it silently
returns wrong pixels near block borders. `b_spline` needs a notably wide 16,
and 32 on float64 images, since it prefilters coefficients over the whole image
it is handed, so cropping perturbs them everywhere and the perturbation only
decays with distance, further than float32 can resolve but not further than
float64 can.

### Non-linear transforms

Deformable registration is supported. For a linear transform the region is
derived from the transformed grid corners, which is exact because a linear map
sends a rectangle to a convex region. For a non-linear one that would
*under*-bound the region -- an interior edge point can map outside the hull of
the transformed corners -- so the whole grid boundary is walked instead. Cost is
proportional to the boundary, not the pixel count, and per block that boundary
is small.

Displacement-field transforms work directly:

```python
>>> region = nz.resample_bounding_box(  # doctest: +SKIP
...     displacement_field_transform, fixed, moving)
```

### Registration transforms from Elastix

Transforms produced by a registration library can be passed directly, including
the `itk.CompositeTransform` that Elastix returns:

```python
>>> import itk
>>> composite = registration_method.GetCombinedTransform()  # doctest: +SKIP
>>> region = nz.resample_bounding_box(  # doctest: +SKIP
...     composite, fixed_block, moving)
```

The two kinds of transform are interpreted in **different coordinate spaces**,
which matters when anatomical orientation is present:

- An **RFC-5 coordinate transformation** acts on the intrinsic coordinate
  system, where a point is `translation + scale * index`. Its parameters are in
  Zarr axis order, and no direction matrix applies.
- An **ITK transform** acts on ITK physical space, so the image geometry is
  built exactly the way `ngff_image_to_itk_image` builds it, including the
  direction matrix derived from [RFC-4](./rfc4.md) anatomical orientation.

In both cases the transform maps *fixed* points into *moving* space, matching
the direction registration libraries return. A transformation's `input` and
`output` identifiers are **not resolved** here: the bounding box always maps
from the fixed image's intrinsic system to the moving image's, whatever the
identifiers name. They matter when the transformation is written into a store,
not when a region is computed.

## Converting transforms

Transforms convert in both directions, mirroring `itk_image_to_ngff_image` and
`ngff_image_to_itk_image`:

| Function | Direction |
| --- | --- |
| `ngff_transform_to_itk_transform` | RFC-5 to ITK |
| `itk_transform_to_ngff_transform` | ITK to RFC-5 |
| `itk_transform_to_ngff_matrix` | ITK to RFC-5, as raw numbers |
| `ngff_displacement_field_to_itk_transform` | RFC-5 `displacements` to an ITK field |
| `itk_displacement_field_to_ngff_transform` | ITK field to RFC-5 `displacements` |

`itk_transform_to_ngff_matrix` returns the `(matrix, offset)` pair in Zarr axis
order instead of an RFC-5 dataclass. Reach for it to *inspect* a registration
numerically, a rotation angle or a determinant say: because the conversion
simplifies by default, `itk_transform_to_ngff_transform` may hand back a
`translation` or a `scale`, so there is no `affine` field to read.

Both reconcile the two conventions that differ between the specifications:

- **Axis order.** RFC-5 orders parameters like the Zarr array, so a `zyx` image
  has `z` first; ITK orders points fastest-axis-first *by name*: x, then y,
  then z. The spatial block is permuted in both rows and columns by that
  naming, which for the canonical `zyx` is the axis reversal and for any other
  order is not.
- **Composition order.** An RFC-5 `sequence` applies its *first* entry first,
  while an ITK transform list applies its *last* entry first. The chain is
  collapsed into a single affine so the result does not depend on that
  inversion.

There is also a **center of rotation**: ITK computes `y = A(x - c) + t + c`,
while an RFC-5 affine has no center. Converting to RFC-5 folds it into the
offset as `b = t + c - A c`, so the mapping is preserved exactly.

### Storing a registration result

`itk_transform_to_ngff_transform` is what lets a registration be written into
the OME-Zarr store. It accepts any linear ITK transform, including the
`CompositeTransform` an Elastix registration returns. Every ITK transform built
on `MatrixOffsetTransformBase` answers `GetMatrix()` and `GetOffset()`, center
of rotation already folded in, whatever it stores underneath, so
parameterizations that hold angles or a quaternion (`Euler2DTransform`,
`VersorRigid3DTransform`, ...) convert exactly and just as well as an
`AffineTransform`; a composite composes its children the same way. A transform
carrying no such pair is recovered by evaluating it at the origin and along
each axis instead. Either way the result is confronted with one more
evaluation, so a transform whose matrix does not describe what it actually
does is refused rather than written:

```python
>>> import itk
>>> import ngff_zarr as nz
>>> from ngff_zarr.v06.zarr_metadata import (
...     CoordinateSystem, CoordinateSystemIdentifier)
>>>
>>> transform = registration_method.GetCombinedTransform()  # doctest: +SKIP
>>> rfc5 = nz.itk_transform_to_ngff_transform(  # doctest: +SKIP
...     transform, multiscales.metadata.dimension_names)
>>>
>>> # A transformation written into multiscales metadata must name the
>>> # coordinate systems it maps between, so declare the target and point the
>>> # transformation at it. Omit this and the store fails validation, quietly:
>>> # the write succeeds and only `from_ome_zarr(..., validate=True)` complains.
>>> intrinsic = multiscales.metadata.intrinsic_coordinate_system  # doctest: +SKIP
>>> registered = CoordinateSystem(  # doctest: +SKIP
...     name='registered', axes=list(intrinsic.axes))
>>> multiscales.metadata.coordinateSystems.append(registered)  # doctest: +SKIP
>>> rfc5.input = CoordinateSystemIdentifier(name=intrinsic.name)  # doctest: +SKIP
>>> rfc5.output = CoordinateSystemIdentifier(name=registered.name)  # doctest: +SKIP
>>>
>>> multiscales.metadata.coordinateTransformations = [rfc5]  # doctest: +SKIP
>>> nz.to_ome_zarr(  # doctest: +SKIP
...     'registered.ome.zarr', multiscales, version='0.6')
```

When the images carry [RFC-4](./rfc4.md) anatomical orientation, pass them:

```python
>>> rfc5 = nz.itk_transform_to_ngff_transform(  # doctest: +SKIP
...     transform, fixed.dims, fixed=fixed, moving=moving)
```

An ITK transform acts on physical space, direction matrix included, while an
RFC-5 transformation acts on the intrinsic coordinate systems. Given both
`NgffImage`s the conversion changes frames exactly; without them it copies the
numbers unchanged, which is exact only when neither image is oriented.
`ngff_transform_to_itk_transform` accepts the same pair for the reverse
direction.

By default the result is the least expressive transformation that represents
the mapping exactly -- `identity`, `translation`, `scale`, or a `sequence` of
scale and translation -- falling back to `affine`. RFC-5 recommends this. Pass
`simplify=False` to always get an `affine`.

A mirror never simplifies to a `scale`, however diagonal its matrix looks,
because RFC-5 requires every scale factor to be strictly positive; it falls
through to `affine`, which carries the sign.

The two slots that hold a transformation accept different things, and the
example above uses the permissive one. `multiscales > coordinateTransformations`
takes any RFC-5 type. `multiscales > datasets` takes exactly one entry, and only
a single `scale`, a single `identity`, or a two-element `sequence` of scale and
translation: a bare `translation` and an `affine` are both rejected there.

Every RFC-5 transformation type converts to ITK. `identity`, `scale`,
`translation`, `rotation`, `affine`, `mapAxis`, `byDimension` and `bijection`,
and any `sequence` of them, describe a linear mapping, and each is folded into
the single affine ITK gets: a `mapAxis` becomes its permutation matrix, a
`byDimension` writes each item into the rows its `outputAxes` name, and a
`bijection` contributes its `forward` direction, since ITK inverts an affine
itself. A `byDimension` that leaves an output axis unproduced is refused rather
than resampled, because the zero row it would leave collapses the image.
`displacements` and `coordinates` become a `DisplacementField`, given the field
they point at; the two differ by the position of the grid point itself, which
the conversion subtracts, since ITK has no absolute-coordinate transform.

Going the other way is narrower, because an ITK transform is either affine or a
field. A B-spline or a velocity field has no RFC-5 equivalent and raises
`NotImplementedError`; a field comes back as `displacements` rather than
`coordinates`.

Computing a bounding box from an **ITK** transform does not require linearity
-- that is the section above. An RFC-5 `displacements` or `coordinates`
transformation is converted first, so `resample_bounding_box` and `resample`
take the same `fields=` mapping to find its field. Because that branch works on
the intrinsic systems, where no direction matrix applies, a field carrying an
anatomical orientation is refused there: convert it with
`ngff_transform_to_itk_transform` and its `fixed=`/`moving=` pair, and pass the
ITK transform that returns.

### Displacement fields

A displacement field is a transformation and an array at once. ITK keeps the
array inside the transform; RFC-5 keeps it in the store, as a multiscale image
the `displacements` entry points at by `path`. The conversion therefore has two
outputs, the transform and the field to write next to the image, and the
functions stay free of any I/O:

```python
>>> transform, field = nz.itk_displacement_field_to_ngff_transform(  # doctest: +SKIP
...     warp, ['z', 'y', 'x'], path='displacement_field')
>>> nz.to_ome_zarr(  # doctest: +SKIP
...     'registered.ome.zarr/displacement_field',
...     nz.to_multiscales(field, scale_factors=[]), version='0.6')
>>> transform.input = CoordinateSystemIdentifier(name=intrinsic.name)  # doctest: +SKIP
>>> transform.output = CoordinateSystemIdentifier(name=registered.name)  # doctest: +SKIP
>>> multiscales.metadata.coordinateTransformations = [transform]  # doctest: +SKIP
>>> nz.to_ome_zarr(  # doctest: +SKIP
...     'registered.ome.zarr', multiscales, version='0.6', overwrite=False)
```

`warp` may be an `itk.DisplacementFieldTransform`, the vector `itk.Image` or
`itkwasm.Image` a registration tool writes the field as, or an ITK-Wasm
`DisplacementField` transform. Here `dims` names the spatial axes and nothing
else: ITK has no notion of a time or channel axis, and RFC-5 wants one field
component per input axis.

The field comes back as an `NgffImage` whose first axis holds the components
(`type: "displacement"`) followed by the spatial axes, with the grid's spacing
and origin as its scale and translation. Its components follow the axes of
`dims`, as RFC-5 requires, so an ITK `(dx, dy, dz)` vector is stored as
`(dz, dy, dx)` on a `zyx` image.

Going back, pass the field the transform points at, loaded from the same
store, keyed by its `path`:

```python
>>> imported = nz.from_ome_zarr('registered.ome.zarr')  # doctest: +SKIP
>>> transform = imported.metadata.coordinateTransformations[0]  # doctest: +SKIP
>>> field = nz.from_ome_zarr(f'registered.ome.zarr/{transform.path}')  # doctest: +SKIP
>>> itk_transforms = nz.ngff_transform_to_itk_transform(  # doctest: +SKIP
...     transform, imported.metadata.dimension_names,
...     fields={transform.path: field})
>>> region = nz.resample_bounding_box(  # doctest: +SKIP
...     transform, fixed, moving, fields={transform.path: field})
```

The result is an ITK-Wasm `TransformList` with one `DisplacementField` entry;
`itk.transform_from_dict` turns it into a native
`itk.DisplacementFieldTransform`. ITK interpolates a field linearly, so a
transform asking for another `interpolation` converts with a warning, which
RFC-5 allows: the field's interpolation is a recommendation to consumers, not
a requirement.

The `fixed` and `moving` images change frames here exactly as for an affine,
with one more rule. The field's own grid must be oriented like the fixed image
(the identity when no images are passed): RFC-5 maps the field's array to the
input coordinate system through the field's scale and translation, which
cannot express a differently oriented grid. Registrations sample the field on
the fixed grid, so this holds for their output; a field sampled elsewhere is
refused rather than written with a mapping a reader would misread, and should
be resampled onto the fixed grid first.

ITK has no notion of a non-spatial axis. Going RFC-5 to ITK, a component acting
purely on `t` or `c` -- a frame interval, say -- is therefore **projected
away**: the spatial mapping stays exact, but the ITK transform is not a
faithful copy of the input. A component that *couples* the two kinds of axis,
where `y` would depend on `c`, is refused instead, because dropping that one
would move the image. Coming back the other way, non-spatial axes are left
untransformed.

In the TypeScript package the equivalents are `ngffTransformToItkTransform`,
`itkTransformToNgffTransform` and `itkTransformToNgffMatrix`, and for fields
`itkDisplacementFieldToNgffTransform` and `ngffDisplacementFieldToItkTransform`,
both async since the field is read from and written to a Zarr array.
`resampleBoundingBox` takes the fields as an option there rather
than an argument, `{ fields: { [path]: field } }`, and
`ngffTransformToItkTransform` stays synchronous by leaving fields to the pair
above. TypeScript has no `itk` package to fall back on, so only
parameterizations that carry a matrix (`Identity`, `Translation`, `Scale`,
`Affine`) or a field (`DisplacementField`) convert there; angle- and
quaternion-based ones must be converted to an affine first.

## TypeScript

The TypeScript package provides `resampleBoundingBox`. It is async,
takes options as an object, and returns a `ResampleBoundingBox` whose
`selection()` yields a zarrita selection instead of Python slices. It accepts an
RFC-5 transformation just as the Python function does:

```typescript
import {
  createAffine,
  resampleBoundingBox,
  zarrGet,
} from "@fideus-labs/ngff-zarr";

const region = await resampleBoundingBox(
  createAffine([[1, 0, 12], [0, 1, -4]]),
  fixed,
  moving,
  { padding: 1 },
);

if (!region.isEmpty) {
  const block = await zarrGet(moving.data, region.selection(moving.dims));
}
```
