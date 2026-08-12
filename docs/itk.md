<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ⚕️ Insight Toolkit (ITK)

Interoperability is available with the [Insight Toolkit (ITK)](https://itk.org).

Bidirectional type conversion that preserves spatial metadata is available with
`itk_image_to_ngff_image` and `ngff_image_to_itk_image`.

Once represented as an `NgffImage`, a multiscale representation can be generated
with `to_multiscales`. And an OME-Zarr can be generated from the multiscales
with `to_ngff_zarr`. For more information, see the
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

`itk_transform_resample_bounding_box` answers *which moving-image indices will
the resample actually read?* from image geometry alone. The pixel buffers are
never touched and the Dask graphs are never computed:

```python
>>> import itk
>>> import ngff_zarr as nz
>>>
>>> # Any linear or deformable ITK transform, including the CompositeTransform
>>> # an Elastix registration returns. It maps fixed points into moving space.
>>> transform = registration_method.GetCombinedTransform()   # doctest: +SKIP
>>> region = nz.itk_transform_resample_bounding_box(         # doctest: +SKIP
...     transform, fixed_block, moving)
>>> region.start_index                                       # doctest: +SKIP
{'y': 11, 'x': -5}
```

The result is keyed by dimension name, so there is no ambiguity about axis
order -- the underlying pipeline reports arrays fastest-axis-first, the reverse
of the Zarr order.

`region.crop(moving)` returns a lazily sliced `NgffImage` whose `translation`
has been shifted to match, ready to hand to `ngff_image_to_itk_image`:

```python
>>> block = region.crop(moving)                              # doctest: +SKIP
>>> moving_itk = nz.ngff_image_to_itk_image(block, wasm=False)  # doctest: +SKIP
```

Only that block's chunks are read. `crop` returns `None` when the transformed
grid does not overlap the moving image at all, so a tiling loop can skip it
instead of resampling nothing. Start indices may be negative when the grid
extends past the moving origin; `crop`, `slices` and `clamped` clamp into
bounds rather than letting a negative index wrap around.

The image geometry is built the way `ngff_image_to_itk_image` builds it,
including the direction matrix derived from [RFC-4](./rfc4.md) anatomical
orientation, so the transform is applied in the space a registration produced
it in.

Use `padding` to cover the interpolator's support. The default of `1` covers
linear interpolation, which reads one neighbor beyond the continuous index
bound; pass `0` for the tight region or a larger value for wider kernels.

### Non-linear transforms

Deformable registration is supported. For a linear transform the region is
derived from the transformed grid corners, which is exact because a linear map
sends a rectangle to a convex region. For a non-linear one that would
*under*-bound the region -- an interior edge point can map outside the hull of
the transformed corners -- so the whole grid boundary is walked instead. Cost is
proportional to the boundary, not the pixel count, and per block that boundary
is small.

## TypeScript

The TypeScript package provides `itkTransformResampleBoundingBox`. It is async,
takes options as an object, and returns a `ResampleBoundingBox` whose
`selection()` yields a zarrita selection instead of Python slices:

```typescript
import { itkTransformResampleBoundingBox, zarrGet } from "@fideus-labs/ngff-zarr";

const region = await itkTransformResampleBoundingBox(transform, fixed, moving, {
  padding: 1,
});

if (!region.isEmpty) {
  const block = await zarrGet(moving.data, region.selection(moving.dims));
}
```
