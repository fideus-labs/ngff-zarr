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

### Resampling the whole grid

`itk_transform_resample` does the loop for you: it returns a lazy `NgffImage`
on the grid of `fixed`, where every block reads only the chunks of `moving`
inside its own region and resamples that crop. The regions are computed when
the graph is built, and the blocks are tasks of one Dask graph that reference
the moving chunks directly, so a chunk that several blocks need is read and
decoded once. The full moving image is never loaded, and nothing runs until
the result is computed.

```python
>>> resampled = nz.itk_transform_resample(              # doctest: +SKIP
...     transform, fixed, moving)
>>> nz.to_ngff_zarr("resampled.zarr",                   # doctest: +SKIP
...     nz.to_multiscales(resampled))
```

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
...     nz.to_ngff_zarr("resampled.zarr",
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
