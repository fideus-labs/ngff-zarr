<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🧭 RFC-5: Coordinate Systems and Transformations

[RFC-5] extends OME-NGFF with named **coordinate systems** and a richer set of
**coordinate transformations** — identity, scale, translation, rotation,
affine, axis permutations, transformation sequences, per-dimension and
invertible wrappers, and array-backed *displacement* and *coordinate* fields.
This is the OME-Zarr v0.6 data model. `ngff-zarr` reads and writes it in both
the Python and TypeScript packages.

## Overview

In v0.4/v0.5 each dataset carries only a scale and translation. RFC-5 (v0.6)
generalizes this: a multiscales document declares one or more
`coordinateSystems` (the intrinsic pixel system plus any output systems), and
maps between them with `coordinateTransformations`. Transformations may live on
a dataset (the per-scale scale/translation) or at the top level of the
multiscales (an affine, a displacement field, a sequence, ...), mapping the
image into another coordinate system.

Two of these transformations reference their parameters as an array stored in
the Zarr hierarchy rather than inline:

- **`displacements`** — a vector field added to each coordinate.
- **`coordinates`** — a vector field of absolute output coordinates.

The field is itself an ordinary OME-Zarr image, so it can be written into the
**same store** as the image it transforms. That end-to-end pattern is the focus
of the [Write an image and its transformation into one store](#write-an-image-and-its-transformation-into-one-store)
section below.

## Transformations

The transformation data classes live in `ngff_zarr.v06.zarr_metadata`, the public
version-scoped API. Import from the spec version you target: `CoordinateSystem` is
defined in both `v06` and `v09` with different fields, so there is no single top-level
export. The field transforms (`Displacements`, `Coordinates`, ...) are shared, and `v09`
reuses them from `v06`.

| Class | `type` | Parameters |
| --- | --- | --- |
| `Identity` | `identity` | — |
| `Scale` | `scale` | `scale: list[float]` |
| `Translation` | `translation` | `translation: list[float]` |
| `Rotation` | `rotation` | `rotation: list[list[float]]` |
| `Affine` | `affine` | `affine: list[list[float]]` |
| `Displacements` | `displacements` | `path: str`, `interpolation: str` |
| `Coordinates` | `coordinates` | `path: str`, `interpolation: str` |
| `TransformSequence` | `sequence` | `transformations: list[Transform]` |
| `MapAxis` | `mapAxis` | `mapAxis: list[int]` |
| `ProjectAxis` | `projectAxis` | `droppedInputs: list[int]`, `createdOutputs: list[int]` |
| `ByDimension` | `byDimension` | `transformations: list[ByDimensionItem]` |
| `Bijection` | `bijection` | `forward: Transform`, `inverse: Transform` |

`MapAxis` stores an axis permutation as a transpose vector: the value at
position `i` is the input axis that becomes the `i`-th output axis, and every
zero-based input axis index appears exactly once.

`ProjectAxis` changes the dimensionality of a coordinate vector:
`droppedInputs` names the indices of the input vector to remove and
`createdOutputs` the indices of the output vector where a zero is inserted. At
least one of the two is given, the indices in each are unique, and the output
dimensionality is the input dimensionality less the dropped axes plus the
created ones. Dropping an axis loses information, so a projection is not
invertible in general.

`ByDimension` builds a high dimensional transform from lower dimensional ones;
each `ByDimensionItem` wraps a transformation with the `inputAxes` and `outputAxes`
(zero-based indices into the parent's coordinate systems) it applies to, and
every output axis is produced by exactly one item.

`Bijection` pairs an explicit `forward` transformation with its `inverse`.
Constraints that follow from a transform's parameters alone are enforced
when it is constructed, so an invalid instance cannot exist; the ones that
depend on the resolved input and output coordinate systems are checked on read, and by
`ngff_zarr.v06.zarr_metadata.validate_transform` (or the transform's own
`validate` method) for programmatically built transforms.

Every transform has an `input` and `output`, each a
`CoordinateSystemIdentifier` naming a coordinate system (`name=`) or referencing
a dataset (`path=`). `Displacements`/`Coordinates` additionally carry a `path`
pointing at the field array within the store.

Inline transforms (affine, rotation, scale, ...) are stored directly in the
multiscales metadata, so a single `to_ome_zarr` call writes both the image
pixel data and the transformation:

```python
import numpy as np
import ngff_zarr as nz
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
)

array = np.random.random((64, 64, 64)).astype(np.float32)
image = nz.to_ngff_image(array, dims=["z", "y", "x"])
multiscales = nz.to_multiscales(image, scale_factors=[])

# An affine that maps the intrinsic pixel system to an "output" system.
output_cs = CoordinateSystem(
    name="output",
    axes=[Axis(name=d, type="space") for d in ("z", "y", "x")],
)

affine = Affine(
    affine=[
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 10.0],
        [0.0, 0.0, 1.0, 15.0],
    ],
    input=CoordinateSystemIdentifier(
        name=multiscales.metadata.intrinsic_coordinate_system.name
    ),
    output=CoordinateSystemIdentifier(name=output_cs.name),
    name="to_output",
)

multiscales.metadata.coordinateSystems.append(output_cs)
multiscales.metadata.coordinateTransformations = [affine]

nz.to_ome_zarr("affine.ome.zarr", multiscales, version="0.6")
```

## Displacement and coordinate fields

A displacement (or coordinate) field is a multiscale image whose
vector-component axis, in the channel position, carries `type="displacement"`
(or `type="coordinate"`) instead of `type="channel"`. Like a channel/component
axis it is `discrete` — it indexes vector components rather than a continuous
coordinate. Set the type with the `axes_types` argument of `NgffImage`, mapping
a dimension name to its axis type:

```python
import dask.array as da
import numpy as np
import ngff_zarr as nz

# A 2-component displacement field over a yx image: the "c" dimension holds the
# (y, x) displacement vector, one component per output spatial axis.
data = da.from_array(np.zeros((2, 256, 256), dtype=np.float32))
field = nz.NgffImage(
    data=data,
    dims=("c", "y", "x"),
    scale={"c": 1.0, "y": 1.0, "x": 1.0},
    translation={"c": 0.0, "y": 0.0, "x": 0.0},
    axes_types={"c": "displacement"},
)

multiscales = nz.to_multiscales(field, scale_factors=[])
nz.to_ome_zarr("displacement.ome.zarr", multiscales, version="0.6")
```

The axis type round-trips through reading and writing. It can also be set after
the fact by editing the metadata directly, for example
`multiscales.metadata.intrinsic_coordinate_system.axes[0].type = "displacement"`.

## Write an image and its transformation into one store

A `displacements` or `coordinates` transform points at a field array by `path`.
To keep the image and the field it references together, write the field as an
OME-Zarr image into a subgroup of the same store, then reference it from the
image's transform.

Because the store metadata is consolidated after the last write, **write the
field subgroup first, then the image at the store root with `overwrite=False`**.
The final write consolidates metadata for the whole store, so the field is
discoverable under its `path`.

```python
import dask.array as da
import numpy as np
import ngff_zarr as nz
from ngff_zarr.v06.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Displacements,
)

store = "warped.ome.zarr"
field_path = "displacement_field"

# The primary image.
image = nz.to_ngff_image(
    np.random.random((256, 256)).astype(np.float32), dims=["y", "x"]
)
multiscales = nz.to_multiscales(image, scale_factors=[])

# The displacement field, an image with a displacement component axis.
field = nz.NgffImage(
    data=da.from_array(np.zeros((2, 256, 256), dtype=np.float32)),
    dims=("c", "y", "x"),
    scale={"c": 1.0, "y": 1.0, "x": 1.0},
    translation={"c": 0.0, "y": 0.0, "x": 0.0},
    axes_types={"c": "displacement"},
)
field_multiscales = nz.to_multiscales(field, scale_factors=[])

# A displacements transform whose `path` resolves to the field subgroup.
output_cs = CoordinateSystem(
    name="output",
    axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
)
transform = Displacements(path=field_path, interpolation="linear")
transform.input = CoordinateSystemIdentifier(
    name=multiscales.metadata.intrinsic_coordinate_system.name
)
transform.output = CoordinateSystemIdentifier(name=output_cs.name)
transform.name = "warp"
multiscales.metadata.coordinateSystems.append(output_cs)
multiscales.metadata.coordinateTransformations = [transform]

# Write the field subgroup first, then the image at the store root.
nz.to_ome_zarr(f"{store}/{field_path}", field_multiscales, version="0.6")
nz.to_ome_zarr(store, multiscales, version="0.6", overwrite=False)
```

Reading back resolves both the image and the field from the one store:

```python
imported = nz.from_ome_zarr(store)
transform = imported.metadata.coordinateTransformations[0]
assert transform.path == field_path

field = nz.from_ome_zarr(f"{store}/{transform.path}")
axes = field.metadata.intrinsic_coordinate_system.axes
assert [a.type for a in axes] == ["displacement", "space", "space"]
```

Use `Coordinates` instead of `Displacements` (and `axes_types={"c":
"coordinate"}` on the field) for an absolute coordinate field; the store layout
is identical.

### Interoperating with ITK

Every transformation in the table above converts to ITK with
`ngff_transform_to_itk_transform`, and `itk_transform_to_ngff_transform`
converts back. The second is how a registration result gets into the store:
convert the `CompositeTransform` an Elastix registration returns and attach it
to the multiscales metadata as shown above.

`identity`, `scale`, `translation`, `rotation`, `affine`, `mapAxis`,
`byDimension`, `bijection` and any `sequence` of them describe a linear mapping
and are folded into the single affine ITK gets. A `mapAxis` becomes its
permutation matrix, a `byDimension` writes each item into the rows its
`outputAxes` name, and a `bijection` contributes its `forward` direction.

Both directions reconcile the places where the conventions differ: RFC-5 orders
parameters in Zarr axis order while ITK orders them fastest-axis-first, an
RFC-5 `sequence` applies its first entry first while an ITK transform list
applies its last entry first, and ITK's center of rotation is folded into the
offset since an RFC-5 affine has none.

A `displacements` or `coordinates` transformation converts too, with
`itk_displacement_field_to_ngff_transform` and its inverse: the field is an
array rather than a handful of numbers, so those return, and take, the field
image alongside the transformation. Both reach ITK as a displacement field,
since ITK has no absolute-coordinate transform; coming back, a field is
`displacements`.

Building on that, `resample_bounding_box` computes which region of a moving
image a resample through the transformation would read, from geometry alone,
and `resample` resamples the grid block by block through the same
transformation. See [Out-of-core resampling](./itk.md#out-of-core-resampling)
and [Converting transforms](./itk.md#converting-transforms).

## TypeScript

The TypeScript package (`@fideus-labs/ngff-zarr`) mirrors the Python API. Field
axis types are set with the `axesTypes` option, and the same field-first,
`overwrite: false` ordering writes an image and its field into one store:

```typescript
import {
  createAxis,
  createCoordinateSystem,
  fromOmeZarr,
  toMultiscales,
  toNgffImage,
  toOmeZarr,
  type V06Transform,
} from "@fideus-labs/ngff-zarr";

const store = "warped.ome.zarr";
const fieldPath = "displacement_field";

const image = await toNgffImage(new Float32Array(256 * 256), {
  dims: ["y", "x"],
  shape: [256, 256],
});
const multiscales = await toMultiscales(image, { scaleFactors: [] });

const field = await toNgffImage(new Float32Array(2 * 256 * 256), {
  dims: ["c", "y", "x"],
  shape: [2, 256, 256],
  axesTypes: { c: "displacement" },
});
const fieldMultiscales = await toMultiscales(field, { scaleFactors: [] });

const intrinsic = multiscales.metadata.coordinateSystems![0];
const outputCs = createCoordinateSystem("output", [
  createAxis("y", "space"),
  createAxis("x", "space"),
]);
multiscales.metadata.coordinateSystems!.push(outputCs);
multiscales.metadata.coordinateTransformations = [{
  type: "displacements",
  path: fieldPath,
  interpolation: "linear",
  input: { name: intrinsic.name },
  output: { name: outputCs.name },
  name: "warp",
} as V06Transform];

// Write the field subgroup first, then the image at the store root.
await toOmeZarr(`${store}/${fieldPath}`, fieldMultiscales, { version: "0.6" });
await toOmeZarr(store, multiscales, { version: "0.6", overwrite: false });

const imported = await fromOmeZarr(store, { version: "0.6" });
const transform = imported.metadata.coordinateTransformations![0];
const field2 = await fromOmeZarr(`${store}/${(transform as { path: string }).path}`, {
  version: "0.6",
});
```

## Compatibility

The v0.6 data model requires a Zarr v3 store; the zarrista-backed writer
produces these natively. Reading v0.1–v0.5 stores is unchanged; on read, older
metadata is converted into the v0.6 data model (a single `intrinsic` coordinate
system with per-dataset scale/translation sequences).

[RFC-5]: https://ngff.openmicroscopy.org/rfc/5/index.html
