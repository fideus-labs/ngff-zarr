<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🧭 RFC-5: Coordinate Systems and Transformations

[RFC-5] extends OME-NGFF with named **coordinate systems** and a richer set of
**coordinate transformations** — identity, scale, translation, rotation,
affine, axis permutations, transformation sequences, per-dimension and
invertible wrappers, and array-backed *displacement* and *coordinate* fields.
This is the OME-Zarr v0.6 data model. OME-Zarr 0.6 was released in September
2026: see the [specification](https://ngff.openmicroscopy.org/0.6/) and the
[release announcement](https://forum.image.sc/t/ngff-specification-0-6-released/122551).
`ngff-zarr` reads and writes it in both the Python and TypeScript packages.

## Overview

In v0.4/v0.5 each dataset carries only a scale and translation. RFC-5 (v0.6)
generalizes this: a multiscales document declares one or more
`coordinateSystems` (the intrinsic pixel system plus any output systems), and
maps between them with `coordinateTransformations`. Transformations may live on
a dataset (the per-scale scale/translation) or at the top level of the
multiscales (an affine, a displacement field, a sequence, ...), mapping the
image into another coordinate system.

Two of these transformations are defined by a field image stored in the Zarr
hierarchy:

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
reuses them from `v06`. `ngff_zarr.v09` is a development model that changes between
releases; it is reachable by its import path but not re-exported from the package.

| Class | `type` | Parameters |
| --- | --- | --- |
| `Identity` | `identity` | — |
| `Scale` | `scale` | `scale: list[float]` |
| `Translation` | `translation` | `translation: list[float]` |
| `Rotation` | `rotation` | `rotation: list[list[float]]`, `path: str` (optional) |
| `Affine` | `affine` | `affine: list[list[float]]`, `path: str` (optional) |
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
pointing at the field array within the store, and `Rotation`/`Affine` one
pointing at the array that holds their matrix (see
[Matrix parameters in Zarr arrays](#matrix-parameters-in-zarr-arrays)).

Scale, translation and the other transforms with vector or index parameters
are stored inline in the multiscales metadata. Rotation and affine matrices go
to small arrays in the same store, so a single `to_ome_zarr` call writes the
image pixel data and the transformation together:

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

### Matrix parameters in Zarr arrays

RFC-5 lets a `rotation` or `affine` carry its matrix either inline, as nested
JSON arrays, or in a 2D Zarr array the transform names by `path`; the schema
accepts one form or the other, never both. `to_ome_zarr` always writes the
array form. A JSON number is decimal text, kept only to the precision of every
tool that parses and re-serializes the document, while a float64 array holds
each matrix entry bit for bit.

The example above writes the transform into the root `zarr.json` as

```json
{
  "type": "affine",
  "name": "to_output",
  "path": "coordinateTransformations/to_output",
  "input": { "name": "intrinsic" },
  "output": { "name": "output" }
}
```

and the matrix into `coordinateTransformations/to_output`: a `3 x 4` float64
array in a single uncompressed chunk, its first dimension indexing rows.

The array path is relative to the multiscales group. A transform that already
names a `path` keeps it. Otherwise the path is `coordinateTransformations/<name>`,
with the transform type (`rotation` or `affine`) standing in for a name that is
absent or not a valid Zarr node name, and a `_1`, `_2`, ... suffix when an
earlier transform took the path. Matrices nested in a `sequence`, `bijection`
or `byDimension` are written the same way. Before it touches the store, the
writer refuses a `path` that is absolute, has `.` or `..` segments, or overlaps
one of the image's own dataset arrays. A write that keeps what the store holds
-- `overwrite=False`, an append with `start_level`, or an in-place
`upgrade_ome_zarr` -- also refuses to put a matrix where the store already
holds a node, unless the store's current metadata names that node as a matrix
array.

`from_ome_zarr` loads each array back into the `rotation` or `affine` field and
keeps `path`, so the in-memory transform holds its values, converts to ITK as
is, and is rewritten to the same path. A matrix that another tool stored only
in an array reads the same way. A store an earlier release wrote with inline
matrices gets arrays when it is read and written out with `to_ome_zarr`, or
upgraded in place to a newer version with `upgrade_ome_zarr`.

The TypeScript package's `toOmeZarr`, `fromOmeZarr` and `upgradeOmeZarr` write
and read the same layout, so either package reads the matrices the other wrote.

## Displacement and coordinate fields

A displacement (or coordinate) field is a multiscale image whose
vector-component axis, in the channel position, carries `type="displacement"`
(or `type="coordinate"`) instead of `type="channel"`. Like a channel/component
axis it is `discrete` — it indexes vector components rather than a continuous
coordinate. Set the type with the `axes_types` argument of `NgffImage`, mapping
a dimension name to its axis type. `AxisType` names the types the specification
defines; RFC-3 permits any string, so a plain one is accepted just as well:

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
    axes_types={"c": nz.AxisType.Displacement},
)

multiscales = nz.to_multiscales(field, scale_factors=[])
nz.to_ome_zarr("displacement.ome.zarr", multiscales, version="0.6")
```

The axis type round-trips through reading and writing. It can also be set after
the fact by editing the metadata directly:

```python
system = multiscales.metadata.intrinsic_coordinate_system
system.axes[0].type = nz.AxisType.Displacement
```

## A standalone field store, written region by region

A registration's artifact is often the field itself. `declare_field_transform`
declares it on the field's own multiscales -- a spatial coordinate system
derived from its axes, mapped onto itself -- and the declared store can then be
created with `metadata_only=True` and filled through `open_array`, so a field
larger than memory is never assembled:

```python
import dask.array as da
import ngff_zarr as nz

field = nz.to_ngff_image(
    da.zeros((3, 512, 512, 512), dtype="float32", chunks=(3, 64, 512, 512)),
    dims=["c", "z", "y", "x"],
    scale={"c": 1.0, "z": 2.0, "y": 1.0, "x": 1.0},
    translation={"c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
)
field.axes_types = {"c": nz.AxisType.Displacement}
multiscales = nz.to_multiscales(field, scale_factors=[])
multiscales = nz.declare_field_transform(multiscales)

store = "field.ome.zarr"
nz.to_ome_zarr(store, multiscales, version="0.6", metadata_only=True)
array = nz.open_array(store, multiscales.metadata.datasets[0].path)
for start in range(0, 512, 64):
    # Whatever produces the field: a registration, a simulation, a read.
    block = my_solver.displacements(z_start=start, rows=64)
    array[:, start : start + 64] = block
```

The component order is the specification's: component *i* displaces the *i*-th
spatial axis (`z`, `y`, `x` here). A producer holding ITK-ordered vectors
(`x`, `y`, `z` components) reverses its component axis before assigning.

The declaration requires OME-Zarr 0.6: earlier versions cannot carry
multiscale-level transformations, and `to_ome_zarr` refuses to write one there
rather than dropping the declaration silently.

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
from ngff_zarr import CoordinateSystem
from ngff_zarr.v06.zarr_metadata import Axis

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
    axes_types={"c": nz.AxisType.Displacement},
)
field_multiscales = nz.to_multiscales(field, scale_factors=[])

# A displacements transform whose `path` resolves to the field subgroup.
output_cs = CoordinateSystem(
    name="output",
    axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
)
multiscales.metadata.coordinateSystems.append(output_cs)
multiscales = nz.declare_field_transform(
    multiscales,
    path=field_path,
    input_system=multiscales.metadata.intrinsic_coordinate_system.name,
    output_system=output_cs.name,
)

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
assert [a.type for a in axes] == [
    nz.AxisType.Displacement,
    nz.AxisType.Space,
    nz.AxisType.Space,
]
```

Use `Coordinates` instead of `Displacements` (and
`axes_types={"c": nz.AxisType.Coordinate}` on the field) for an absolute
coordinate field; the store layout is identical.

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

## Scenes

A **scene** groups images that share a space: the tiles of one sample, or
the same sample imaged twice. Its `ome.scene` metadata lists the
transformations between the images' coordinate systems, and may declare
coordinate systems of its own, such as a `world` system the images map into.
Each end of a transformation is a `CoordinateSystemIdentifier`: `path` points
to an image, `name` picks one of its coordinate systems.

The images are ordinary multiscales groups below the scene:

```
scene.ome.zarr
├── zarr.json      # ome.scene: the transformations between the images
├── tile_0
│   └── zarr.json  # ome.multiscales, declares the "intrinsic" system
└── tile_1
    └── zarr.json
```

`to_ome_zarr` writes the scene and its images. `from_ome_zarr` with
`kind="scene"` reads them back:

```python
import numpy as np
import ngff_zarr as nz
from ngff_zarr import CoordinateSystem, CoordinateSystemIdentifier, NgffScene
from ngff_zarr.v06.zarr_metadata import Axis, Translation


def tile(seed):
    rng = np.random.default_rng(seed)
    image = nz.to_ngff_image(
        rng.integers(0, 255, (256, 256), dtype=np.uint8),
        dims=["y", "x"],
        scale={"y": 0.5, "x": 0.5},
        axes_units={"y": "micrometer", "x": "micrometer"},
    )
    return nz.to_multiscales(image, scale_factors=[2])


world = CoordinateSystem(
    name="world",
    axes=[
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    ],
)


def to_world(path, offset):
    # "intrinsic" is the coordinate system to_multiscales gives an image.
    return Translation(
        translation=offset,
        input=CoordinateSystemIdentifier(path=path, name="intrinsic"),
        output=CoordinateSystemIdentifier(name="world"),
    )


scene = NgffScene(
    images={"tile_0": tile(0), "tile_1": tile(1)},
    coordinateSystems=[world],
    coordinateTransformations=[
        to_world("tile_0", [0.0, 0.0]),
        to_world("tile_1", [0.0, 128.0]),
    ],
)
nz.to_ome_zarr("scene.ome.zarr", scene)

scene = nz.from_ome_zarr("scene.ome.zarr", kind="scene")
scene.images["tile_1"]  # an NgffMultiscales, its pixels read lazily
scene.coordinateTransformations[1].translation  # [0.0, 128.0]
```

The writer checks the scene before writing anything:

- coordinate system names are unique, and their axes follow the version's
  rules (five axes at most at 0.6, any number at 0.9.dev1);
- every transformation names both ends, each found in the scene or in the
  image at its `path`;
- a transformation fits the systems it joins: one translation value per axis,
  a `mapAxis` that permutes them;
- coordinate systems and images form one connected graph;
- every node referenced by `path` exists in the store.

A scene that fails a check raises `ValueError`. `validate=True` runs the same
checks after reading. Put the common coordinate system first: viewers use it
as the default. A scene reads from a local directory, a URL or an `.ozx`
archive, and `to_ome_zarr` passes options such as `chunks_per_shard` on to
each image.

`kind` says what a store holds, like zarrita's `open`. Without it,
`from_ome_zarr` reads multiscales images only. `NgffScene.from_ome_zarr` is a
shortcut for `kind="scene"`, and a path into one of the scene's images reads
that image.

An example with a displacement field:

```python
from ngff_zarr.v06.zarr_metadata import Displacements

# The field is an image of its own: write it first, below the scene, then
# the scene with overwrite=False to keep it.
field_path = "coordinateTransformations/dfield"
nz.to_ome_zarr(f"scene.ome.zarr/{field_path}", field_multiscales, version="0.6")

scene.coordinateTransformations.append(
    Displacements(
        path=field_path,
        interpolation="linear",
        input=CoordinateSystemIdentifier(path="tile_0", name="intrinsic"),
        output=CoordinateSystemIdentifier(path="tile_1", name="intrinsic"),
    )
)
nz.to_ome_zarr("scene.ome.zarr", scene, overwrite=False)

scene = nz.from_ome_zarr("scene.ome.zarr", kind="scene")
field = nz.from_ome_zarr(f"scene.ome.zarr/{scene.coordinateTransformations[2].path}")
```

## A transformation on its own

A transformation can be stored without an image or a scene. The root
group's `ome.coordinateTransformations` holds it, as in the spec's standalone
examples. `to_ome_zarr` takes the transformation object, and `from_ome_zarr`
with `kind="transformation"` reads it back as written:

```python
import ngff_zarr as nz
from ngff_zarr import CoordinateSystemIdentifier
from ngff_zarr.v06.zarr_metadata import Affine

registration = Affine(
    affine=[[1.0, 0.0, 12.5], [0.0, 1.0, -3.0]],
    input=CoordinateSystemIdentifier(name="moving"),
    output=CoordinateSystemIdentifier(name="fixed"),
)
nz.to_ome_zarr("registration.ome.zarr", registration)

transform = nz.from_ome_zarr("registration.ome.zarr", kind="transformation")
nz.ngff_transform_to_itk_transform(transform, dims=["y", "x"])
```

A transformation stored as an array, such as a `displacements` field, points
to it by `path`. Write the field first, below the store, then the
transformation with `overwrite=False`, as for a scene.

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

A scene is written and read the same way. `kind: "scene"` types the result
as an `NgffScene`:

```typescript
import {
  fromOmeZarr,
  NgffScene,
  toMultiscales,
  toNgffImage,
  toOmeZarr,
} from "@fideus-labs/ngff-zarr";

async function tile(seed: number) {
  const data = new Uint8Array(256 * 256).map((_, i) => (i * 31 + seed) % 256);
  const image = await toNgffImage(data, {
    dims: ["y", "x"],
    shape: [256, 256],
    scale: { y: 0.5, x: 0.5 },
  });
  return toMultiscales(image, { scaleFactors: [2] });
}

const scene = new NgffScene({
  images: { tile_0: await tile(0), tile_1: await tile(1) },
  coordinateSystems: [{
    name: "world",
    axes: [
      { name: "y", type: "space", unit: "micrometer" },
      { name: "x", type: "space", unit: "micrometer" },
    ],
  }],
  coordinateTransformations: [
    {
      type: "translation",
      translation: [0, 0],
      input: { path: "tile_0", name: "intrinsic" },
      output: { name: "world" },
    },
    {
      type: "translation",
      translation: [0, 128],
      input: { path: "tile_1", name: "intrinsic" },
      output: { name: "world" },
    },
  ],
});
await toOmeZarr("scene.ome.zarr", scene);

const read = await fromOmeZarr("scene.ome.zarr", {
  kind: "scene",
  validate: true,
});
read.images.tile_1; // an NgffMultiscales, its pixels read lazily
```

The same checks run as in Python. A scene reads from a local directory, an
HTTP(S) URL, or a store object such as a `MemoryStore` or a `ZipFileStore`
over an `.ozx` archive. Without `kind: "scene"`, `fromOmeZarr` keeps its
`NgffMultiscales` result type and refuses a scene store. A field is written
first with `toOmeZarr`, then the scene with `{ overwrite: false }`. A
transformation on its own works the same way: `toOmeZarr(store, transform)`
writes it, to a directory or a `MemoryStore`, and
`fromOmeZarr(store, { kind: "transformation" })` reads it back.

Both are written in the browser too. The browser module's `toOmeZarr` writes
a scene or a transformation to a `MemoryStore`, and `toOmeZarrOzx` zips
either into an `.ozx` archive, as it does an image; in Node.js,
`toOmeZarr("scene.ozx", scene)` writes the archive to a file. A scene's
images are sharded as an image's `.ozx` is, and `onProgress` counts the
writes of all of them:

```typescript
import {
  fromOmeZarr,
  toOmeZarr,
  toOmeZarrOzx,
} from "@fideus-labs/ngff-zarr/browser";

const archive = await toOmeZarrOzx(scene, {
  onProgress: (written, total) => console.log(`${written} of ${total}`),
});
// e.g. download it as scene.ome.zarr.ozx

const store = new Map<string, Uint8Array>();
await toOmeZarr(store, scene);
const read = await fromOmeZarr(store, { kind: "scene" });
```

`toOmeZarrOzx` writes an archive in one piece, so a scene or a
transformation that references a stored node by `path`, such as a
`displacements` field, is staged instead, as Python stages a directory and
packs it with `write_store_to_zip`. The field is written below its path of a
`MemoryStore` (or a directory, in Node.js) with the `path` option, then the
scene with `{ overwrite: false }`, and `storeToZip` packs the store:

```typescript
import {
  fromOmeZarr,
  itkDisplacementFieldToNgffTransform,
  storeToZip,
  toMultiscales,
  toOmeZarr,
} from "@fideus-labs/ngff-zarr/browser";

// itkField: an ITK-Wasm displacement field image, such as a registration's
const { transform: warp, field } = await itkDisplacementFieldToNgffTransform(
  itkField,
  ["y", "x"],
  { path: "coordinateTransformations/dfield" },
);
warp.input = { path: "tile_0", name: "intrinsic" };
warp.output = { path: "tile_1", name: "intrinsic" };
scene.coordinateTransformations.push(warp);

const store = new Map<string, Uint8Array>();
await toOmeZarr(store, await toMultiscales(field, { scaleFactors: [] }), {
  version: "0.6",
  path: warp.path,
});
await toOmeZarr(store, scene, { overwrite: false });
const archive = storeToZip(store); // records the version the scene declares

// The field reads back below its path of the store, or of the archive
const read = await fromOmeZarr(store, { path: warp.path });
```

In Node.js, `await storeToZip("scene.ome.zarr", "scene.ozx")` packs a staged
directory into a file. The root of the archive consolidates the field's
arrays along with the scene's images.

## Compatibility

The v0.6 data model requires a Zarr v3 store; the zarrista-backed writer
produces these natively. Reading v0.1–v0.5 stores is unchanged; on read, older
metadata is converted into the v0.6 data model (a single `intrinsic` coordinate
system with per-dataset scale/translation sequences).

[RFC-5]: https://ngff.openmicroscopy.org/rfc/5/index.html
