<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🗂️ RFC-8: Collections and Extensibility

[RFC-8] introduces a general extensibility mechanism for OME-Zarr metadata:
typed paths, references, and a common node structure through which images and
other data objects can be grouped into collections, nested, referenced locally
or remotely, and enriched with additional metadata.

ngff-zarr gates the RFC-8 node model on the development version `0.9.dev3`,
which extends `0.9.dev1` (RFC-3 axes and RFC-4 orientation stay normative).
At `0.9.dev3` the root `ome` value is a typed node document; the
`multiscales` wrapper of earlier versions is replaced. RFC-8 is an early
draft (status D1), so this surface is expected to evolve with it.

Support is being built in stages
([issue #714](https://github.com/fideus-labs/ngff-zarr/issues/714)). The
current stage covers the building blocks and collections:

- the `OmePath`, `Reference`, `Node` and `Collection` model
  (`ngff_zarr.rfc8`), open-world: unknown node types and prefixed extension
  attributes (e.g. `mobie:grid`) round-trip unchanged;
- reading and writing collection documents in a Zarr group
  (`from_collection_zarr` / `to_collection_zarr`) or standalone JSON
  (`from_collection_json` / `to_collection_json`);
- lazy dereferencing of path-referenced nodes (`NgffCollection.load`),
  including multiscales image stores written at earlier versions;
- the seven RFC-8 structural rules, dispatched by `validate_collection` (see
  the [validation rule reference](./validation/rule-reference.md)).

Writing *images* at `0.9.dev3` (the RFC-8 multiscale/singlescale node model)
is not implemented yet: `to_ome_zarr(version="0.9.dev3")` explains the
staging. Write the image at `0.9.dev1` and reference it from a `0.9.dev3`
collection.

## Usage

### Build and write a collection

```python
import ngff_zarr
from ngff_zarr import Collection, Node, OmePath

collection = Collection(
    "study",
    id="study",
    nodes=[
        Node(type="multiscale", name="raw", id="raw", path=OmePath("zarr", "./raw")),
        Collection("annotations", path=OmePath("json", "./annotations.json")),
    ],
    attributes={"mobie:grid": "true"},
)

# Metadata-only: the referenced stores are written separately, at their own
# versions. Nothing beneath the root is ever removed.
ngff_zarr.to_collection_zarr("study.ome.zarr", collection)
ngff_zarr.to_ome_zarr("study.ome.zarr/raw", multiscales, version="0.9.dev1")
```

### Read and dereference

```python
collection = ngff_zarr.from_collection_zarr("study.ome.zarr", validate=True)

image = collection.load("raw")  # -> NgffMultiscales
nested = collection.load("annotations")  # -> NgffCollection

# Standalone JSON documents work the same way
gallery = ngff_zarr.from_collection_json("gallery.json")
```

Relative paths resolve against the document that holds them (the Zarr group
or the JSON file), per the RFC; `file://` paths and `http(s)://` URLs stand
alone. Dereferenced documents are cached per resolved location.

Dereferencing follows whatever locations a document declares, local paths
and http(s) URLs alike, with no allowlist in between. Load collections from
sources you trust, or inspect `resolve_location` results first; a resolver
policy hook is a candidate once the RFC stabilizes.

### Coordinate systems and transformations

RFC-8 defines `coordinateSystems` and `coordinateTransformations` as node
attribute keys and references coordinate systems by a required `id` (the
`name` becomes an optional label); transformation inputs and outputs are
References. The `CoordinateSystem` and `CoordinateSystemIdentifier` model
classes carry the `id`, and typed accessors read and write the attribute
keys:

```python
from ngff_zarr.rfc8 import (
    coordinate_systems,
    coordinate_transformations,
    set_coordinate_systems,
    set_coordinate_transformations,
)
from ngff_zarr.v09.zarr_metadata import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Scale,
)

node = ngff_zarr.Node(type="multiscale", name="img", id="img")
set_coordinate_systems(
    node,
    [
        CoordinateSystem(
            name="pixels",
            id="s0",
            axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
        ),
        CoordinateSystem(
            name="physical",
            id="physical",
            axes=[Axis(name="y", type="space"), Axis(name="x", type="space")],
        ),
    ],
)
set_coordinate_transformations(
    node,
    [
        Scale(
            scale=[1.0, 2.0],
            input=CoordinateSystemIdentifier(id="s0"),
            output=CoordinateSystemIdentifier(id="physical"),
        ),
    ],
)
```

An identifier's `id` resolves before its RFC-5 `name`; a reference whose id
is not declared in the document must carry a `path` locating its document
(`reference-path-required`).

### Validation

`validate_collection` runs the RFC-8 structural rules over a raw `ome`
document or a parsed node tree. The rules are normative at `0.9.dev3`, stay
on when no version is declared, and are inert when an earlier version is
declared. `validate=True` on the readers additionally checks the bundled
RFC-8 node schema. RFC-8 publishes no normative JSON schema yet, so that
schema is authored in this repository from the RFC text; the vendored
`spec/0.9` tree remains verbatim upstream and only covers `0.9.dev1`.

### TypeScript

The TypeScript package mirrors the surface with `fromCollectionZarr` /
`toCollectionZarr`, `fromCollectionJson` / `toCollectionJson`,
`loadCollectionNode`, `validateCollection`, and the `OmeNode` / `OmePath` /
`Reference` / `NgffCollection` types (the RFC's `Node` name would collide
with the DOM type). Rule identifiers, messages and locations are
byte-identical across the two ports, pinned by a shared case fixture.

[RFC-8]: https://ngff.openmicroscopy.org/rfc/8/index.html
