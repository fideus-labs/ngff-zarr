// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Declaring a field image as an RFC-5 field transform.
 *
 * Mirrors `py/test/test_declare_field_transform.py`. What is checked is the
 * entry the declaration produces and the refusals it owes: a store that only
 * types its component axis is labelled, not applicable, and every mistake this
 * function rules out leaves a plausible-looking field behind.
 */

import { assertEquals, assertThrows } from "@std/assert";
import * as zarr from "zarrita";
import {
  type CoordinateSystem,
  declareFieldTransform,
  type Metadata,
  NgffImage,
  NgffMultiscales,
} from "../src/mod.ts";

const DIMS = ["c", "z", "y", "x"];

async function fieldImage(
  shape: number[] = [3, 4, 5, 6],
  dims: string[] = DIMS,
  axesTypes: Record<string, string> = { c: "displacement" },
): Promise<NgffImage> {
  const data = await zarr.create(zarr.root(new Map()).resolve("data"), {
    shape,
    chunk_shape: shape,
    data_type: "float64",
    fill_value: 0,
  });
  return new NgffImage({
    data,
    dims,
    scale: Object.fromEntries(dims.map((dim, i) => [dim, i === 0 ? 1 : i])),
    translation: Object.fromEntries(dims.map((dim) => [dim, 0])),
    name: "image",
    axesUnits: undefined,
    axesTypes,
    computedCallbacks: undefined,
  });
}

function multiscalesOf(image: NgffImage, systems?: CoordinateSystem[]) {
  const axes = image.dims.map((name) => ({
    name,
    type: name === "c" ? "channel" : "space",
    unit: undefined,
  }));
  const metadata = {
    axes,
    datasets: [{ path: "scale0/image", coordinateTransformations: [] }],
    coordinateTransformations: undefined,
    coordinateSystems: systems ?? [{ name: "intrinsic", axes }],
    omero: undefined,
    name: "image",
    version: "0.6",
  } as unknown as Metadata;
  return new NgffMultiscales({
    images: [image],
    metadata,
    scaleFactors: undefined,
    method: undefined,
    chunks: undefined,
  });
}

Deno.test("a standalone declaration names a system mapped onto itself", async () => {
  const declared = declareFieldTransform(multiscalesOf(await fieldImage()));
  const entries = declared.metadata.coordinateTransformations ?? [];
  assertEquals(entries.length, 1);
  const entry = entries[0];
  assertEquals(entry.type, "displacements");
  assertEquals(entry.input?.name, "physical");
  assertEquals(entry.output?.name, "physical");
  assertEquals(
    (entry as { path: string }).path,
    declared.metadata.datasets[0].path,
  );
  const physical = (declared.metadata.coordinateSystems ?? []).find(
    (system) => system.name === "physical",
  );
  assertEquals(physical?.axes.map((axis) => axis.name), ["z", "y", "x"]);
});

Deno.test("declaring appends and leaves the argument alone", async () => {
  const multiscales = multiscalesOf(await fieldImage());
  const once = declareFieldTransform(multiscales);
  const twice = declareFieldTransform(once, { interpolation: "nearest" });
  assertEquals(
    (twice.metadata.coordinateTransformations ?? []).map((entry) =>
      (entry as { interpolation?: string }).interpolation
    ),
    ["linear", "nearest"],
  );
  // The spatial system is declared once, not once per call.
  assertEquals(
    (twice.metadata.coordinateSystems ?? []).filter((system) =>
      system.name === "physical"
    ).length,
    1,
  );
  assertEquals(multiscales.metadata.coordinateTransformations, undefined);
});

Deno.test("a declaration beside the image references its systems", async () => {
  const axes = ["z", "y", "x"].map((name) => ({
    name,
    type: "space",
    unit: undefined,
  }));
  const multiscales = multiscalesOf(await fieldImage(), [
    { name: "fixed", axes },
    { name: "moving", axes },
  ] as CoordinateSystem[]);

  const declared = declareFieldTransform(multiscales, {
    path: "displacements/field",
    inputSystem: "fixed",
    outputSystem: "moving",
  });
  const entry = (declared.metadata.coordinateTransformations ?? [])[0];
  assertEquals((entry as { path: string }).path, "displacements/field");
  assertEquals(entry.input?.name, "fixed");
  assertEquals(entry.output?.name, "moving");
});

Deno.test("a named system that is not declared is refused", async () => {
  const multiscales = multiscalesOf(await fieldImage());
  assertThrows(
    () =>
      declareFieldTransform(multiscales, {
        inputSystem: "fixed",
        outputSystem: "moving",
      }),
    Error,
    "names no declared coordinate system",
  );
});

Deno.test("a path without named systems is refused", async () => {
  // With path the field is elsewhere: deriving a system from the image's own
  // axes would declare a transform over the wrong space.
  const multiscales = multiscalesOf(await fieldImage());
  assertThrows(
    () => declareFieldTransform(multiscales, { path: "displacements/field" }),
    Error,
    "pass inputSystem and outputSystem",
  );
});

Deno.test("one-sided system naming is refused", async () => {
  const multiscales = multiscalesOf(await fieldImage());
  assertThrows(
    () => declareFieldTransform(multiscales, { inputSystem: "fixed" }),
    Error,
    "both inputSystem and outputSystem",
  );
});

Deno.test("a field without a typed component axis is refused", async () => {
  const multiscales = multiscalesOf(await fieldImage([3, 4, 5, 6], DIMS, {}));
  assertThrows(
    () => declareFieldTransform(multiscales),
    Error,
    "exactly one axis of type",
  );
});

Deno.test("a non-spatial axis is refused", async () => {
  // A field transform is spatial: relabelling a time axis "space" declares a
  // transform the package's own converter then refuses to apply.
  const multiscales = multiscalesOf(
    await fieldImage([3, 2, 5, 6], ["c", "t", "y", "x"]),
  );
  assertThrows(
    () => declareFieldTransform(multiscales),
    Error,
    "spatial axes only",
  );
});

Deno.test("a component count that misses an axis is refused", async () => {
  const multiscales = multiscalesOf(await fieldImage([2, 4, 5, 6]));
  assertThrows(
    () => declareFieldTransform(multiscales),
    Error,
    "2 components per point",
  );
});

Deno.test("a transform type off the prototype chain is refused", async () => {
  // `transformType in COMPONENT_TYPES` is true for "toString", which a JS
  // caller is free to pass: the table then yields a function where the axis
  // type belongs, and with `path` the name lands in the entry's own `type`.
  const multiscales = multiscalesOf(await fieldImage([3, 4, 5, 6]));
  for (const inherited of ["toString", "constructor"]) {
    assertThrows(
      () =>
        declareFieldTransform(multiscales, {
          transformType: inherited as "displacements",
        }),
      Error,
      "transformType must be one of",
    );
  }
});

Deno.test("a declared system that is not spatial is refused", async () => {
  // The names line up, so the arity check passes; the types do not, and the
  // transform would map over a system the converter refuses to apply.
  const multiscales = multiscalesOf(await fieldImage([3, 4, 5, 6]));
  multiscales.metadata.coordinateSystems = [
    ...(multiscales.metadata.coordinateSystems ?? []),
    {
      name: "physical",
      axes: [
        { name: "z", type: "time", unit: undefined },
        { name: "y", type: "space", unit: undefined },
        { name: "x", type: "space", unit: undefined },
      ],
    },
  ];
  assertThrows(
    () => declareFieldTransform(multiscales),
    Error,
    "spatial axes only",
  );
});

Deno.test("named systems of different dimension are refused", async () => {
  // With `path` the field's shape is elsewhere, but the two systems can be
  // checked against each other: the converter builds one dimension for both
  // ends, so a 2D input against a 3D output is unapplicable.
  const system = (name: string, dims: string[]) => ({
    name,
    axes: dims.map((d) => ({ name: d, type: "space", unit: undefined })),
  });
  const multiscales = multiscalesOf(await fieldImage([3, 4, 5, 6]));
  multiscales.metadata.coordinateSystems = [
    ...(multiscales.metadata.coordinateSystems ?? []),
    system("flat", ["y", "x"]),
    system("volume", ["z", "y", "x"]),
  ];
  assertThrows(
    () =>
      declareFieldTransform(multiscales, {
        path: "displacements/field",
        inputSystem: "flat",
        outputSystem: "volume",
      }),
    Error,
    "same dimension",
  );
});
