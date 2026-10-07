// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Tests for the RFC-3 write gate in `buildRootDocument`, the single function
 * behind every TypeScript writer.
 *
 * An RFC-3 axis model can only be serialized at OME-Zarr 0.9.dev1; targeting
 * 0.4, 0.5 or 0.6 throws. Mirrors `py/test/test_rfc3_axes.py`.
 */

import { assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { buildRootDocument } from "../src/io/to_ngff_zarr_ozx_common.ts";
import {
  type Axis,
  createScale,
  type MetadataInterface,
} from "../src/types/zarr_metadata.ts";

type TargetVersion = "0.4" | "0.5" | "0.6" | "0.9.dev1";

const PRE_RFC3: TargetVersion[] = ["0.4", "0.5", "0.6"];

/** Multiscales metadata carrying `axes`, with one matching scale per level. */
function buildMetadata(axes: Axis[]): MetadataInterface {
  return {
    axes,
    datasets: [
      {
        path: "0",
        coordinateTransformations: [createScale(axes.map(() => 1.0))],
      },
    ],
    coordinateTransformations: undefined,
    omero: undefined,
    name: "image",
    version: "0.4",
  };
}

function space(name: string): Axis {
  return { name, type: "space", unit: undefined };
}

/** One entry per RFC-3 sample-dataset shape, with the rule each one trips. */
const RFC3_SHAPES: Array<[label: string, axes: Axis[]]> = [
  ["ramp_6d", ["a", "b", "c", "d", "e", "f"].map(space)],
  ["ecg_1d", [{ name: "t", type: "time", unit: undefined }]],
  ["astronaut_xcy", [
    space("x"),
    { name: "c", type: "channel", unit: undefined },
    space("y"),
  ]],
];

Deno.test("write gate - refuses RFC-3 axis models below 0.9.dev1", () => {
  for (const [label, axes] of RFC3_SHAPES) {
    for (const version of PRE_RFC3) {
      const error = assertThrows(
        () => buildRootDocument(buildMetadata(axes), version),
        Error,
        undefined,
        `${label} should be refused at ${version}`,
      );
      assertStringIncludes(error.message, "Cannot write OME-Zarr");
      assertStringIncludes(error.message, "0.9.dev1");
    }
  }
});

Deno.test("write gate - accepts RFC-3 axis models at 0.9.dev1", () => {
  for (const [, axes] of RFC3_SHAPES) {
    const attrs = buildRootDocument(buildMetadata(axes), "0.9.dev1")
      .attributes as {
        ome: { version: string };
      };
    assertEquals(attrs.ome.version, "0.9.dev1");
  }
});

Deno.test("write gate - conventional axes still write at every version", () => {
  const axes = ["z", "y", "x"].map(space);
  for (const version of [...PRE_RFC3, "0.9.dev1"] as TargetVersion[]) {
    buildRootDocument(buildMetadata(axes), version);
  }
});

Deno.test("write gate - a non-canonical class order is refused below 0.9.dev1", () => {
  // Axis order is a spec MUST: the writer refuses what validateStructural
  // rejects, rather than reporting it and writing anyway.
  const channelLast = buildMetadata([
    { name: "z", type: "space", unit: undefined },
    { name: "y", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
    { name: "c", type: "channel", unit: undefined },
  ]);
  for (const version of PRE_RFC3) {
    const error = assertThrows(
      () => buildRootDocument(channelLast, version),
      Error,
    );
    assertStringIncludes(error.message, "axis-order");
  }
  // RFC-3 lifts the ordering rule, so the same model writes at 0.9.dev1.
  buildRootDocument(channelLast, "0.9.dev1");
});

Deno.test("write gate - repeated axis names are refused at every version", () => {
  // RFC-3 *adds* the unique-name rule rather than lifting one, so 0.9.dev1
  // is not a way out of this one.
  const axes = [space("y"), space("y"), space("x")];
  for (const version of [...PRE_RFC3, "0.9.dev1"] as TargetVersion[]) {
    const error = assertThrows(
      () => buildRootDocument(buildMetadata(axes), version),
      Error,
    );
    assertStringIncludes(error.message, "Cannot write OME-Zarr");
  }
});

Deno.test("the gate reads the axes the writer serializes", () => {
  // `buildV06MultiscalesEntry` writes the first coordinate system from
  // `metadata.axes`; `coordinateSystems[0].axes` is not tied to it, so a gate
  // reading the declared list would miss what actually lands on disk.
  const metadata = buildMetadata([
    { name: "z", type: "space", unit: undefined },
    { name: "z", type: "space", unit: undefined },
    { name: "x", type: "space", unit: undefined },
  ]);
  metadata.coordinateSystems = [
    {
      name: "intrinsic",
      axes: [
        { name: "z", type: "space", unit: undefined },
        { name: "y", type: "space", unit: undefined },
        { name: "x", type: "space", unit: undefined },
      ],
    },
  ];
  assertThrows(
    () => buildRootDocument(metadata, "0.9.dev1"),
    Error,
    "axis names must be unique",
  );
});

Deno.test("the gate skips coordinate systems the target version drops", () => {
  // v0.4 and v0.5 serialize the flat `axes` and drop `coordinateSystems`, so a
  // system the downgrade discards must not block the write. Mirrors Python,
  // which gates after `Metadata.to_version`.
  const metadata = buildMetadata([space("z"), space("y"), space("x")]);
  metadata.coordinateSystems = [
    { name: "intrinsic", axes: [space("z"), space("y"), space("x")] },
    { name: "extra", axes: ["a", "b", "c", "d", "e", "f"].map(space) },
  ];

  for (const version of ["0.4", "0.5"] as TargetVersion[]) {
    buildRootDocument(metadata, version);
  }
  // 0.6 writes both systems, so there the six-axis one is refused.
  const error = assertThrows(
    () => buildRootDocument(metadata, "0.6"),
    Error,
  );
  assertStringIncludes(
    error.message,
    "multiscales[0].coordinateSystems[1].axes",
  );
});

// The gate's message bytes are part of the cross-language contract: this
// literal is pinned identically in `py/test/test_rfc3_axes.py`.
const CANONICAL_GATE_MESSAGE_REPEATED_NAME =
  "Cannot write OME-Zarr version=\"0.9.dev1\": Axis name 'z' is repeated; " +
  "axis names must be unique within a dataset. Axes at multiscales[0].axes: " +
  "['z'(type='space'), 'z'(type='space'), 'x'(type='space')].";

const CANONICAL_GATE_MESSAGE_AXIS_COUNT =
  'Cannot write OME-Zarr version="0.4": this axis model violates that ' +
  "version's [axis-count] rule. OME-Zarr v0.4, v0.5 and v0.6 require between " +
  "2 and 5 axes, inclusive; found 6. Axes at multiscales[0].axes: " +
  "['a'(type='space'), 'b'(type='space'), 'c'(type='space'), " +
  "'d'(type='space'), 'e'(type='space'), 'f'(type='space')]. Pass " +
  'version="0.9.dev1" to write it: 0.9.dev1 is the only OME-Zarr version ' +
  "that adopts RFC-3 (arbitrary axis count, names, types and ordering).";

Deno.test("write gate - message bytes match the Python port", () => {
  const repeated = assertThrows(
    () =>
      buildRootDocument(
        buildMetadata([space("z"), space("z"), space("x")]),
        "0.9.dev1",
      ),
    Error,
  );
  assertEquals(repeated.message, CANONICAL_GATE_MESSAGE_REPEATED_NAME);

  const sixAxis = assertThrows(
    () =>
      buildRootDocument(
        buildMetadata(["a", "b", "c", "d", "e", "f"].map(space)),
        "0.4",
      ),
    Error,
  );
  assertEquals(sixAxis.message, CANONICAL_GATE_MESSAGE_AXIS_COUNT);
});

/** The Python port's message for a two-value scale over three axes. */
const CANONICAL_GATE_MESSAGE_SPAN =
  "the multiscales coordinateTransformations[0] (scale) gives 2 scale " +
  "values for the 3 axes it applies to; a transform that does not span its " +
  "axes cannot be applied by a reader.";

for (const values of [[2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]) {
  Deno.test(`a transform of ${values.length} values over 3 axes is refused`, () => {
    // The 0.4 and 0.5 models validate nothing of their own and the schemas
    // constrain what these vectors hold, not how many. Short and long both.
    const metadata = buildMetadata([space("z"), space("y"), space("x")]);
    metadata.coordinateTransformations = [createScale(values)];
    const error = assertThrows(
      () => buildRootDocument(metadata, "0.4"),
      Error,
      "span its axes",
    );
    if (values.length === 2) {
      assertEquals(error.message, CANONICAL_GATE_MESSAGE_SPAN);
    }
  });
}

Deno.test("a dataset-level transform is checked too", () => {
  const metadata = buildMetadata([space("z"), space("y"), space("x")]);
  metadata.datasets[0].coordinateTransformations = [createScale([2.0, 2.0])];
  assertThrows(
    () => buildRootDocument(metadata, "0.4"),
    Error,
    "dataset '0' coordinateTransformations[0] (scale) gives 2 scale values",
  );
});

Deno.test("a transform naming a coordinate system spans that system", () => {
  const metadata = buildMetadata([space("y"), space("x")]);
  metadata.coordinateSystems = [
    { name: "intrinsic", axes: metadata.axes },
    { name: "volume", axes: [space("z"), space("y"), space("x")] },
  ];
  metadata.coordinateTransformations = [{
    type: "scale",
    scale: [1.0, 1.0, 1.0],
    input: { name: "volume" },
    output: { name: "volume" },
  }];
  buildRootDocument(metadata, "0.6");
});

Deno.test("the first system spans the axes the writer serializes", () => {
  // `buildV06MultiscalesEntry` writes the first coordinate system from
  // `metadata.axes`, so a vector is checked against those axes, not against
  // the axes the system declares.
  const metadata = buildMetadata([space("y"), space("x")]);
  metadata.coordinateSystems = [
    { name: "intrinsic", axes: [space("z"), space("y"), space("x")] },
  ];
  metadata.coordinateTransformations = [{
    type: "scale",
    scale: [1.0, 1.0, 1.0],
    input: { name: "intrinsic" },
    output: { name: "intrinsic" },
  }];
  assertThrows(
    () => buildRootDocument(metadata, "0.6"),
    Error,
    "gives 3 scale values for the 2 axes",
  );
});

Deno.test("the browser reader routes a 0.9.dev1 store through the v0.6 reader", async () => {
  // The browser build dispatched on `isV06Version` alone, which does not cover
  // 0.9.dev1, so a store this package's own 0.9 writer produced fell through to
  // the v0.4/v0.5 parser and failed on its absent flat `axes`.
  const { toNgffZarr } = await import("../src/mod.ts");
  const { fromOmeZarr } = await import("../src/io/from_ngff_zarr-browser.ts");
  const { fromNgffZarr } = await import("../src/mod.ts");

  const testStorePath = new URL(
    "../../py/test/data/input/v04/6001240.zarr",
    import.meta.url,
  );
  const resolvedPath = testStorePath.pathname.replace(/^\/([A-Za-z]:)/, "$1");
  const source = await fromNgffZarr(resolvedPath, { version: "0.4" });

  const store: Map<string, Uint8Array> = new Map();
  await toNgffZarr(store, source, { version: "0.9.dev1" });

  const read = await fromOmeZarr(store, { version: "0.9.dev1" });
  assertEquals(
    read.metadata.axes.map((ax) => ax.name),
    source.metadata.axes.map((ax) => ax.name),
  );
  assertEquals(read.metadata.version, "0.9.dev1");
});

Deno.test("0.9.dev1 round-trip goes through the v0.6 reader", async () => {
  // `isV06Version` does not cover 0.9.dev1, so a store tagged 0.9.dev1 would
  // otherwise fall through to the v0.4 reader and fail on its coordinate
  // systems.
  const { fromNgffZarr, toNgffZarr } = await import("../src/mod.ts");

  const testStorePath = new URL(
    "../../py/test/data/input/v04/6001240.zarr",
    import.meta.url,
  );
  const resolvedPath = testStorePath.pathname.replace(/^\/([A-Za-z]:)/, "$1");
  const source = await fromNgffZarr(resolvedPath, { version: "0.4" });

  const store: Map<string, Uint8Array> = new Map();
  await toNgffZarr(store, source, { version: "0.9.dev1" });

  const roundTripped = await fromNgffZarr(store, { version: "0.9.dev1" });
  assertEquals(
    roundTripped.metadata.axes.map((ax) => ax.name),
    source.metadata.axes.map((ax) => ax.name),
  );
  // The v0.6 parser records `0.6`; a 0.9.dev1 store must report its own
  // version, which is how a caller tells the two models apart.
  assertEquals(roundTripped.metadata.version, "0.9.dev1");
});
