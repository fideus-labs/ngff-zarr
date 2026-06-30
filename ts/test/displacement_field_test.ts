#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * OME-Zarr v0.6 displacement / coordinate field axis types.
 *
 * A displacement (or coordinate) field is an ordinary multiscale image whose
 * vector-component axis, in the channel position, carries `type: "displacement"`
 * (or `type: "coordinate"`) instead of `type: "channel"`. The axis type is set
 * with the `axesTypes` option of `toMultiscales`.
 *
 * TypeScript equivalent of the Python test_displacement_field.py.
 */

import { assertEquals } from "@std/assert";
import {
  toMultiscales,
  toNgffImage,
} from "../src/process/to_multiscales-node.ts";
import {
  type Coordinates,
  createAxis,
  createCoordinateSystem,
  type Displacements,
  fromNgffZarr,
  toNgffZarr,
  type V06Transform,
} from "../src/mod.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";

/** A 2-component displacement field over a `yx` image. */
function fieldImage() {
  const data = new Float32Array(2 * 4 * 5);
  return toNgffImage(data, {
    dims: ["c", "y", "x"],
    shape: [2, 4, 5],
    scale: { c: 1, y: 1, x: 1 },
    translation: { c: 0, y: 0, x: 0 },
  });
}

Deno.test("axesTypes sets the displacement axis type", async () => {
  const multiscales = await toMultiscales(await fieldImage(), {
    scaleFactors: [],
    axesTypes: { c: "displacement" },
  });
  const axes = multiscales.metadata.coordinateSystems![0].axes;
  assertEquals(axes.map((a) => a.name), ["c", "y", "x"]);
  assertEquals(axes.map((a) => a.type), ["displacement", "space", "space"]);
});

Deno.test("axesTypes sets the coordinate axis type", async () => {
  const multiscales = await toMultiscales(await fieldImage(), {
    scaleFactors: [],
    axesTypes: { c: "coordinate" },
  });
  const axes = multiscales.metadata.coordinateSystems![0].axes;
  assertEquals(axes.map((a) => a.type), ["coordinate", "space", "space"]);
});

Deno.test("axesTypes default keeps the channel axis type", async () => {
  const multiscales = await toMultiscales(await fieldImage(), {
    scaleFactors: [],
  });
  assertEquals(
    multiscales.metadata.coordinateSystems![0].axes[0].type,
    "channel",
  );
});

Deno.test("displacement axis type survives a v0.6 round trip", async () => {
  const multiscales = await toMultiscales(await fieldImage(), {
    scaleFactors: [],
    axesTypes: { c: "displacement" },
  });
  const store: MemoryStore = new Map();
  await toNgffZarr(store, multiscales, { version: "0.6" });
  const imported = await fromNgffZarr(store, { version: "0.6" });
  const axes = imported.metadata.coordinateSystems![0].axes;
  assertEquals(axes.map((a) => a.type), ["displacement", "space", "space"]);
});

for (const transformType of ["displacements", "coordinates"] as const) {
  Deno.test(
    `a ${transformType} transform referencing a field round-trips`,
    async () => {
      const image = await toNgffImage(new Float32Array(4 * 5), {
        dims: ["y", "x"],
        shape: [4, 5],
        scale: { y: 1, x: 1 },
        translation: { y: 0, x: 0 },
      });
      const multiscales = await toMultiscales(image, { scaleFactors: [] });

      const intrinsic = multiscales.metadata.coordinateSystems![0];
      const outputCs = createCoordinateSystem("output", [
        createAxis("y", "space"),
        createAxis("x", "space"),
      ]);
      const transform = {
        type: transformType,
        path: "displacement_field",
        interpolation: "linear",
        input: { name: intrinsic.name },
        output: { name: outputCs.name },
        name: `to_${transformType}`,
      } as V06Transform;
      multiscales.metadata.coordinateSystems!.push(outputCs);
      multiscales.metadata.coordinateTransformations = [transform];

      const store: MemoryStore = new Map();
      await toNgffZarr(store, multiscales, { version: "0.6" });
      const imported = await fromNgffZarr(store, { version: "0.6" });

      const out = imported.metadata.coordinateTransformations!;
      assertEquals(out.length, 1);
      assertEquals(out[0].type, transformType);
      assertEquals(
        (out[0] as Displacements | Coordinates).path,
        "displacement_field",
      );
      assertEquals(
        (out[0] as Displacements | Coordinates).interpolation,
        "linear",
      );
    },
  );
}
