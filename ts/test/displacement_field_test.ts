#!/usr/bin/env -S deno test --allow-read --allow-write

/**
 * OME-Zarr v0.6 displacement / coordinate field axis types.
 *
 * A displacement (or coordinate) field is an ordinary multiscale image whose
 * vector-component axis, in the channel position, carries `type: "displacement"`
 * (or `type: "coordinate"`) instead of `type: "channel"`. The axis type is set
 * with the `axesTypes` option of `NgffImage` / `toNgffImage`.
 *
 * TypeScript equivalent of the Python test_displacement_field.py.
 */

import { assertEquals, assertRejects } from "@std/assert";
import * as zarr from "zarrita";
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
  fromOmeZarr,
  toNgffZarr,
  toOmeZarr,
  type V06Transform,
} from "../src/mod.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";

/** A 2-component displacement field over a `yx` image. */
function fieldImage(axesTypes?: Record<string, string>) {
  const data = new Float32Array(2 * 4 * 5);
  return toNgffImage(data, {
    dims: ["c", "y", "x"],
    shape: [2, 4, 5],
    scale: { c: 1, y: 1, x: 1 },
    translation: { c: 0, y: 0, x: 0 },
    axesTypes,
  });
}

Deno.test("axesTypes sets the displacement axis type", async () => {
  const multiscales = await toMultiscales(
    await fieldImage({ c: "displacement" }),
    { scaleFactors: [] },
  );
  const axes = multiscales.metadata.coordinateSystems![0].axes;
  assertEquals(axes.map((a) => a.name), ["c", "y", "x"]);
  assertEquals(axes.map((a) => a.type), ["displacement", "space", "space"]);
  assertEquals(axes[0].discrete, true);
});

Deno.test("axesTypes sets the coordinate axis type", async () => {
  const multiscales = await toMultiscales(
    await fieldImage({ c: "coordinate" }),
    { scaleFactors: [] },
  );
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
  const multiscales = await toMultiscales(
    await fieldImage({ c: "displacement" }),
    { scaleFactors: [] },
  );
  const store: MemoryStore = new Map();
  await toNgffZarr(store, multiscales, { version: "0.6" });
  const imported = await fromNgffZarr(store, { version: "0.6" });
  const axes = imported.metadata.coordinateSystems![0].axes;
  assertEquals(axes.map((a) => a.type), ["displacement", "space", "space"]);
  assertEquals(axes[0].discrete, true);
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
      assertEquals(out[0].name, `to_${transformType}`);
      assertEquals(out[0].input, { name: intrinsic.name });
      assertEquals(out[0].output, { name: outputCs.name });
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

for (const transformType of ["displacements", "coordinates"] as const) {
  Deno.test(
    `an image and its ${transformType} field share a single store`,
    async () => {
      // The field, itself an OME-Zarr image, is written first into a
      // subgroup; the image is then written at the store root so the
      // transform `path` resolves within the same store.
      const axisType = transformType === "displacements"
        ? "displacement"
        : "coordinate";
      const fieldPath = `${transformType}_field`;

      const imageData = new Float32Array(4 * 5);
      for (let i = 0; i < imageData.length; i++) imageData[i] = i / 10;
      const image = await toNgffImage(imageData, {
        dims: ["y", "x"],
        shape: [4, 5],
        scale: { y: 1, x: 1 },
        translation: { y: 0, x: 0 },
      });
      const multiscales = await toMultiscales(image, { scaleFactors: [] });

      const fieldData = new Float32Array(2 * 4 * 5);
      for (let i = 0; i < fieldData.length; i++) fieldData[i] = i / 100;
      const field = await toNgffImage(fieldData, {
        dims: ["c", "y", "x"],
        shape: [2, 4, 5],
        scale: { c: 1, y: 1, x: 1 },
        translation: { c: 0, y: 0, x: 0 },
        axesTypes: { c: axisType },
      });
      const fieldMultiscales = await toMultiscales(field, {
        scaleFactors: [],
      });

      const intrinsic = multiscales.metadata.coordinateSystems![0];
      const outputCs = createCoordinateSystem("output", [
        createAxis("y", "space"),
        createAxis("x", "space"),
      ]);
      const transform = {
        type: transformType,
        path: fieldPath,
        interpolation: "linear",
        input: { name: intrinsic.name },
        output: { name: outputCs.name },
        name: "warp",
      } as V06Transform;
      multiscales.metadata.coordinateSystems!.push(outputCs);
      multiscales.metadata.coordinateTransformations = [transform];

      const tmpDir = await Deno.makeTempDir();
      try {
        await toOmeZarr(`${tmpDir}/${fieldPath}`, fieldMultiscales, {
          version: "0.6",
        });
        await toOmeZarr(tmpDir, multiscales, {
          version: "0.6",
          overwrite: false,
        });

        const imported = await fromOmeZarr(tmpDir, { version: "0.6" });
        const transforms = imported.metadata.coordinateTransformations!;
        assertEquals(transforms.length, 1);
        assertEquals(transforms[0].type, transformType);
        const importedPath =
          (transforms[0] as Displacements | Coordinates).path;
        assertEquals(importedPath, fieldPath);

        const importedImage = await zarr.get(imported.images[0].data);
        assertEquals(importedImage.data as Float32Array, imageData);

        const importedField = await fromOmeZarr(`${tmpDir}/${importedPath}`, {
          version: "0.6",
        });
        const fieldAxes = importedField.metadata.coordinateSystems![0].axes;
        assertEquals(fieldAxes.map((a) => a.type), [
          axisType,
          "space",
          "space",
        ]);
        assertEquals(fieldAxes[0].discrete, true);
        const importedFieldChunk = await zarr.get(
          importedField.images[0].data,
        );
        assertEquals(importedFieldChunk.data as Float32Array, fieldData);
      } finally {
        await Deno.remove(tmpDir, { recursive: true });
      }
    },
  );
}

Deno.test("axesTypes with an unknown dimension throws", async () => {
  const image = await fieldImage({ q: "displacement" });
  await assertRejects(() => toMultiscales(image, { scaleFactors: [] }));
});
