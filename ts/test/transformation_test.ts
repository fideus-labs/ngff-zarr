// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * A transformation on its own, read and written with `kind: "transformation"`.
 * Mirrors `py/test/test_transformation.py`.
 */
import { assertEquals, assertRejects } from "@std/assert";
import {
  type Affine,
  type Displacements,
  fromOmeZarr,
  NgffMultiscales,
  toMultiscales,
  toNgffImage,
  toOmeZarr,
  type TransformSequence,
} from "../src/mod.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";

function affine(): Affine {
  return {
    type: "affine",
    affine: [[1, 2, 3], [4, 5, 6]],
    input: { name: "ji" },
    output: { name: "yx" },
    name: "ji to yx",
  };
}

async function rootOme(store: string): Promise<Record<string, unknown>> {
  const document = JSON.parse(await Deno.readTextFile(`${store}/zarr.json`));
  return (document.attributes as Record<string, unknown>).ome as Record<
    string,
    unknown
  >;
}

async function withTempDir(
  body: (dir: string) => Promise<void>,
): Promise<void> {
  const dir = await Deno.makeTempDir();
  try {
    await body(dir);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
}

Deno.test("transformation round trip", async () => {
  await withTempDir(async (dir) => {
    const store = `${dir}/affine.ome.zarr`;
    await toOmeZarr(store, affine());
    const ome = await rootOme(store);
    assertEquals(ome.version, "0.6");
    assertEquals(ome.coordinateTransformations, [{
      type: "affine",
      name: "ji to yx",
      input: { name: "ji" },
      output: { name: "yx" },
      affine: [[1, 2, 3], [4, 5, 6]],
    }]);
    assertEquals(
      await fromOmeZarr(store, { kind: "transformation" }),
      affine(),
    );

    const memory: MemoryStore = new Map();
    await toOmeZarr(memory, affine());
    assertEquals(
      await fromOmeZarr(memory, { kind: "transformation" }),
      affine(),
    );
  });
});

Deno.test("nested transformation round trips", async () => {
  await withTempDir(async (dir) => {
    const sequence: TransformSequence = {
      type: "sequence",
      transformations: [
        { type: "translation", translation: [0.1, 0.9] },
        { type: "scale", scale: [2, 3] },
      ],
      input: { name: "in" },
      output: { name: "out" },
      name: "in to out",
    };
    const store = `${dir}/sequence.ome.zarr`;
    await toOmeZarr(store, sequence, { version: "0.9.dev1" });
    assertEquals((await rootOme(store)).version, "0.9.dev1");
    assertEquals(
      await fromOmeZarr(store, { kind: "transformation", validate: true }),
      sequence,
    );
  });
});

Deno.test("the kind option selects what a store holds", async () => {
  await withTempDir(async (dir) => {
    const store = `${dir}/affine.ome.zarr`;
    await toOmeZarr(store, affine());
    await assertRejects(
      () => fromOmeZarr(store),
      Error,
      'kind: "transformation"',
    );
    await assertRejects(
      () => fromOmeZarr(store, { kind: "scene" }),
      Error,
      'kind: "transformation"',
    );

    const image = await toMultiscales(
      await toNgffImage(new Uint8Array(64), {
        dims: ["y", "x"],
        shape: [8, 8],
      }),
      { scaleFactors: [] },
    );
    const imageStore = `${dir}/image.ome.zarr`;
    await toOmeZarr(imageStore, image, { version: "0.6" });
    assertEquals(
      (await fromOmeZarr(imageStore)) instanceof NgffMultiscales,
      true,
    );
    await assertRejects(
      () => fromOmeZarr(imageStore, { kind: "transformation" }),
      Error,
      "holds no transformation",
    );
    await assertRejects(
      () => toOmeZarr(`${dir}/affine.ozx`, affine()),
      Error,
      "directory path",
    );
    await assertRejects(
      () => toOmeZarr(`${dir}/affine05.ome.zarr`, affine(), { version: "0.5" }),
      Error,
      "0.6",
    );
  });
});

Deno.test("transformation with a displacement field", async () => {
  await withTempDir(async (dir) => {
    const warp: Displacements = {
      type: "displacements",
      path: "coordinateTransformations/dfield",
      interpolation: "linear",
      input: { name: "fixed" },
      output: { name: "moving" },
    };
    const store = `${dir}/warp.ome.zarr`;
    await assertRejects(
      () => toOmeZarr(store, warp),
      Error,
      "write those first",
    );
    await assertRejects(
      () => toOmeZarr(store, warp, { overwrite: false }),
      Error,
      "which the store does not hold",
    );
    await assertRejects(() => Deno.stat(store), Deno.errors.NotFound);

    const field = await toNgffImage(new Float32Array(2 * 8 * 8), {
      dims: ["c", "y", "x"],
      shape: [2, 8, 8],
      scale: { c: 1, y: 0.5, x: 0.5 },
      translation: { c: 0, y: 0, x: 0 },
      axesTypes: { c: "displacement" },
    });
    await toOmeZarr(
      `${store}/${warp.path}`,
      await toMultiscales(field, {
        scaleFactors: [],
      }),
      { version: "0.6" },
    );
    await toOmeZarr(store, warp, { overwrite: false });
    const document = JSON.parse(await Deno.readTextFile(`${store}/zarr.json`));
    const consolidated = document.consolidated_metadata as {
      metadata: Record<string, unknown>;
    };
    assertEquals("coordinateTransformations" in consolidated.metadata, true);
    assertEquals(warp.path in consolidated.metadata, true);

    const read = await fromOmeZarr(store, { kind: "transformation" });
    assertEquals(read, warp);
    const fieldRead = await fromOmeZarr(
      `${store}/${(read as Displacements).path}`,
    );
    assertEquals(
      fieldRead.metadata.coordinateSystems![0].axes[0].type,
      "displacement",
    );
  });
});

for (const path of ["../dfield", "..\\dfield", "%2e%2e/dfield"]) {
  Deno.test(`the path ${path} outside the store is refused`, async () => {
    await withTempDir(async (dir) => {
      const warp: Displacements = {
        type: "displacements",
        path,
        input: { name: "fixed" },
        output: { name: "moving" },
      };
      await assertRejects(
        () => toOmeZarr(`${dir}/warp.ome.zarr`, warp, { overwrite: false }),
        Error,
        "relative path below the scene group",
      );

      const store = `${dir}/affine.ome.zarr`;
      await toOmeZarr(store, affine());
      const document = JSON.parse(
        await Deno.readTextFile(`${store}/zarr.json`),
      );
      (document.attributes.ome as Record<string, unknown>)
        .coordinateTransformations = [{ type: "displacements", path }];
      await Deno.writeTextFile(`${store}/zarr.json`, JSON.stringify(document));
      const read = await fromOmeZarr(store, { kind: "transformation" });
      assertEquals((read as Displacements).path, path);
      await assertRejects(
        () => fromOmeZarr(store, { kind: "transformation", validate: true }),
        Error,
        "relative path below the scene group",
      );
    });
  });
}
