// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-8 collection reading, writing and lazy loading, mirroring
 * `py/test/test_rfc8_io.py` over in-memory and temporary-directory stores.
 */

import { assertEquals, assertRejects, assertStringIncludes } from "@std/assert";
import { NgffMultiscales } from "../src/types/multiscales.ts";
import {
  fromCollectionJson,
  fromCollectionZarr,
  fromOmeZarr,
  loadCollectionNode,
  NgffCollection,
  type OmeNode,
  toCollectionJson,
  toCollectionZarr,
  toOmeZarr,
} from "../src/mod.ts";
import { buildRootAttributes } from "../src/io/to_ngff_zarr_ozx_common.ts";
import { documentBase, resolveLocation } from "../src/io/collection_common.ts";
import type { ImageVersion } from "../src/types/supported_versions.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";

const SOURCE_IMAGE = new URL(
  "../../py/test/data/input/v04/6001240.zarr",
  import.meta.url,
).pathname.replace(/^\/([A-Za-z]:)/, "$1");

function studyCollection(): OmeNode {
  return {
    type: "collection",
    name: "study",
    id: "study",
    attributes: { "mobie:grid": "true" },
    nodes: [
      {
        type: "multiscale",
        name: "raw",
        id: "raw",
        path: { type: "zarr", path: "./raw" },
      },
      {
        type: "collection",
        name: "nested",
        path: { type: "json", path: "./nested_collection.json" },
      },
    ],
  };
}

function imageMultiscales(): Promise<NgffMultiscales> {
  return fromOmeZarr(SOURCE_IMAGE, { version: "0.4" });
}

Deno.test("memory-store round trip with root attributes", async () => {
  const store: MemoryStore = new Map();
  await toCollectionZarr(store, studyCollection(), {
    rootAttributes: { lab: "fideus" },
  });
  const document = JSON.parse(
    new TextDecoder().decode(store.get("/zarr.json")),
  );
  assertEquals(document.zarr_format, 3);
  assertEquals(document.attributes.ome.version, "0.9.dev3");
  assertEquals(document.attributes.lab, "fideus");

  const collection = await fromCollectionZarr(store, { validate: true });
  assertEquals(collection.root, studyCollection());
  assertEquals(collection.version, "0.9.dev3");
  assertEquals(collection.rootAttributes, { lab: "fideus" });
  assertEquals(collection.base, undefined);
});

Deno.test("collection writer accepts only 0.9.dev3", async () => {
  await assertRejects(
    () =>
      toCollectionZarr(new Map(), studyCollection(), { version: "0.9.dev1" }),
    Error,
    "0.9.dev3",
  );
});

Deno.test("collection writer validates by default", async () => {
  const invalid: OmeNode = { type: "collection", name: "c" };
  await assertRejects(
    () => toCollectionZarr(new Map(), invalid),
    Error,
    "node-nodes-xor-path",
  );
  await toCollectionZarr(new Map(), invalid, { validate: false });
});

Deno.test("standalone JSON round trip", async () => {
  const document = await toCollectionJson(studyCollection());
  assertEquals(
    (document.ome as Record<string, unknown>).version,
    "0.9.dev3",
  );
  const collection = await fromCollectionJson(
    document as Record<string, unknown>,
  );
  assertEquals(collection.root, studyCollection());
  assertEquals(collection.base, undefined);
});

Deno.test("fromOmeZarr names the collection reader on a dev3 store", async () => {
  const store: MemoryStore = new Map();
  await toCollectionZarr(store, studyCollection());
  const error = await assertRejects(() => fromOmeZarr(store), Error);
  assertStringIncludes(String(error), "fromCollectionZarr");
});

Deno.test("the browser reader also names the collection reader", async () => {
  const { fromOmeZarr: fromOmeZarrBrowser } = await import(
    "../src/io/from_ngff_zarr-browser.ts"
  );
  const store: MemoryStore = new Map();
  await toCollectionZarr(store, studyCollection());
  const error = await assertRejects(() => fromOmeZarrBrowser(store), Error);
  assertStringIncludes(String(error), "fromCollectionZarr");
});

Deno.test("fromCollectionZarr names the image reader on an image store", async () => {
  const store: MemoryStore = new Map();
  const multiscales = await imageMultiscales();
  await toOmeZarr(store, multiscales, { version: "0.5" });
  const error = await assertRejects(() => fromCollectionZarr(store), Error);
  assertStringIncludes(String(error), "fromOmeZarr");
});

Deno.test("buildRootAttributes refuses 0.9.dev3 with guidance", async () => {
  const multiscales = await imageMultiscales();
  let message = "";
  try {
    buildRootAttributes(
      multiscales.metadata,
      "0.9.dev3" as unknown as ImageVersion,
    );
  } catch (error) {
    message = String(error);
  }
  assertStringIncludes(message, "toCollectionZarr");
});

Deno.test("load resolves nodes across a temporary directory layout", async () => {
  const tmp = await Deno.makeTempDir();
  try {
    const storePath = `${tmp}/collection.ome.zarr`;
    await toCollectionZarr(storePath, studyCollection());
    const multiscales = await imageMultiscales();
    await toOmeZarr(`${storePath}/raw`, multiscales, { version: "0.9.dev1" });
    await toCollectionJson(
      { type: "collection", name: "nested", nodes: [] },
      { path: `${storePath}/nested_collection.json` },
    );

    const collection = await fromCollectionZarr(storePath, {
      validate: true,
    });
    assertEquals(collection.base, storePath);

    const image = await loadCollectionNode(collection, "raw");
    assertEquals(image instanceof NgffMultiscales, true);
    assertEquals((image as NgffMultiscales).metadata.version, "0.9.dev1");
    // Loaded documents are cached per resolved location.
    const again = await loadCollectionNode(collection, "raw");
    assertEquals(again === image, true);
    const byReference = await loadCollectionNode(collection, { id: "raw" });
    assertEquals(byReference === image, true);

    const nested = await loadCollectionNode(collection, "nested");
    assertEquals(nested instanceof NgffCollection, true);
    assertEquals((nested as NgffCollection).base, storePath);
  } finally {
    await Deno.remove(tmp, { recursive: true });
  }
});

Deno.test("the external collection fixture layout loads", async () => {
  const fixtures = new URL(
    "../../py/test/fixtures/rfc8/external_collection/",
    import.meta.url,
  );
  const tmp = await Deno.makeTempDir();
  try {
    for (const name of ["parent_collection.json", "child_collection.json"]) {
      await Deno.writeTextFile(
        `${tmp}/${name}`,
        await Deno.readTextFile(new URL(name, fixtures)),
      );
    }
    const parent = await fromCollectionJson(`${tmp}/parent_collection.json`);
    const child = await loadCollectionNode(parent, "sub-collection");
    assertEquals(child instanceof NgffCollection, true);
    assertEquals((child as NgffCollection).root.name, "sub-collection");
    assertEquals(
      (child as NgffCollection).root.nodes?.[0].name,
      "image-label",
    );
    // The child collection resolves its own relative paths from its file.
    assertEquals((child as NgffCollection).base, tmp);
  } finally {
    await Deno.remove(tmp, { recursive: true });
  }
});

Deno.test("an external reference must declare the requested id", async () => {
  const tmp = await Deno.makeTempDir();
  try {
    await Deno.writeTextFile(
      `${tmp}/child.json`,
      JSON.stringify(
        await toCollectionJson({
          type: "collection",
          name: "child",
          nodes: [{ type: "multiscale", name: "img", id: "img" }],
        }),
      ),
    );
    const parent = new NgffCollection({
      root: { type: "collection", name: "parent", nodes: [] },
      base: tmp,
    });
    await assertRejects(
      () =>
        loadCollectionNode(parent, {
          id: "missing",
          path: { type: "json", path: "./child.json" },
        }),
      Error,
      "missing",
    );
  } finally {
    await Deno.remove(tmp, { recursive: true });
  }
});

Deno.test("relative locations keep parent traversals above a relative base", () => {
  assertEquals(
    resolveLocation({ type: "json", path: "../child.json" }, "."),
    "../child.json",
  );
  assertEquals(
    resolveLocation({ type: "json", path: "../../c.json" }, "a"),
    "../c.json",
  );
});

Deno.test("documentBase handles bare and root-level filenames", () => {
  assertEquals(documentBase({ type: "json", path: "x" }, "child.json"), ".");
  assertEquals(documentBase({ type: "json", path: "x" }, "/a.json"), "/");
});

Deno.test("the browser writer preserves foreign root attributes", async () => {
  const zarr = await import("zarrita");
  const { toCollectionZarr: toCollectionZarrBrowser } = await import(
    "../src/io/to_collection_zarr-browser.ts"
  );
  const store = new Map<string, Uint8Array>();
  await zarr.create(zarr.root(store), {
    attributes: { "myorg:note": "kept", ome: { stale: true } },
  });
  await toCollectionZarrBrowser(store, {
    type: "collection",
    name: "c",
    nodes: [],
  });
  const written = JSON.parse(
    new TextDecoder().decode(store.get("/zarr.json")),
  );
  assertEquals(written.attributes["myorg:note"], "kept");
  assertEquals(written.attributes.ome.type, "collection");
});
