// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * OME-Zarr 0.9.dev3 multiscale node stores, mirroring
 * `py/test/test_rfc8_multiscale.py`: write, read, upgrade, collection load.
 */

import { assertEquals, assertStringIncludes } from "@std/assert";
import {
  fromOmeZarr,
  loadCollectionNode,
  NgffMultiscales,
  type OmeNode,
  toCollectionZarr,
  toOmeZarr,
  upgradeOmeZarr,
} from "../src/mod.ts";
import { fromCollectionZarr } from "../src/io/from_collection_zarr.ts";
import type { MemoryStore } from "../src/io/from_ngff_zarr.ts";

const SOURCE_IMAGE = new URL(
  "../../py/test/data/input/v04/6001240.zarr",
  import.meta.url,
).pathname.replace(/^\/([A-Za-z]:)/, "$1");

function sourceMultiscales(): Promise<NgffMultiscales> {
  return fromOmeZarr(SOURCE_IMAGE, { version: "0.4" });
}

function readOme(store: MemoryStore): Record<string, unknown> {
  const document = JSON.parse(
    new TextDecoder().decode(store.get("/zarr.json")),
  );
  return document.attributes.ome;
}

Deno.test("a 0.9.dev3 store is a multiscale node document", async () => {
  const source = await sourceMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, source, { version: "0.9.dev3" });

  const ome = readOme(store);
  assertEquals(ome.version, "0.9.dev3");
  assertEquals(ome.type, "multiscale");
  assertEquals("multiscales" in ome, false);
  const attributes = ome.attributes as Record<string, unknown>;
  const systems = attributes.coordinateSystems as Record<string, unknown>[];
  for (const system of systems) {
    assertEquals(typeof system.id, "string");
  }
  const nodes = ome.nodes as Record<string, unknown>[];
  for (const node of nodes) {
    assertEquals(node.type, "singlescale");
    const path = node.path as Record<string, unknown>;
    assertEquals(path.type, "zarr");
    assertStringIncludes(path.path as string, "./");
    const nodeAttributes = node.attributes as Record<string, unknown>;
    const [transform] = nodeAttributes.coordinateTransformations as Record<
      string,
      unknown
    >[];
    assertEquals(transform.input, { id: node.id });
    assertEquals(transform.output, { id: systems[0].id });
  }
});

Deno.test("0.9.dev3 round-trips through both readers", async () => {
  const source = await sourceMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, source, { version: "0.9.dev3" });

  const read = await fromOmeZarr(store, {
    version: "0.9.dev3",
    validate: true,
  });
  assertEquals(read.metadata.version, "0.9.dev3");
  assertEquals(
    read.metadata.axes.map((axis) => axis.name),
    source.metadata.axes.map((axis) => axis.name),
  );

  const { fromOmeZarr: fromOmeZarrBrowser } = await import(
    "../src/io/from_ngff_zarr-browser.ts"
  );
  const browserRead = await fromOmeZarrBrowser(store, {
    version: "0.9.dev3",
  });
  assertEquals(browserRead.metadata.version, "0.9.dev3");
  assertEquals(
    browserRead.metadata.axes.map((axis) => axis.name),
    source.metadata.axes.map((axis) => axis.name),
  );
});

Deno.test("upgrade converts 0.9.dev1 to 0.9.dev3 in place", async () => {
  const source = await sourceMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, source, { version: "0.9.dev1" });

  await upgradeOmeZarr(store, { version: "0.9.dev3" });
  const ome = readOme(store);
  assertEquals(ome.version, "0.9.dev3");
  assertEquals(ome.type, "multiscale");
  const upgraded = await fromOmeZarr(store);
  assertEquals(upgraded.metadata.version, "0.9.dev3");
});

Deno.test("a collection loads a 0.9.dev3 image child", async () => {
  const tmp = await Deno.makeTempDir();
  try {
    const rootStore = `${tmp}/study.ome.zarr`;
    const collection: OmeNode = {
      type: "collection",
      name: "study",
      nodes: [
        {
          type: "multiscale",
          name: "img",
          id: "img",
          path: { type: "zarr", path: "./img.ome.zarr" },
        },
      ],
    };
    await toCollectionZarr(rootStore, collection);
    const source = await sourceMultiscales();
    await toOmeZarr(`${rootStore}/img.ome.zarr`, source, {
      version: "0.9.dev3",
    });

    const loaded = await loadCollectionNode(
      await fromCollectionZarr(rootStore),
      "img",
    );
    assertEquals(loaded instanceof NgffMultiscales, true);
    assertEquals(
      (loaded as NgffMultiscales).metadata.version,
      "0.9.dev3",
    );
  } finally {
    await Deno.remove(tmp, { recursive: true });
  }
});

Deno.test("the distributed singlescale layout reads", async () => {
  const source = await sourceMultiscales();
  const store: MemoryStore = new Map();
  await toOmeZarr(store, source, { version: "0.9.dev3" });

  // Move each level's transformations from the root document into the
  // level's own zarr.json, as the RFC-8 distributed layout stores them.
  const rootDocument = JSON.parse(
    new TextDecoder().decode(store.get("/zarr.json")),
  );
  const ome = rootDocument.attributes.ome;
  for (const node of ome.nodes) {
    const attributes = node.attributes;
    delete node.attributes;
    const levelKey = `/${(node.path.path as string).slice(2)}/zarr.json`;
    const levelDocument = JSON.parse(
      new TextDecoder().decode(store.get(levelKey)),
    );
    levelDocument.attributes = {
      ...(levelDocument.attributes ?? {}),
      ome: {
        version: "0.9.dev3",
        type: "singlescale",
        id: node.id,
        name: node.name,
        attributes,
      },
    };
    store.set(
      levelKey,
      new TextEncoder().encode(JSON.stringify(levelDocument)),
    );
  }
  store.set(
    "/zarr.json",
    new TextEncoder().encode(JSON.stringify(rootDocument)),
  );

  const read = await fromOmeZarr(store);
  assertEquals(read.metadata.version, "0.9.dev3");
  assertEquals(
    read.metadata.axes.map((axis) => axis.name),
    source.metadata.axes.map((axis) => axis.name),
  );
});
