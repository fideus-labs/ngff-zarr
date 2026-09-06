// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * RFC-8 HCS attributes, mirroring `py/test/test_rfc8_hcs.py`.
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  acquisition,
  type OmeNode,
  plate,
  setAcquisition,
  setPlate,
  setWell,
  validateCollection,
  ValidationError,
  well,
} from "../src/mod.ts";

function widePlate(): OmeNode {
  const root: OmeNode = {
    type: "collection",
    name: "hcs-plate-001",
    nodes: [],
  };
  setPlate(root, {
    columns: [{ id: "1", name: "1" }],
    rows: [{ id: "A", name: "A" }],
    acquisitions: [{ id: "acq_0", name: "Acquisition Round 1" }],
  });
  const wellNode: OmeNode = { type: "collection", name: "well A01", nodes: [] };
  setWell(wellNode, { column: { id: "1" }, row: { id: "A" } });
  const image: OmeNode = {
    type: "multiscale",
    name: "A01_0",
    id: "A01_0",
    path: { type: "zarr", path: "./A/01/001.img" },
  };
  setAcquisition(image, { id: "acq_0" });
  wellNode.nodes!.push(image);
  root.nodes!.push(wellNode);
  return root;
}

Deno.test("plate attributes round-trip", () => {
  const root = widePlate();
  const serialized = root.attributes?.plate as Record<string, unknown>;
  assertEquals(serialized.columns, [{ id: "1", name: "1" }]);
  assertEquals(
    (serialized.acquisitions as Record<string, unknown>[])[0].id,
    "acq_0",
  );

  const parsed = plate(root);
  assertEquals(parsed?.columns, [{ id: "1", name: "1" }]);
  assertEquals(parsed?.rows, [{ id: "A", name: "A" }]);

  const wellNode = root.nodes![0];
  assertEquals(well(wellNode), { column: { id: "1" }, row: { id: "A" } });
  const image = wellNode.nodes![0];
  assertEquals(acquisition(image), { id: "acq_0" });

  setAcquisition(image, undefined);
  assertEquals(acquisition(image), undefined);
  setWell(wellNode, undefined);
  assertEquals(well(wellNode), undefined);
  setPlate(root, undefined);
  assertEquals(plate(root), undefined);
});

Deno.test("the wide plate validates", () => {
  validateCollection(widePlate(), "0.9.dev3");
});

Deno.test("the tall layout validates", () => {
  const root = widePlate();
  const wellNode = root.nodes![0];
  const image = wellNode.nodes![0];
  setAcquisition(image, undefined);
  const sub: OmeNode = { type: "collection", name: "A01_acq0", nodes: [image] };
  setAcquisition(sub, { id: "acq_0" });
  wellNode.nodes = [sub];
  validateCollection(root, "0.9.dev3");
});

Deno.test("the HCS rules fire", () => {
  let root = widePlate();
  delete (root.attributes?.plate as Record<string, unknown>).columns;
  let error = assertThrows(
    () => validateCollection(root, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "plate-columns-rows-required");

  root = widePlate();
  (root.nodes![0].attributes?.well as Record<string, unknown>).column = {
    id: "2",
  };
  error = assertThrows(
    () => validateCollection(root, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "well-reference-resolves");

  root = widePlate();
  (root.nodes![0].nodes![0].attributes as Record<string, unknown>)
    .acquisition = { id: "acq_9" };
  error = assertThrows(
    () => validateCollection(root, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "acquisition-reference-resolves");

  const orphan: OmeNode = { type: "collection", name: "orphan", nodes: [] };
  setWell(orphan, { column: { id: "1" }, row: { id: "A" } });
  error = assertThrows(
    () => validateCollection(orphan, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "well-reference-resolves");
});

Deno.test("plate entry ids join the document namespace", () => {
  const root = widePlate();
  root.id = "A"; // collides with the row id
  const error = assertThrows(
    () => validateCollection(root, "0.9.dev3"),
    ValidationError,
  );
  assertEquals(error.rule, "node-id-unique");
});

Deno.test("malformed plate metadata is rejected before typing", () => {
  const node: OmeNode = {
    type: "collection",
    name: "plate",
    nodes: [],
    attributes: { plate: { columns: [{}], rows: [{ id: "A" }] } },
  };
  assertThrows(() => plate(node), Error, "string 'id'");

  const wellNode: OmeNode = {
    type: "collection",
    name: "well",
    nodes: [],
    attributes: {
      well: { column: { id: "1", path: { type: "zarr" } }, row: { id: "A" } },
    },
  };
  assertThrows(() => well(wellNode), Error, "'type' and 'path'");
});
