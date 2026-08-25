/**
 * The TypeScript types accept what the bundled schema of their version declares,
 * and the structural rules treat an absent field the way the Python port does.
 */
import { assertEquals, assertThrows } from "@std/assert";
import { parseOmero } from "../src/utils/parse_metadata.ts";
import {
  validateAxisOrder,
  validateOmeroColorHex,
} from "../src/utils/structural_validation.ts";
import type { Metadata } from "../src/types/zarr_metadata.ts";

Deno.test("every omero channel keeps its place", () => {
  const omero = parseOmero({
    version: "0.4",
    channels: [
      {
        color: "FF0000",
        window: { min: 0, max: 255, start: 0, end: 255 },
        label: "first",
        active: true,
      },
      { color: "00FF00", label: "no window" },
      { window: { min: 0, max: 1 }, label: "no color" },
    ],
  });

  assertEquals(omero?.channels.map((c) => c.label), [
    "first",
    "no window",
    "no color",
  ]);
  assertEquals(omero?.version, "0.4");
  assertEquals(omero?.channels[0].active, true);
  assertEquals(omero?.channels[1].window, undefined);
  assertEquals(omero?.channels[2].color, undefined);
  // min/max standing in for start/end, as this library has always read them
  assertEquals(omero?.channels[2].window, { min: 0, max: 1, start: 0, end: 1 });
});

Deno.test("a channel without a color passes the color rule", () => {
  const metadata = {
    omero: { channels: [{ label: "no color" }] },
  } as unknown as Metadata;
  validateOmeroColorHex(metadata);

  const bad = {
    omero: { channels: [{ color: "nope" }] },
  } as unknown as Metadata;
  assertThrows(() => validateOmeroColorHex(bad));
});

Deno.test("an axis with no type carries no ordering constraint", () => {
  const metadata = {
    axes: [
      { name: "x", type: "space", unit: undefined },
      { name: "t", unit: undefined },
      { name: "c", type: "channel", unit: undefined },
    ],
    datasets: [],
  } as unknown as Metadata;
  validateAxisOrder(metadata);

  const ordered = {
    axes: [
      { name: "x", type: "space", unit: undefined },
      { name: "c", type: "channel", unit: undefined },
    ],
    datasets: [],
  } as unknown as Metadata;
  assertThrows(() => validateAxisOrder(ordered));
});
