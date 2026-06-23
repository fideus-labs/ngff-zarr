// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * Unit tests for RFC 4 anatomical orientation implementation.
 * Ported from py/test/test_rfc4.py
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  addAnatomicalOrientationToAxis,
  anatomicalOrientationToItkDirection,
  AnatomicalOrientationValues,
  createAnatomicalOrientation,
  itkLpsToAnatomicalOrientation,
  LPS,
  orientationFromName,
  RAS,
  removeAnatomicalOrientationFromAxis,
} from "../src/types/rfc4.ts";

Deno.test("createAnatomicalOrientation - creates valid object", () => {
  const orientation = createAnatomicalOrientation(
    AnatomicalOrientationValues.LeftToRight,
  );
  assertEquals(orientation.type, "anatomical");
  assertEquals(orientation.value, AnatomicalOrientationValues.LeftToRight);
});

Deno.test("itkLpsToAnatomicalOrientation - x axis returns right-to-left", () => {
  const xOrientation = itkLpsToAnatomicalOrientation("x");
  assertEquals(xOrientation !== undefined, true);
  assertEquals(xOrientation?.type, "anatomical");
  assertEquals(xOrientation?.value, AnatomicalOrientationValues.RightToLeft);
});

Deno.test("itkLpsToAnatomicalOrientation - y axis returns anterior-to-posterior", () => {
  const yOrientation = itkLpsToAnatomicalOrientation("y");
  assertEquals(yOrientation !== undefined, true);
  assertEquals(yOrientation?.type, "anatomical");
  assertEquals(
    yOrientation?.value,
    AnatomicalOrientationValues.AnteriorToPosterior,
  );
});

Deno.test("itkLpsToAnatomicalOrientation - z axis returns inferior-to-superior", () => {
  const zOrientation = itkLpsToAnatomicalOrientation("z");
  assertEquals(zOrientation !== undefined, true);
  assertEquals(zOrientation?.type, "anatomical");
  assertEquals(
    zOrientation?.value,
    AnatomicalOrientationValues.InferiorToSuperior,
  );
});

Deno.test("itkLpsToAnatomicalOrientation - non-spatial axis returns undefined", () => {
  const cOrientation = itkLpsToAnatomicalOrientation("c");
  assertEquals(cOrientation, undefined);

  const tOrientation = itkLpsToAnatomicalOrientation("t");
  assertEquals(tOrientation, undefined);
});

Deno.test("orientationFromName - resolves LPS and RAS presets", () => {
  assertEquals(orientationFromName("LPS"), LPS);
  assertEquals(orientationFromName("RAS"), RAS);
  // Case-insensitive
  assertEquals(orientationFromName("lps"), LPS);
  assertEquals(orientationFromName("ras"), RAS);
});

Deno.test("orientationFromName - throws on unknown preset", () => {
  assertThrows(
    () => orientationFromName("FOO"),
    Error,
    "Unknown anatomical orientation preset",
  );
});

Deno.test("addAnatomicalOrientationToAxis - adds orientation to axis dict", () => {
  const axisDict = { name: "x", type: "space", unit: "micrometer" };
  const orientation = createAnatomicalOrientation(
    AnatomicalOrientationValues.LeftToRight,
  );

  const result = addAnatomicalOrientationToAxis(axisDict, orientation);

  assertEquals("orientation" in result, true);
  assertEquals(
    (result.orientation as { type: string; value: string }).type,
    "anatomical",
  );
  assertEquals(
    (result.orientation as { type: string; value: string }).value,
    "left-to-right",
  );
});

Deno.test("removeAnatomicalOrientationFromAxis - removes orientation from axis dict", () => {
  const axisDict = {
    name: "x",
    type: "space",
    unit: "micrometer",
    orientation: { type: "anatomical", value: "left-to-right" },
  };

  const result = removeAnatomicalOrientationFromAxis(axisDict);

  assertEquals("orientation" in result, false);
  assertEquals(result.name, "x");
  assertEquals(result.type, "space");
  assertEquals(result.unit, "micrometer");
});

Deno.test("removeAnatomicalOrientationFromAxis - handles axis without orientation", () => {
  const axisDict = { name: "y", type: "space" };

  const result = removeAnatomicalOrientationFromAxis(axisDict);

  assertEquals("orientation" in result, false);
  assertEquals(result.name, "y");
  assertEquals(result.type, "space");
});

Deno.test("AnatomicalOrientationValues enum - correct string values", () => {
  assertEquals(AnatomicalOrientationValues.LeftToRight, "left-to-right");
  assertEquals(AnatomicalOrientationValues.RightToLeft, "right-to-left");
  assertEquals(
    AnatomicalOrientationValues.AnteriorToPosterior,
    "anterior-to-posterior",
  );
  assertEquals(
    AnatomicalOrientationValues.PosteriorToAnterior,
    "posterior-to-anterior",
  );
  assertEquals(
    AnatomicalOrientationValues.InferiorToSuperior,
    "inferior-to-superior",
  );
  assertEquals(
    AnatomicalOrientationValues.SuperiorToInferior,
    "superior-to-inferior",
  );
  // Subject-local layered- and polarized-tissue values (per ome/ngff#528)
  assertEquals(
    AnatomicalOrientationValues.SuperficialToDeep,
    "superficial-to-deep",
  );
  assertEquals(
    AnatomicalOrientationValues.DeepToSuperficial,
    "deep-to-superficial",
  );
  assertEquals(AnatomicalOrientationValues.ApicalToBasal, "apical-to-basal");
  assertEquals(AnatomicalOrientationValues.BasalToApical, "basal-to-apical");
  assertEquals(AnatomicalOrientationValues.ApexToBase, "apex-to-base");
  assertEquals(AnatomicalOrientationValues.BaseToApex, "base-to-apex");
});

Deno.test("LPS coordinate system - correct orientations", () => {
  assertEquals(LPS.x.type, "anatomical");
  assertEquals(LPS.x.value, AnatomicalOrientationValues.RightToLeft);

  assertEquals(LPS.y.type, "anatomical");
  assertEquals(LPS.y.value, AnatomicalOrientationValues.AnteriorToPosterior);

  assertEquals(LPS.z.type, "anatomical");
  assertEquals(LPS.z.value, AnatomicalOrientationValues.InferiorToSuperior);
});

Deno.test("RAS coordinate system - correct orientations", () => {
  assertEquals(RAS.x.type, "anatomical");
  assertEquals(RAS.x.value, AnatomicalOrientationValues.LeftToRight);

  assertEquals(RAS.y.type, "anatomical");
  assertEquals(RAS.y.value, AnatomicalOrientationValues.PosteriorToAnterior);

  assertEquals(RAS.z.type, "anatomical");
  assertEquals(RAS.z.value, AnatomicalOrientationValues.InferiorToSuperior);
});

// --- Tests for anatomicalOrientationToItkDirection ---

Deno.test("anatomicalOrientationToItkDirection - RightToLeft → [1,0,0]", () => {
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.RightToLeft,
    ),
    [1, 0, 0],
  );
});

Deno.test("anatomicalOrientationToItkDirection - LeftToRight → [-1,0,0]", () => {
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.LeftToRight,
    ),
    [-1, 0, 0],
  );
});

Deno.test("anatomicalOrientationToItkDirection - A→P → [0,1,0]", () => {
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.AnteriorToPosterior,
    ),
    [0, 1, 0],
  );
});

Deno.test("anatomicalOrientationToItkDirection - P→A → [0,-1,0]", () => {
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.PosteriorToAnterior,
    ),
    [0, -1, 0],
  );
});

Deno.test("anatomicalOrientationToItkDirection - I→S → [0,0,1]", () => {
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.InferiorToSuperior,
    ),
    [0, 0, 1],
  );
});

Deno.test("anatomicalOrientationToItkDirection - S→I → [0,0,-1]", () => {
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.SuperiorToInferior,
    ),
    [0, 0, -1],
  );
});

Deno.test("anatomicalOrientationToItkDirection - non-LPS returns undefined", () => {
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.DorsalToVentral,
    ),
    undefined,
  );
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.RostralToCaudal,
    ),
    undefined,
  );
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.ProximalToDistal,
    ),
    undefined,
  );
  // Subject-local layered/polarized-tissue values (ome/ngff#528)
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.SuperficialToDeep,
    ),
    undefined,
  );
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.DeepToSuperficial,
    ),
    undefined,
  );
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.ApicalToBasal,
    ),
    undefined,
  );
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.BasalToApical,
    ),
    undefined,
  );
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.ApexToBase,
    ),
    undefined,
  );
  assertEquals(
    anatomicalOrientationToItkDirection(
      AnatomicalOrientationValues.BaseToApex,
    ),
    undefined,
  );
});
