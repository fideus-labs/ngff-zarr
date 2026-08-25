// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * The `scale-length-mismatch` rule and v0.6 multiscale-level transforms.
 *
 * At v0.6 a multiscale-level `coordinateTransformations` entry maps between two
 * coordinate systems referenced by name, and `spec/0.6/schemas/image.schema`
 * requires a `name` on both its `input` and its `output`. Its dimensionality
 * follows those two named systems. `metadata.axes` holds the intrinsic system's
 * axes instead, so measuring such a transform against `metadata.axes.length`
 * rejects a legal document.
 *
 * The Python port drops these transforms in `_flat_model` for the same reason,
 * covered by `test_top_level_inter_system_transform_is_not_length_checked` in
 * `py/test/test_structural_validation_v06.py`. This suite is its counterpart,
 * so the two ports reach the same verdict on the same document. It also pins
 * that per-dataset transforms, which do map an array to the intrinsic system,
 * keep being measured at v0.6.
 */

import { assertThrows } from "@std/assert";
import {
  SpecRule,
  validateScaleLength,
  validateStructural,
  ValidationError,
} from "../src/utils/structural_validation.ts";
import {
  createScale,
  type CoordinateSystem,
  type Metadata,
} from "../src/types/zarr_metadata.ts";

/** Three intrinsic axes, plus two named 2D systems a transform maps between. */
function coordinateSystems(): CoordinateSystem[] {
  return [
    {
      name: "intrinsic",
      axes: [
        { name: "z", type: "space", unit: undefined },
        { name: "y", type: "space", unit: undefined },
        { name: "x", type: "space", unit: undefined },
      ],
    },
    {
      name: "slide",
      axes: [
        { name: "y", type: "space", unit: undefined },
        { name: "x", type: "space", unit: undefined },
      ],
    },
    {
      name: "stage",
      axes: [
        { name: "y", type: "space", unit: undefined },
        { name: "x", type: "space", unit: undefined },
      ],
    },
  ];
}

/**
 * A v0.6 document whose multiscale-level transform has two components against
 * three intrinsic axes, which is what the two named systems it joins declare.
 */
function v06MetadataWithNamedSystemTransform(): Metadata {
  const systems = coordinateSystems();
  return {
    axes: systems[0].axes,
    coordinateSystems: systems,
    datasets: [
      {
        path: "0",
        coordinateTransformations: [createScale([1.0, 1.0, 1.0])],
      },
      {
        path: "1",
        coordinateTransformations: [createScale([1.0, 2.0, 2.0])],
      },
    ],
    coordinateTransformations: [createScale([0.5, 0.5])],
    omero: undefined,
    name: "image",
    version: "0.6",
  };
}

Deno.test("a v0.6 transform between two named systems is not measured against the intrinsic axes", () => {
  validateScaleLength(v06MetadataWithNamedSystemTransform());
  validateStructural(v06MetadataWithNamedSystemTransform());
});

Deno.test("a v0.4 global transform is still measured against the axis count", () => {
  const { coordinateSystems: _dropped, ...flat } =
    v06MetadataWithNamedSystemTransform();
  const metadata: Metadata = { ...flat, version: "0.4" };

  assertThrows(
    () => validateScaleLength(metadata),
    ValidationError,
    "Global",
  );
});

Deno.test("a v0.6 per-dataset transform is still measured against the intrinsic axes", () => {
  const metadata = v06MetadataWithNamedSystemTransform();
  metadata.datasets[1].coordinateTransformations = [createScale([2.0, 2.0])];

  const error = assertThrows(
    () => validateScaleLength(metadata),
    ValidationError,
    "Dataset 1",
  ) as ValidationError;
  if (error.rule !== SpecRule.ScaleLengthMismatch) {
    throw new Error(`expected scale-length-mismatch, got ${error.rule}`);
  }
});
