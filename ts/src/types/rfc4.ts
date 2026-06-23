// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * RFC 4 implementation for anatomical orientation in OME-NGFF.
 *
 * This module implements RFC 4 which adds anatomical orientation support
 * to OME-NGFF axes, based on the LinkML schema.
 */

/**
 * Anatomical orientation refers to the specific arrangement and directional
 * alignment of anatomical structures within an imaging dataset. It is crucial
 * for ensuring accurate alignment and comparison of images to anatomical atlases,
 * facilitating consistent analysis and interpretation of biological data.
 */
export enum AnatomicalOrientationValues {
  // Describes the directional orientation from the left side to the right lateral side of an anatomical structure or body
  LeftToRight = "left-to-right",
  // Describes the directional orientation from the right side to the left lateral side of an anatomical structure or body
  RightToLeft = "right-to-left",
  // Describes the directional orientation from the front (anterior) to the back (posterior) of an anatomical structure or body
  AnteriorToPosterior = "anterior-to-posterior",
  // Describes the directional orientation from the back (posterior) to the front (anterior) of an anatomical structure or body
  PosteriorToAnterior = "posterior-to-anterior",
  // Describes the directional orientation from below (inferior) to above (superior) in an anatomical structure or body
  InferiorToSuperior = "inferior-to-superior",
  // Describes the directional orientation from above (superior) to below (inferior) in an anatomical structure or body
  SuperiorToInferior = "superior-to-inferior",
  // Describes the directional orientation from the top/upper (dorsal) to the belly/lower (ventral) in an anatomical structure or body
  DorsalToVentral = "dorsal-to-ventral",
  // Describes the directional orientation from the belly/lower (ventral) to the top/upper (dorsal) in an anatomical structure or body
  VentralToDorsal = "ventral-to-dorsal",
  // Describes the directional orientation from the top/upper (dorsal) to the palm of the hand (palmar) in a body
  DorsalToPalmar = "dorsal-to-palmar",
  // Describes the directional orientation from the palm of the hand (palmar) to the top/upper (dorsal) in a body
  PalmarToDorsal = "palmar-to-dorsal",
  // Describes the directional orientation from the top/upper (dorsal) to the sole of the foot (plantar) in a body
  DorsalToPlantar = "dorsal-to-plantar",
  // Describes the directional orientation from the sole of the foot (plantar) to the top/upper (dorsal) in a body
  PlantarToDorsal = "plantar-to-dorsal",
  // Describes the directional orientation from the nasal (rostral) to the tail (caudal) end of an anatomical structure, typically used in reference to the central nervous system
  RostralToCaudal = "rostral-to-caudal",
  // Describes the directional orientation from the tail (caudal) to the nasal (rostral) end of an anatomical structure, typically used in reference to the central nervous system
  CaudalToRostral = "caudal-to-rostral",
  // Describes the directional orientation from the head (cranial) to the tail (caudal) end of an anatomical structure or body
  CranialToCaudal = "cranial-to-caudal",
  // Describes the directional orientation from the tail (caudal) to the head (cranial) end of an anatomical structure or body
  CaudalToCranial = "caudal-to-cranial",
  // Describes the directional orientation from the center of the body to the periphery of an anatomical structure or limb
  ProximalToDistal = "proximal-to-distal",
  // Describes the directional orientation from the periphery of an anatomical structure or limb to the center of the body
  DistalToProximal = "distal-to-proximal",
  // Describes the directional orientation from the outer surface (superficial) to the inner depth (deep) of a layered tissue such as skin, gut, or cortex
  SuperficialToDeep = "superficial-to-deep",
  // Describes the directional orientation from the inner depth (deep) to the outer surface (superficial) of a layered tissue such as skin, gut, or cortex
  DeepToSuperficial = "deep-to-superficial",
  // Describes the directional orientation from the apical surface (apical) to the basal surface (basal) of an epithelial layer or polarized cell structure
  ApicalToBasal = "apical-to-basal",
  // Describes the directional orientation from the basal surface (basal) to the apical surface (apical) of an epithelial layer or polarized cell structure
  BasalToApical = "basal-to-apical",
  // Describes the directional orientation from the tip (apex) to the broad base (base) of an organ such as the heart or lung
  ApexToBase = "apex-to-base",
  // Describes the directional orientation from the broad base (base) to the tip (apex) of an organ such as the heart or lung
  BaseToApex = "base-to-apex",
}

/**
 * Anatomical orientation specification for spatial axes.
 */
export interface AnatomicalOrientation {
  readonly type: "anatomical";
  readonly value: AnatomicalOrientationValues;
}

/**
 * Create an anatomical orientation object.
 */
export function createAnatomicalOrientation(
  value: AnatomicalOrientationValues,
): AnatomicalOrientation {
  return {
    type: "anatomical",
    value,
  };
}

/**
 * LPS (Left-Posterior-Superior) coordinate system orientations.
 * In LPS, the axes increase from:
 * - X: right-to-left (L = Left)
 * - Y: anterior-to-posterior (P = Posterior)
 * - Z: inferior-to-superior (S = Superior)
 * This is the standard coordinate system used by ITK and many medical imaging applications.
 */
export const LPS: Record<string, AnatomicalOrientation> = {
  x: createAnatomicalOrientation(AnatomicalOrientationValues.RightToLeft),
  y: createAnatomicalOrientation(
    AnatomicalOrientationValues.AnteriorToPosterior,
  ),
  z: createAnatomicalOrientation(
    AnatomicalOrientationValues.InferiorToSuperior,
  ),
};

/**
 * RAS (Right-Anterior-Superior) coordinate system orientations.
 * In RAS, the axes increase from:
 * - X: left-to-right (R = Right)
 * - Y: posterior-to-anterior (A = Anterior)
 * - Z: inferior-to-superior (S = Superior)
 * This coordinate system is commonly used in neuroimaging applications like FreeSurfer and FSL.
 */
export const RAS: Record<string, AnatomicalOrientation> = {
  x: createAnatomicalOrientation(AnatomicalOrientationValues.LeftToRight),
  y: createAnatomicalOrientation(
    AnatomicalOrientationValues.PosteriorToAnterior,
  ),
  z: createAnatomicalOrientation(
    AnatomicalOrientationValues.InferiorToSuperior,
  ),
};

/**
 * Preset anatomical orientation coordinate systems, keyed by lowercased name.
 */
const ORIENTATION_PRESETS: Record<
  string,
  Record<string, AnatomicalOrientation>
> = {
  lps: LPS,
  ras: RAS,
};

/**
 * Resolve an anatomical orientation preset name to per-axis orientations.
 *
 * @param name - A preset coordinate-system name: "LPS" or "RAS" (case-insensitive)
 * @returns A mapping of spatial axis name to AnatomicalOrientation
 * @throws Error if the name is not a recognized preset
 */
export function orientationFromName(
  name: string,
): Record<string, AnatomicalOrientation> {
  const preset = ORIENTATION_PRESETS[name.toLowerCase()];
  if (preset === undefined) {
    const supported = Object.keys(ORIENTATION_PRESETS)
      .map((p) => p.toUpperCase())
      .join(", ");
    throw new Error(
      `Unknown anatomical orientation preset: "${name}". ` +
        `Supported presets: ${supported}.`,
    );
  }
  return { ...preset };
}

/**
 * Convert ITK LPS coordinate system to anatomical orientation.
 *
 * ITK uses the LPS (Left-Posterior-Superior) coordinate system by default.
 * In LPS, the axes increase from:
 * - X: right-to-left (L = Left)
 * - Y: anterior-to-posterior (P = Posterior)
 * - Z: inferior-to-superior (S = Superior)
 *
 * @param axisName - The axis name ('x', 'y', or 'z')
 * @returns The corresponding anatomical orientation, or undefined for non-spatial axes
 */
export function itkLpsToAnatomicalOrientation(
  axisName: string,
): AnatomicalOrientation | undefined {
  return LPS[axisName];
}

/**
 * Maximum number of spatial dimensions (for ITK LPS coordinate system).
 * LPS has 3 axes: Left/Right (X), Anterior/Posterior (Y),
 * Inferior/Superior (Z).
 */
const MAX_SPATIAL_DIMENSIONS = 3;

/**
 * LPS orientation pairs: positive and negative orientations for each
 * physical axis in the LPS coordinate system.
 *
 * Each row is `[positiveComponentOrientation, negativeComponentOrientation]`,
 * where "positive component" (positive direction cosine value) indicates
 * the direction along that axis, and "negative component" (negative
 * direction cosine value) indicates the opposite direction.
 *
 * - Row 0 (L/R): positive component → RightToLeft
 *   (positive direction points Left) / negative component → LeftToRight
 *   (negative direction points Right)
 * - Row 1 (A/P): positive component → AnteriorToPosterior
 *   (positive direction points Posterior) / negative component →
 *   PosteriorToAnterior (negative direction points Anterior)
 * - Row 2 (I/S): positive component → InferiorToSuperior
 *   (positive direction points Superior) / negative component →
 *   SuperiorToInferior (negative direction points Inferior)
 */
const LPS_ORIENTATION_PAIRS: [
  AnatomicalOrientationValues,
  AnatomicalOrientationValues,
][] = [
  [
    AnatomicalOrientationValues.RightToLeft,
    AnatomicalOrientationValues.LeftToRight,
  ],
  [
    AnatomicalOrientationValues.AnteriorToPosterior,
    AnatomicalOrientationValues.PosteriorToAnterior,
  ],
  [
    AnatomicalOrientationValues.InferiorToSuperior,
    AnatomicalOrientationValues.SuperiorToInferior,
  ],
];

/**
 * Determine the anatomical orientation of a single image axis from its
 * direction cosine vector in ITK LPS physical space.
 *
 * The direction vector is a column of the ITK direction matrix. Its
 * dominant component (largest absolute value) determines which physical
 * axis (L/R, A/P, or S/I) this image axis aligns with, and the sign
 * of that component determines the direction.
 *
 * In LPS physical space:
 * - Component 0: L/R axis (positive = Right→Left, negative = Left→Right)
 * - Component 1: A/P axis (positive = Anterior→Posterior, negative = Posterior→Anterior)
 * - Component 2: I/S axis (positive = Inferior→Superior, negative = Superior→Inferior)
 *
 * For 2D images, the direction column will have only 2 components, with the
 * third component implicitly zero.
 *
 * @param directionColumn - Direction cosine vector for the image axis.
 *   Can be 2-element (for 2D images) or 3-element (for 3D images).
 *   `[lps_x, lps_y]` or `[lps_x, lps_y, lps_z]`
 *   Must have at least 2 elements.
 * @returns The anatomical orientation corresponding to the dominant
 *   direction of this axis
 * @throws Error if directionColumn has fewer than 2 elements
 */
export function itkDirectionToAnatomicalOrientation(
  directionColumn: readonly number[],
): AnatomicalOrientation {
  if (directionColumn.length < 2) {
    throw new Error(
      `Direction column must have at least 2 elements for 2D/3D images, got ${directionColumn.length}`,
    );
  }
  // Find the dominant physical axis (largest absolute component)
  let dominantAxis = 0;
  let maxAbs = Math.abs(directionColumn[0]);
  const n = Math.min(directionColumn.length, MAX_SPATIAL_DIMENSIONS);
  for (let i = 1; i < n; i++) {
    const absVal = Math.abs(directionColumn[i]);
    if (absVal > maxAbs) {
      maxAbs = absVal;
      dominantAxis = i;
    }
  }

  // Positive component → positive LPS direction, negative → opposite
  const pair = LPS_ORIENTATION_PAIRS[dominantAxis];
  const value = directionColumn[dominantAxis] >= 0 ? pair[0] : pair[1];
  return createAnatomicalOrientation(value);
}

/**
 * Convert an anatomical orientation to an ITK direction cosine vector.
 *
 * This is the reverse of {@link itkDirectionToAnatomicalOrientation}.
 * Given an anatomical orientation value, it returns the unit direction
 * column vector in ITK LPS physical space.
 *
 * Only the six LPS-compatible orientations are supported.  For any other
 * orientation (e.g. dorsal, palmar, rostral, etc.) the function returns
 * `undefined`, signalling that the caller should fall back to an identity
 * direction matrix.
 *
 * @param orientation - The anatomical orientation value.
 * @returns A 3-element unit direction column vector `[lps_x, lps_y, lps_z]`,
 *   or `undefined` if the orientation is not one of the six LPS-compatible
 *   values.
 */
export function anatomicalOrientationToItkDirection(
  orientation: AnatomicalOrientationValues,
): number[] | undefined {
  for (
    let axisIndex = 0;
    axisIndex < LPS_ORIENTATION_PAIRS.length;
    axisIndex++
  ) {
    const [pos, neg] = LPS_ORIENTATION_PAIRS[axisIndex];
    if (orientation === pos) {
      const col = [0, 0, 0];
      col[axisIndex] = 1;
      return col;
    }
    if (orientation === neg) {
      const col = [0, 0, 0];
      col[axisIndex] = -1;
      return col;
    }
  }
  return undefined;
}

/**
 * Add anatomical orientation to an axis object.
 *
 * @param axisDict - The axis object to modify
 * @param orientation - The anatomical orientation to add
 * @returns The modified axis object
 */
export function addAnatomicalOrientationToAxis(
  axisDict: Record<string, unknown>,
  orientation: AnatomicalOrientation,
): Record<string, unknown> {
  return {
    ...axisDict,
    orientation: {
      type: orientation.type,
      value: orientation.value,
    },
  };
}

/**
 * Remove anatomical orientation from an axis object.
 *
 * @param axisDict - The axis object to modify
 * @returns The modified axis object
 */
export function removeAnatomicalOrientationFromAxis(
  axisDict: Record<string, unknown>,
): Record<string, unknown> {
  const { orientation: _orientation, ...rest } = axisDict;
  return rest;
}
