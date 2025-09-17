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
 * Check if RFC 4 is enabled in the list of enabled RFCs.
 */
export function isRfc4Enabled(enabledRfcs?: number[]): boolean {
  return enabledRfcs !== undefined && enabledRfcs.includes(4);
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
