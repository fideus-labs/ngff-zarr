// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import type { AxesType, SupportedDims, Units } from "./units.ts";

export interface Axis {
  name: SupportedDims;
  type: AxesType;
  unit: Units | undefined;
}

export interface Identity {
  type: "identity";
}

export interface Scale {
  scale: number[];
  type: "scale";
}

export interface Translation {
  translation: number[];
  type: "translation";
}

export type Transform = Scale | Translation;

export interface Dataset {
  path: string;
  coordinateTransformations: Transform[];
}

export interface OmeroWindow {
  min?: number;
  max?: number;
  start?: number;
  end?: number;
}

export interface OmeroChannel {
  color: string;
  window: OmeroWindow;
  label?: string;
  active?: boolean;
}

export interface Omero {
  channels: OmeroChannel[];
  version?: string;
}

export interface MethodMetadata {
  description: string;
  method: string;
  version: string;
}

export interface Metadata {
  axes: Axis[];
  datasets: Dataset[];
  coordinateTransformations: Transform[] | undefined;
  omero: Omero | undefined;
  name: string;
  version: string;
  type?: string;
  metadata?: MethodMetadata;
}

export function validateColor(color: string): void {
  if (!/^[0-9A-Fa-f]{6}$/.test(color)) {
    throw new Error(`Invalid color '${color}'. Must be 6 hex digits.`);
  }
}

export function createScale(scale: number[]): Scale {
  return { scale: [...scale], type: "scale" };
}

export function createTranslation(translation: number[]): Translation {
  return { translation: [...translation], type: "translation" };
}

export function createIdentity(): Identity {
  return { type: "identity" };
}
