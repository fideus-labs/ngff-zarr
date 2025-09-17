// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import * as zarr from "zarrita";
import type { Units } from "./units.ts";
import type { AnatomicalOrientation } from "./rfc4.ts";

export type ComputedCallback = () => void;

export interface NgffImageOptions {
  data: zarr.Array<zarr.DataType, zarr.Readable>;
  dims: string[];
  scale: Record<string, number>;
  translation: Record<string, number>;
  name: string | undefined;
  axesUnits: Record<string, Units> | undefined;
  axesOrientations?: Record<string, AnatomicalOrientation> | undefined;
  computedCallbacks: ComputedCallback[] | undefined;
}

export class NgffImage {
  public readonly data: zarr.Array<zarr.DataType, zarr.Readable>;
  public readonly dims: string[];
  public readonly scale: Record<string, number>;
  public readonly translation: Record<string, number>;
  public readonly name: string;
  public readonly axesUnits: Record<string, Units> | undefined;
  public readonly axesOrientations:
    | Record<string, AnatomicalOrientation>
    | undefined;
  public readonly computedCallbacks: ComputedCallback[];

  constructor(options: NgffImageOptions) {
    this.data = options.data;
    this.dims = [...options.dims];
    this.scale = { ...options.scale };
    this.translation = { ...options.translation };
    this.name = options.name ?? "image";
    this.axesUnits = options.axesUnits ? { ...options.axesUnits } : undefined;
    this.axesOrientations = options.axesOrientations
      ? { ...options.axesOrientations }
      : undefined;
    this.computedCallbacks = [...(options.computedCallbacks ?? [])];
  }

  toString(): string {
    const axesUnitsStr = this.axesUnits
      ? JSON.stringify(this.axesUnits)
      : "None";

    const axesOrientationsStr = this.axesOrientations
      ? JSON.stringify(this.axesOrientations)
      : "None";

    // Create array string representation using zarr.Array properties
    const path = this.data.path || this.name;
    const chunks = this.data.chunks.join(", ");

    const arrayStr = `Array(name=${path}, shape=(${
      this.data.shape.join(
        ", ",
      )
    }), dtype=${this.data.dtype}, chunksize=(${chunks}), chunktype=TypedArray)`;

    return `NgffImage(
    data=${arrayStr},
    dims=[${this.dims.map((d) => `'${d}'`).join(", ")}],
    scale=${JSON.stringify(this.scale)},
    translation=${JSON.stringify(this.translation)},
    name='${this.name}',
    axes_units=${axesUnitsStr},
    axes_orientations=${axesOrientationsStr},
    computed_callbacks=[${this.computedCallbacks.length} callbacks]
)`;
  }
}
