// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
import type { NgffImage } from "./ngff_image.ts";
import type { Metadata } from "./zarr_metadata.ts";
import type { Methods } from "./methods.ts";

export type ChunkSpec =
  | number
  | number[]
  | number[][]
  | Record<string, number | number[] | undefined>;

export interface MultiscalesOptions {
  images: NgffImage[];
  metadata: Metadata;
  scaleFactors: (Record<string, number> | number)[] | undefined;
  method: Methods | undefined;
  chunks: ChunkSpec | undefined;
}

export class Multiscales {
  public readonly images: NgffImage[];
  public readonly metadata: Metadata;
  public readonly scaleFactors: (Record<string, number> | number)[] | undefined;
  public readonly method: Methods | undefined;
  public readonly chunks: ChunkSpec | undefined;

  constructor(options: MultiscalesOptions) {
    this.images = [...options.images];
    this.metadata = { ...options.metadata };
    this.scaleFactors = options.scaleFactors
      ? [...options.scaleFactors]
      : undefined;
    this.method = options.method;
    this.chunks = options.chunks;
  }

  toString(): string {
    const imagesStr = this.images
      .map((img, _i) =>
        `        ${img.toString().replace(/\n/g, "\n        ")}`
      )
      .join(",\n");

    const metadataStr = JSON.stringify(this.metadata, null, 4)
      .split("\n")
      .map((line) => `        ${line}`)
      .join("\n");

    return `Multiscales(
    images=[
${imagesStr}
    ],
    metadata=${metadataStr},
    scale_factors=${JSON.stringify(this.scaleFactors)},
    method=${
      this.method ? `Methods.${this.method.toUpperCase()}` : "undefined"
    },
    chunks=${JSON.stringify(this.chunks)}
)`;
  }
}
