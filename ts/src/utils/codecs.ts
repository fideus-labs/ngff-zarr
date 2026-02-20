// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Codec descriptor used in zarr v3 array metadata.
 *
 * @see {@link defaultCodecs} for the standard blosc+zstd pipeline.
 * @see {@link bytesOnlyCodecs} for an uncompressed pipeline.
 */
export type ZarrCodec = {
  name: string;
  configuration: Record<string, unknown>;
};

/**
 * Return the byte size for a zarr v3 data type string.
 *
 * The Zarr v3 blosc codec spec requires an explicit `typesize` when
 * shuffling is enabled. This helper maps a `DataType` name to the
 * number of bytes per element so that `defaultCodecs()` can populate
 * the field automatically.
 */
export function typeSizeForDtype(dtype: string): number {
  const sizes: Record<string, number> = {
    bool: 1,
    int8: 1,
    uint8: 1,
    int16: 2,
    uint16: 2,
    float16: 2,
    int32: 4,
    uint32: 4,
    float32: 4,
    int64: 8,
    uint64: 8,
    float64: 8,
  };
  return sizes[dtype] ?? 1;
}

/**
 * Build the default zarr v3 codec pipeline for a given data type.
 *
 * Includes the required `bytes` array-to-bytes codec (little-endian)
 * and `blosc` bytes-to-bytes compression with zstd.
 *
 * The `shuffle` value uses the **string** enum required by the Zarr v3
 * blosc codec specification (`"shuffle"`, `"noshuffle"`, or
 * `"bitshuffle"`) rather than the legacy integer convention used by
 * Zarr v2 / the C blosc library.
 *
 * Without explicit codecs, zarrita writes `"codecs": []` in zarr.json,
 * which is not spec-compliant and causes errors in other zarr v3
 * implementations (e.g., Python zarr-python).
 */
export function defaultCodecs(dataType: string): ZarrCodec[] {
  return [
    { name: "bytes", configuration: { endian: "little" } },
    {
      name: "blosc",
      configuration: {
        cname: "zstd",
        clevel: 5,
        shuffle: "shuffle",
        typesize: typeSizeForDtype(dataType),
        blocksize: 0,
      },
    },
  ];
}

/**
 * Build an uncompressed zarr v3 codec pipeline.
 *
 * Returns only the required `bytes` array-to-bytes codec (little-endian)
 * with no bytes-to-bytes compression. This is useful when zarr arrays
 * are ephemeral (e.g., in-memory multiscale pyramids that will be
 * immediately re-encoded into another format like OME-TIFF) and the
 * blosc compress/decompress round-trip would be wasted work.
 */
export function bytesOnlyCodecs(): ZarrCodec[] {
  return [{ name: "bytes", configuration: { endian: "little" } }];
}

/**
 * Supported codec name strings accepted by {@link codecFromName}.
 *
 * Includes bare {@link "blosc"} (defaults to lz4 compression) as well
 * as prefixed variants like {@link "blosc:zstd"}.
 */
export const AVAILABLE_CODECS = [
  "none",
  "gzip",
  "lz4",
  "zstd",
  "blosc",
  "blosc:blosclz",
  "blosc:lz4",
  "blosc:lz4hc",
  "blosc:snappy",
  "blosc:zlib",
  "blosc:zstd",
] as const;

/** Union of recognised codec name strings. */
export type CodecName = (typeof AVAILABLE_CODECS)[number];

/**
 * Build a zarr v3 codec pipeline from a human-readable codec name.
 *
 * This is the TypeScript equivalent of the Python
 * ``ngff_zarr.codecs.codec_from_name()`` helper. Unlike the Python
 * version (which returns a single *numcodecs* object), this returns a
 * complete ``ZarrCodec[]`` pipeline including the required ``bytes``
 * array-to-bytes codec.
 *
 * @param name - One of the strings in {@link AVAILABLE_CODECS}.
 * @param dataType - Zarr v3 data type string (e.g. ``"float32"``),
 *   needed to set the blosc ``typesize`` field.
 * @param level - Optional compression level.  When omitted a sensible
 *   default is used (gzip=6, zstd=3, blosc=5).
 * @returns A codec pipeline suitable for ``zarr.create()``.
 * @throws {Error} If *name* is not recognised.
 */
export function codecFromName(
  name: string,
  dataType: string,
  level?: number,
): ZarrCodec[] {
  const bytes: ZarrCodec = {
    name: "bytes",
    configuration: { endian: "little" },
  };

  if (name === "none") {
    return [bytes];
  }

  if (name === "gzip") {
    return [bytes, { name: "gzip", configuration: { level: level ?? 6 } }];
  }

  if (name === "zstd") {
    return [bytes, { name: "zstd", configuration: { level: level ?? 3 } }];
  }

  if (name === "lz4") {
    return [bytes, { name: "lz4", configuration: {} }];
  }

  if (name === "blosc" || name.startsWith("blosc:")) {
    let cname: string;

    if (name === "blosc") {
      // Default blosc variant when none is specified explicitly.
      cname = "lz4";
    } else {
      const parts = name.split(":");
      cname = parts[1] ?? "";

      // Reject empty or unknown blosc variants to avoid silently writing
      // unsupported Zarr metadata.
      if (cname === "" || !AVAILABLE_CODECS.includes(name as CodecName)) {
        throw new Error(
          `Unknown codec: "${name}". Available: ${AVAILABLE_CODECS.join(", ")}`,
        );
      }
    }
    return [
      bytes,
      {
        name: "blosc",
        configuration: {
          cname,
          clevel: level ?? 5,
          shuffle: "shuffle",
          typesize: typeSizeForDtype(dataType),
          blocksize: 0,
        },
      },
    ];
  }

  throw new Error(
    `Unknown codec: "${name}". Available: ${AVAILABLE_CODECS.join(", ")}`,
  );
}
