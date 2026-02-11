// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

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
export function defaultCodecs(dataType: string) {
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
