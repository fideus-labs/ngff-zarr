// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Default zarr v3 codecs for array creation.
 *
 * Includes the required `bytes` array-to-bytes codec (little-endian)
 * and `blosc` bytes-to-bytes compression with zstd.
 *
 * Without explicit codecs, zarrita writes `"codecs": []` in zarr.json,
 * which is not spec-compliant and causes errors in other zarr v3
 * implementations (e.g., Python zarr-python).
 */
export const DEFAULT_CODECS = [
  { name: "bytes", configuration: { endian: "little" } },
  {
    name: "blosc",
    configuration: { cname: "zstd", clevel: 5, shuffle: 1, blocksize: 0 },
  },
] as const;
