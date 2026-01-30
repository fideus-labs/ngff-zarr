// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
/**
 * RFC-9: Zipped OME-Zarr (.ozx) support
 *
 * This module provides utilities for writing OME-Zarr hierarchies
 * to ZIP archives according to RFC-9 specification.
 *
 * @see https://ngff.openmicroscopy.org/rfc/9/index.html
 */

import { zipSync } from "fflate";

/** MemoryStore type for zarr data */
export type MemoryStore = Map<string, Uint8Array>;

/**
 * Check if a path refers to a .ozx file.
 *
 * @param path - Path to check
 * @returns True if path ends with .ozx extension
 */
export function isOzxPath(path: string): boolean {
  return path.endsWith(".ozx");
}

/**
 * Order files for RFC-9 compliance.
 *
 * According to RFC-9:
 * - Root-level zarr.json must be the first entry
 * - Other zarr.json files should follow in breadth-first order
 * - Data files come after all zarr.json files
 *
 * @param keys - Array of file paths from the store
 * @returns Ordered array of file paths
 */
export function orderFilesForRfc9(keys: string[]): string[] {
  const zarrJsons = keys.filter((k) => k.endsWith("zarr.json"));
  const others = keys.filter((k) => !k.endsWith("zarr.json"));

  // Root zarr.json first
  const rootIdx = zarrJsons.indexOf("zarr.json");
  if (rootIdx > 0) {
    zarrJsons.splice(rootIdx, 1);
    zarrJsons.unshift("zarr.json");
  }

  // Sort by depth for breadth-first order
  zarrJsons.sort((a, b) => {
    if (a === "zarr.json") return -1;
    if (b === "zarr.json") return 1;
    return a.split("/").length - b.split("/").length;
  });

  return [...zarrJsons, ...others.sort()];
}

/**
 * Add a ZIP comment to existing ZIP data.
 *
 * fflate doesn't support ZIP comments natively, so we manually
 * modify the End of Central Directory (EOCD) record to add one.
 *
 * @param zipData - Original ZIP data
 * @param comment - Comment string to add (UTF-8)
 * @returns New ZIP data with comment
 */
function addZipComment(zipData: Uint8Array, comment: string): Uint8Array {
  const commentBytes = new TextEncoder().encode(comment);

  if (commentBytes.length > 65535) {
    throw new Error("ZIP comment too long (max 65535 bytes)");
  }

  // Find EOCD signature (0x06054b50) searching from end
  // EOCD is at least 22 bytes, so start searching from length - 22
  let eocdOffset = -1;
  for (let i = zipData.length - 22; i >= 0; i--) {
    if (
      zipData[i] === 0x50 &&
      zipData[i + 1] === 0x4b &&
      zipData[i + 2] === 0x05 &&
      zipData[i + 3] === 0x06
    ) {
      eocdOffset = i;
      break;
    }
  }

  if (eocdOffset === -1) {
    throw new Error("Invalid ZIP: End of Central Directory not found");
  }

  // Note: We could read current comment length at EOCD + 20, but we
  // always replace any existing comment, so we just need the EOCD position.

  // Calculate the size of ZIP without the current comment
  const zipWithoutComment = eocdOffset + 22;

  // Create new array: ZIP without old comment + new comment
  const result = new Uint8Array(zipWithoutComment + commentBytes.length);

  // Copy ZIP data up to and including EOCD (excluding old comment)
  result.set(zipData.subarray(0, zipWithoutComment));

  // Update comment length field (little-endian at EOCD + 20)
  result[eocdOffset + 20] = commentBytes.length & 0xff;
  result[eocdOffset + 21] = (commentBytes.length >> 8) & 0xff;

  // Append new comment
  result.set(commentBytes, zipWithoutComment);

  return result;
}

/**
 * Read the OME-Zarr version from ZIP comment.
 *
 * @param zipData - ZIP file data
 * @returns OME-Zarr version string if found, null otherwise
 */
export function readOzxVersion(zipData: Uint8Array): string | null {
  // Find EOCD signature searching from end
  let eocdOffset = -1;
  for (let i = zipData.length - 22; i >= 0; i--) {
    if (
      zipData[i] === 0x50 &&
      zipData[i + 1] === 0x4b &&
      zipData[i + 2] === 0x05 &&
      zipData[i + 3] === 0x06
    ) {
      eocdOffset = i;
      break;
    }
  }

  if (eocdOffset === -1) {
    return null;
  }

  // Get comment length (little-endian uint16 at EOCD + 20)
  const commentLength = zipData[eocdOffset + 20] |
    (zipData[eocdOffset + 21] << 8);

  if (commentLength === 0) {
    return null;
  }

  // Extract comment bytes
  const commentStart = eocdOffset + 22;
  const commentBytes = zipData.subarray(
    commentStart,
    commentStart + commentLength,
  );

  try {
    const commentStr = new TextDecoder("utf-8").decode(commentBytes);
    const commentObj = JSON.parse(commentStr);

    if (commentObj?.ome?.version) {
      return commentObj.ome.version;
    }
  } catch {
    // Invalid JSON or encoding
  }

  return null;
}

/**
 * Check if a file in the ZIP uses compression.
 *
 * @param zipData - ZIP file data
 * @param filename - Name of the file to check
 * @returns Compression method (0 = stored, 8 = deflate, etc.)
 */
export function getZipFileCompressionMethod(
  zipData: Uint8Array,
  filename: string,
): number | null {
  // Search for local file header with matching filename
  const filenameBytes = new TextEncoder().encode(filename);
  let offset = 0;

  while (offset < zipData.length - 30) {
    // Check for local file header signature (0x04034b50)
    if (
      zipData[offset] === 0x50 &&
      zipData[offset + 1] === 0x4b &&
      zipData[offset + 2] === 0x03 &&
      zipData[offset + 3] === 0x04
    ) {
      // Get filename length (little-endian at offset + 26)
      const nameLength = zipData[offset + 26] | (zipData[offset + 27] << 8);

      // Get extra field length (little-endian at offset + 28)
      const extraLength = zipData[offset + 28] | (zipData[offset + 29] << 8);

      // Extract filename
      const nameStart = offset + 30;
      const nameEnd = nameStart + nameLength;

      if (nameEnd <= zipData.length) {
        const nameBytes = zipData.subarray(nameStart, nameEnd);

        // Compare filenames
        if (
          nameBytes.length === filenameBytes.length &&
          nameBytes.every((b, i) => b === filenameBytes[i])
        ) {
          // Found the file, return compression method (at offset + 8)
          return zipData[offset + 8] | (zipData[offset + 9] << 8);
        }
      }

      // Get compressed size to skip to next header
      const compressedSize = zipData[offset + 18] |
        (zipData[offset + 19] << 8) |
        (zipData[offset + 20] << 16) |
        (zipData[offset + 21] << 24);

      offset = nameEnd + extraLength + compressedSize;
    } else {
      offset++;
    }
  }

  return null;
}

/**
 * Get the list of files in a ZIP archive in order.
 *
 * @param zipData - ZIP file data
 * @returns Array of filenames in the order they appear in the central directory
 */
export function getZipFileList(zipData: Uint8Array): string[] {
  const files: string[] = [];

  // Find EOCD to get central directory location
  let eocdOffset = -1;
  for (let i = zipData.length - 22; i >= 0; i--) {
    if (
      zipData[i] === 0x50 &&
      zipData[i + 1] === 0x4b &&
      zipData[i + 2] === 0x05 &&
      zipData[i + 3] === 0x06
    ) {
      eocdOffset = i;
      break;
    }
  }

  if (eocdOffset === -1) {
    return files;
  }

  // Get central directory offset (little-endian uint32 at EOCD + 16)
  let cdOffset = zipData[eocdOffset + 16] |
    (zipData[eocdOffset + 17] << 8) |
    (zipData[eocdOffset + 18] << 16) |
    (zipData[eocdOffset + 19] << 24);

  // Get number of entries (little-endian uint16 at EOCD + 10)
  const numEntries = zipData[eocdOffset + 10] | (zipData[eocdOffset + 11] << 8);

  // Parse central directory entries
  for (let i = 0; i < numEntries && cdOffset < eocdOffset; i++) {
    // Check for central directory file header signature (0x02014b50)
    if (
      zipData[cdOffset] !== 0x50 ||
      zipData[cdOffset + 1] !== 0x4b ||
      zipData[cdOffset + 2] !== 0x01 ||
      zipData[cdOffset + 3] !== 0x02
    ) {
      break;
    }

    // Get filename length (little-endian at offset + 28)
    const nameLength = zipData[cdOffset + 28] | (zipData[cdOffset + 29] << 8);

    // Get extra field length (little-endian at offset + 30)
    const extraLength = zipData[cdOffset + 30] | (zipData[cdOffset + 31] << 8);

    // Get comment length (little-endian at offset + 32)
    const commentLength = zipData[cdOffset + 32] |
      (zipData[cdOffset + 33] << 8);

    // Extract filename
    const nameStart = cdOffset + 46;
    const nameEnd = nameStart + nameLength;

    if (nameEnd <= zipData.length) {
      const filename = new TextDecoder("utf-8").decode(
        zipData.subarray(nameStart, nameEnd),
      );
      files.push(filename);
    }

    // Move to next entry
    cdOffset = nameEnd + extraLength + commentLength;
  }

  return files;
}

export interface MemoryStoreToZipOptions {
  /** OME-Zarr version string (default: "0.5") */
  version?: string;
}

/**
 * Normalize a store key by removing leading slashes.
 * zarrita uses leading slashes (e.g., "/zarr.json") but ZIP files
 * should use relative paths (e.g., "zarr.json").
 */
function normalizeKey(key: string): string {
  return key.startsWith("/") ? key.slice(1) : key;
}

/**
 * Convert a MemoryStore to ZIP data following RFC-9 specification.
 *
 * According to RFC-9:
 * - Root-level zarr.json must be the first entry
 * - Other zarr.json files should follow in breadth-first order
 * - ZIP-level compression should be disabled (ZIP_STORED)
 * - A comment with OME-Zarr version should be added
 *
 * @param store - MemoryStore containing zarr data
 * @param options - Options for ZIP creation
 * @returns ZIP file data as Uint8Array
 */
export function memoryStoreToZip(
  store: MemoryStore,
  options: MemoryStoreToZipOptions = {},
): Uint8Array {
  const version = options.version ?? "0.5";

  // Get all keys, normalize them (remove leading slashes from zarrita),
  // and order them for RFC-9 compliance
  const keys = Array.from(store.keys()).map(normalizeKey);
  const orderedKeys = orderFilesForRfc9(keys);

  // Build files object in the correct order
  // Note: JavaScript objects maintain insertion order for string keys
  const files: Record<string, Uint8Array> = {};
  for (const key of orderedKeys) {
    // Try both normalized and original key (with leading slash) for lookup
    const data = store.get(key) ?? store.get("/" + key);
    if (data) {
      files[key] = data;
    }
  }

  // Create ZIP with no compression (level: 0 = ZIP_STORED equivalent)
  const zipData = zipSync(files, { level: 0 });

  // Add OME-Zarr version comment as per RFC-9
  const comment = JSON.stringify({ ome: { version } });
  const zipWithComment = addZipComment(zipData, comment);

  return zipWithComment;
}
