// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Environment-agnostic core of `upgradeOmeZarr`, shared by the Node
 * (`upgrade_ome_zarr.ts`) and browser (`upgrade_ome_zarr-browser.ts`) wrappers.
 *
 * The version-detection, no-op / same-format / cross-format decision, and the
 * in-place root-metadata rewrite are identical across environments; only the
 * concrete `fromOmeZarr` / `toOmeZarr` implementations and the writable-store
 * resolution differ (the browser build ships no filesystem store). Those three
 * pieces are injected via {@link UpgradeOmeZarrDeps} so this module stays free
 * of any Node-specific imports and safe to bundle for the browser.
 */

import * as zarr from "zarrita";

import type { NgffMultiscales } from "../types/multiscales.ts";
import type { MemoryStore } from "./rfc9_zip.ts";
import { detectVersion } from "../utils/parse_metadata.ts";
import {
  type ImageVersion,
  isV06Version,
  V06_ONDISK_VERSION,
} from "../types/supported_versions.ts";
import { buildRootAttributes } from "./to_ngff_zarr_ozx_common.ts";

/** Stores/paths `upgradeOmeZarr` can read from. */
export type UpgradeInput =
  | string
  | MemoryStore
  | zarr.FetchStore
  | zarr.Readable;

export interface UpgradeOmeZarrOptions {
  /**
   * Destination store or path. Omit (or pass the same store as `input`) for an
   * in-place upgrade; pass a distinct store to write an upgraded copy there.
   * A `FetchStore` is read-only and cannot be a write destination, so it is not
   * accepted here (it remains valid as an `input`).
   */
  output?: string | MemoryStore;
  /** Target OME-Zarr specification version. Defaults to `"0.6"`. */
  version?: ImageVersion;
  /** Validate the source metadata against the NGFF schema while reading. */
  validate?: boolean;
  /**
   * Write-to-new-store only: overwrite pre-existing data at `output`
   * (default `true`). Ignored for in-place upgrades, which never overwrite
   * array data.
   */
  overwrite?: boolean;
}

/** The environment-specific pieces injected into {@link upgradeOmeZarrImpl}. */
export interface UpgradeOmeZarrDeps {
  fromOmeZarr: (
    store: UpgradeInput,
    options?: {
      validate?: boolean;
      version?: ImageVersion;
    },
  ) => Promise<NgffMultiscales>;
  toOmeZarr: (
    store: string | MemoryStore | zarr.FetchStore,
    multiscales: NgffMultiscales,
    options?: {
      overwrite?: boolean;
      version?: ImageVersion;
    },
  ) => Promise<void>;
  /**
   * Resolve `input` to a writable store for an in-place rewrite. Must reject a
   * read-only input (e.g. FetchStore / HTTP URL) — and, in the browser, a local
   * path — with an actionable message directing the user to pass `output`.
   */
  resolveWritableStore: (input: UpgradeInput) => Promise<MemoryStore>;
}

/** OME-Zarr < 0.5 lives in Zarr v2; 0.5 and later live in Zarr v3. */
function zarrFormatForVersion(version: string): 2 | 3 {
  const [major, minor] = version.split(".").map((part) => Number(part));
  return major === 0 && minor < 5 ? 2 : 3;
}

/**
 * Collapse the v0.6 family (including its on-disk pre-release tags) to `"0.6"`
 * so a store tagged with one compares equal to a requested version of `"0.6"`.
 */
function normalizeVersion(version: string): string {
  return isV06Version(version) ? "0.6" : version;
}

/** The raw version string a store records, before any family collapsing. */
function onDiskVersion(rootAttrs: Record<string, unknown>): string | undefined {
  const ome = rootAttrs.ome as Record<string, unknown> | undefined;
  if (ome && typeof ome.version === "string") return ome.version;
  const multiscales = rootAttrs.multiscales as
    | Array<Record<string, unknown>>
    | undefined;
  const version = multiscales?.[0]?.version;
  return typeof version === "string" ? version : undefined;
}

/**
 * The `ome.version` string a store written at `version` carries.
 *
 * Only `0.6` differs from what the caller passes: it is tagged with the
 * pre-release the bundled schemas carry. `0.9.dev1` is itself the on-disk
 * string and is returned unchanged, matching the Python port's
 * `_ondisk_version_for`.
 */
function onDiskVersionFor(
  version: ImageVersion,
): string {
  return version === "0.6" ? V06_ONDISK_VERSION : version;
}

/** Whether `s` looks like a URL with a scheme (e.g. `http://`, `s3://`). */
function isUrl(s: string): boolean {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(s);
}

/**
 * Lexically normalize a local path so different spellings of the same location
 * compare equal: collapse duplicate slashes, drop `.` segments, resolve `..`,
 * and strip a trailing slash. This mirrors (a subset of) the Python side's
 * `Path(...).resolve()` in `_stores_are_same`, so `foo`, `./foo/`, and
 * `bar/../foo` all match and are routed to the safe in-place path rather than
 * the overwriting write-to-new-store path (issue #219). It is purely lexical --
 * it never touches the filesystem -- so it does not resolve symlinks or make a
 * relative path absolute against the cwd; genuinely equivalent forms that
 * differ in those respects still fall back to the never-mutating copy path.
 */
function normalizeLocalPath(p: string): string {
  const isAbsolute = p.startsWith("/");
  const stack: string[] = [];
  for (const seg of p.split("/")) {
    if (seg === "" || seg === ".") continue;
    if (seg === "..") {
      if (stack.length && stack[stack.length - 1] !== "..") {
        stack.pop();
      } else if (!isAbsolute) {
        stack.push("..");
      }
      // A leading `..` on an absolute path is dropped, matching path resolution.
    } else {
      stack.push(seg);
    }
  }
  return (isAbsolute ? "/" : "") + stack.join("/");
}

/** Whether `output` designates the same store as `input` (an in-place request). */
function isSameStore(
  input: UpgradeInput,
  output: UpgradeOmeZarrOptions["output"],
): boolean {
  if (output === undefined) {
    return true;
  }
  if ((input as unknown) === (output as unknown)) {
    return true;
  }
  if (typeof input === "string" && typeof output === "string") {
    // URLs: compare up to a trailing slash (no `.`/`..` resolution -- those are
    // meaningful in a URL path). Local paths: compare lexically-normalized forms
    // so `foo`, `./foo/`, and `bar/../foo` all match, mirroring the Python
    // `_stores_are_same` `.resolve()` comparison. Any remaining ambiguity still
    // routes to the never-mutating write-to-new-store path.
    if (isUrl(input) || isUrl(output)) {
      return input.replace(/\/+$/, "") === output.replace(/\/+$/, "");
    }
    return normalizeLocalPath(input) === normalizeLocalPath(output);
  }
  return false;
}

/**
 * Core `upgradeOmeZarr` algorithm. See the wrappers' public docs for the full
 * two-mode behavior; this function assumes `deps` supplies the environment's
 * I/O and writable-store resolution.
 */
export async function upgradeOmeZarrImpl(
  input: UpgradeInput,
  options: UpgradeOmeZarrOptions,
  deps: UpgradeOmeZarrDeps,
): Promise<void> {
  const version = options.version ?? "0.6";
  const validate = options.validate ?? false;
  const overwrite = options.overwrite ?? true;

  if (!isSameStore(input, options.output)) {
    // Write-to-new-store: read the source lazily and re-write it to `output`
    // through the standard, tested pipeline. `toOmeZarr` re-derives the
    // target-version metadata, so every supported transition (0.4<->0.5<->0.6)
    // works and the source store is never mutated.
    const multiscales = await deps.fromOmeZarr(input, { validate });
    await deps.toOmeZarr(options.output!, multiscales, { version, overwrite });
    return;
  }

  // In-place: resolve a writable store (rejects read-only inputs) and detect
  // the on-disk source version from the root-group attributes.
  const store = await deps.resolveWritableStore(input);
  const location = zarr.root(store);
  const rootGroup = await zarr.open(location, { kind: "group" });
  const rootAttrs = rootGroup.attrs as Record<string, unknown>;
  const sourceVersion = detectVersion(rootAttrs);
  const sourceZarrFormat = zarrFormatForVersion(sourceVersion);
  const targetZarrFormat = zarrFormatForVersion(version);

  // No-op: the store already carries the exact tag the target would write.
  // Compared on the on-disk string rather than the detected family, so a 0.6
  // store tagged with an earlier pre-release is rewritten and its tag catches
  // up with the vendored schemas, which is the only way to re-tag it. Leaves
  // the store byte-for-byte unchanged (no read of arrays, no write).
  if (
    (onDiskVersion(rootAttrs) ?? sourceVersion) === onDiskVersionFor(version)
  ) {
    return;
  }

  if (sourceZarrFormat !== targetZarrFormat) {
    // Cross-format in-place (Zarr v2<->v3, i.e. OME-Zarr 0.4<->0.5/0.6). A safe
    // rewrite would have to re-encode every array's chunk keys; to stay aligned
    // with the Python package's public behavior we direct the user to
    // write-to-new-store, which handles the transition through the tested
    // pipeline without ever touching the source.
    throw new Error(
      "In-place upgrade across the Zarr v2/v3 boundary (OME-Zarr " +
        `${normalizeVersion(sourceVersion)} -> ${version}) is not supported ` +
        "because it cannot preserve array chunk files. Pass `output` (a path " +
        "or MemoryStore) to write a new upgraded store safely.",
    );
  }

  // Same underlying Zarr format (0.5<->0.6): rewrite ONLY the root-group
  // metadata. `fromOmeZarr` reads the metadata and opens the arrays lazily (no
  // chunk data is read), and `zarr.create` overwrites the root `zarr.json`
  // alone — every array `zarr.json` and chunk file is left byte-for-byte
  // untouched. This is the fix for issue #219.
  const multiscales = await deps.fromOmeZarr(store, { validate });
  const attributes = buildRootAttributes(multiscales.metadata, version);
  await zarr.create(location, { attributes });
}
