// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
//
// Tests for the exported version constants in src/types/supported_versions.ts.
// They pin the tag the writers emit for OME-Zarr 0.6 to the final release
// string, and guard against the historical 0.6 pre-release tags creeping back
// into the enum or the supported-version list.
import { assertEquals } from "@std/assert";
import {
  isSupportedVersion,
  isV06Version,
  NgffVersion,
  SUPPORTED_VERSIONS,
  V06_ONDISK_VERSION,
} from "../src/mod.ts";

// The historical pre-release tags earlier releases wrote into a 0.6 store.
const V06_PRERELEASE_TAGS = ["0.6.dev4", "0.6rc0"] as const;

Deno.test("NgffVersion.LATEST is the 0.6 release", () => {
  assertEquals(NgffVersion.LATEST, "0.6");
  assertEquals(NgffVersion.LATEST, NgffVersion.V06);
});

Deno.test("V06_ONDISK_VERSION is the bare 0.6 tag", () => {
  // Since the 0.6 release the on-disk `ome.version` for the API version
  // "0.6" is "0.6" itself; the constant stays as the single place the tag
  // lives so writers and upgradeOmeZarr read it from here.
  assertEquals(V06_ONDISK_VERSION, NgffVersion.V06);
  assertEquals(V06_ONDISK_VERSION, "0.6");
});

Deno.test("SUPPORTED_VERSIONS lists 0.6 and 0.9.dev1 but no 0.6 pre-release", () => {
  const supported = SUPPORTED_VERSIONS.map((version) => version as string);
  assertEquals(supported.includes("0.6"), true);
  assertEquals(supported.includes("0.9.dev1"), true);
  for (const tag of V06_PRERELEASE_TAGS) {
    assertEquals(supported.includes(tag), false, `${tag} is still supported`);
    assertEquals(isSupportedVersion(tag), false);
  }
  assertEquals(isSupportedVersion("0.6"), true);
});

Deno.test("NgffVersion carries no 0.6 pre-release member", () => {
  const values = Object.values(NgffVersion) as string[];
  for (const tag of V06_PRERELEASE_TAGS) {
    assertEquals(values.includes(tag), false, `${tag} is still in the enum`);
  }
});

Deno.test("isV06Version still reads the historical 0.6 pre-release tags", () => {
  // Removed from the enum, the tags stay readable through the family check
  // so stores written by earlier releases keep loading.
  for (const tag of V06_PRERELEASE_TAGS) {
    assertEquals(isV06Version(tag), true);
  }
  assertEquals(isV06Version("0.6"), true);
  assertEquals(isV06Version("0.5"), false);
});
