// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// The oztest validate_zarr cases, answered like the Python oztest dingus and
// scored the way the oztest runner scores them. The cases not passed yet are
// listed in oztest_known_failures.txt; any other result fails, including a
// listed case that passes. The cases are the ones the Python tests download.
import { test } from "vite-plus/test";
import {
  cpSync,
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";
import { fromOmeZarr } from "../src/io/from_ngff_zarr.ts";

const CASES =
  "../py/test/data/oztest/oztest-077ebbe4e1996490516259d2c3ce40a055c72e04/cases";

// An error the reader throws on a bug, not on invalid metadata.
const CRASHES = [TypeError, RangeError, ReferenceError];

async function status(store: string, validity: string): Promise<string> {
  const read = (name: string) =>
    JSON.parse(readFileSync(join(store, name), "utf8"));
  const attributes = existsSync(join(store, "zarr.json"))
    ? (read("zarr.json").attributes ?? {})
    : read(".zattrs");
  const ome = attributes.ome ?? attributes;
  // No HCS reader in this port: the answer is invalid, with xfail.
  if ("plate" in ome || "well" in ome) {
    return validity === "invalid" ? "pass" : "xfail";
  }
  let answer = "valid";
  try {
    if ("scene" in ome) {
      await fromOmeZarr(store, { kind: "scene", validate: true });
    } else {
      await fromOmeZarr(store, { validate: true });
    }
  } catch (error) {
    let cause = error as Error;
    while (cause.cause instanceof Error) cause = cause.cause;
    if (CRASHES.some((type) => cause instanceof type)) return "error";
    answer = "invalid";
  }
  return answer === validity ? "pass" : "fail";
}

test("oztest validate_zarr cases", { timeout: 600_000 }, async () => {
  const known = new Map<string, string>();
  const list = readFileSync("test/oztest_known_failures.txt", "utf8");
  for (const line of list.split("\n")) {
    const entry = line.split("#")[0].trim().split(/\s+/);
    if (entry.length === 2) known.set(entry[1], entry[0]);
  }
  const root = join(CASES, "validate_zarr");
  const got = new Map<string, string>();
  for (const name of readdirSync(root, { recursive: true }) as string[]) {
    if (!name.endsWith(".zarr")) continue;
    const parts = ["validate_zarr", ...name.split(sep)];
    const slug = [...parts.slice(0, -1), parts.at(-1)!.split(".")[0]].join("/");
    got.set(slug, await status(join(root, name), parts[3]));
  }
  const unexpected = [...got]
    .filter(([slug, status]) => status !== (known.get(slug) ?? "pass"))
    .map(
      ([slug, status]) =>
        `${slug}: expected ${known.get(slug) ?? "pass"}, got ${status}`,
    );
  for (const slug of known.keys()) {
    if (!got.has(slug)) unexpected.push(`${slug}: listed, but not a case`);
  }
  if (unexpected.length) throw new Error(unexpected.join("\n"));
});

test("a label image without multiscales is refused", async () => {
  const store = join(mkdtempSync(join(tmpdir(), "ngff-zarr-")), "label.zarr");
  cpSync(
    join(CASES, "validate_zarr/v0.6/core/valid/label/minimal.ome.zarr"),
    store,
    { recursive: true },
  );
  const document = join(store, "labels", "lbl", "zarr.json");
  const node = JSON.parse(readFileSync(document, "utf8"));
  delete node.attributes.ome.multiscales;
  writeFileSync(document, JSON.stringify(node));

  let message = "accepted";
  try {
    await fromOmeZarr(store, { validate: true });
  } catch (error) {
    message = ((error as Error).cause as Error).message;
  }
  if (!message.includes("labels/lbl")) throw new Error(message);
});
