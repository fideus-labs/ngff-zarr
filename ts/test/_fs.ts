// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Filesystem helpers with the shapes of the runtime calls the suite was
 * written against, implemented on `node:fs`.
 *
 * Differences from the raw Node APIs that matter to the tests: `readFile`
 * returns a plain `Uint8Array` (not a `Buffer`, whose prototype breaks deep
 * equality) and `stat` exposes `isFile`/`isDirectory` as booleans.
 */

import { readdirSync, readFileSync } from "node:fs";
import {
  mkdir as fsMkdir,
  mkdtemp,
  readFile as fsReadFile,
  rm,
  stat as fsStat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

type PathLike = string | URL;

export async function readFile(path: PathLike): Promise<Uint8Array> {
  const buffer = await fsReadFile(path);
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

export function readTextFile(path: PathLike): Promise<string> {
  return fsReadFile(path, "utf8");
}

export function readTextFileSync(path: PathLike): string {
  return readFileSync(path, "utf8");
}

export function writeTextFile(path: PathLike, data: string): Promise<void> {
  return writeFile(path, data);
}

export function makeTempDir(
  options: { prefix?: string } = {},
): Promise<string> {
  return mkdtemp(join(tmpdir(), options.prefix ?? "tmp"));
}

export function remove(
  path: PathLike,
  options: { recursive?: boolean } = {},
): Promise<void> {
  return rm(path, { recursive: options.recursive ?? false });
}

export async function mkdir(
  path: PathLike,
  options: { recursive?: boolean } = {},
): Promise<void> {
  await fsMkdir(path, { recursive: options.recursive ?? false });
}

export interface FileInfo {
  isFile: boolean;
  isDirectory: boolean;
  isSymlink: boolean;
  size: number;
  mtime: Date | null;
}

export async function stat(path: PathLike): Promise<FileInfo> {
  const info = await fsStat(path);
  return {
    isFile: info.isFile(),
    isDirectory: info.isDirectory(),
    isSymlink: info.isSymbolicLink(),
    size: info.size,
    mtime: info.mtime,
  };
}

export interface DirEntry {
  name: string;
  isFile: boolean;
  isDirectory: boolean;
  isSymlink: boolean;
}

export function readDirSync(path: PathLike): DirEntry[] {
  return readdirSync(path, { withFileTypes: true }).map((entry) => ({
    name: entry.name,
    isFile: entry.isFile(),
    isDirectory: entry.isDirectory(),
    isSymlink: entry.isSymbolicLink(),
  }));
}
