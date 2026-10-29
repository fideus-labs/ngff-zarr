// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Static file server for the Playwright browser tests: serves
// test/browser-npm/, with short routes for its test pages.

import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";

const PORT = 3000;
const ROOT = resolve("test", "browser-npm");

const CONTENT_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".json": "application/json",
  ".map": "application/json",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
};

const PAGES: Record<string, string> = {
  "/bundle-test": "bundle-test.html",
  "/simple-test": "simple-test.html",
  "/from_ngff_zarr_test": "from_ngff_zarr_test.html",
};

createServer(async (req, res) => {
  const { pathname } = new URL(req.url ?? "/", `http://localhost:${PORT}`);
  const file = resolve(ROOT, PAGES[pathname] ?? pathname.slice(1));
  if (!file.startsWith(ROOT + sep)) {
    res.writeHead(404).end("Not Found");
    return;
  }
  try {
    const body = await readFile(file);
    res.writeHead(200, {
      "content-type":
        CONTENT_TYPES[extname(file)] ?? "application/octet-stream",
    });
    res.end(body);
  } catch (error) {
    const code = (error as { code?: string }).code;
    if (code === "ENOENT" || code === "EISDIR") {
      res.writeHead(404).end("Not Found");
      return;
    }
    console.error(`[server] ${req.url}:`, error);
    res.writeHead(500).end("Internal Server Error");
  }
}).listen(PORT, () => {
  console.log(`Test server running at http://localhost:${PORT}`);
});
