// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// Simple test server for browser tests
import { serveFile } from "@std/http/file-server";
import { join } from "@std/path";

const PORT = 3000;

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);

  // Serve the test HTML file for root requests
  if (url.pathname === "/") {
    const filePath = join(
      Deno.cwd(),
      "test",
      "browser",
      "fixtures",
      "index.html",
    );
    return await serveFile(req, filePath);
  }

  // Serve static files from the src directory
  if (url.pathname.startsWith("/src/")) {
    const filePath = join(Deno.cwd(), url.pathname.substring(1));
    try {
      return await serveFile(req, filePath);
    } catch {
      return new Response("Not Found", { status: 404 });
    }
  }

  // Serve other test fixtures
  if (url.pathname.startsWith("/test/")) {
    const filePath = join(Deno.cwd(), url.pathname.substring(1));
    try {
      return await serveFile(req, filePath);
    } catch {
      return new Response("Not Found", { status: 404 });
    }
  }

  // Serve the npm test page
  if (url.pathname === "/npm-test") {
    const filePath = join(Deno.cwd(), "test", "browser-npm", "index.html");
    return await serveFile(req, filePath);
  }

  // Serve the bundle test page
  if (url.pathname === "/bundle-test") {
    const filePath = join(
      Deno.cwd(),
      "test",
      "browser-npm",
      "bundle-test.html",
    );
    return await serveFile(req, filePath);
  }

  // Serve the simple test page
  if (url.pathname === "/simple-test") {
    const filePath = join(
      Deno.cwd(),
      "test",
      "browser-npm",
      "simple-test.html",
    );
    return await serveFile(req, filePath);
  }

  // Serve the browser test page
  if (url.pathname === "/browser-test") {
    const filePath = join(
      Deno.cwd(),
      "test",
      "browser-npm",
      "browser-test.html",
    );
    return await serveFile(req, filePath);
  }

  // Serve the npm package files from test directory
  if (url.pathname.startsWith("/npm/")) {
    const filePath = join(
      Deno.cwd(),
      "test",
      "browser-npm",
      url.pathname.substring(1),
    );
    try {
      return await serveFile(req, filePath);
    } catch {
      return new Response("Not Found", { status: 404 });
    }
  }

  // Serve the bundle file
  if (url.pathname === "/ngff-zarr.bundle.js") {
    const filePath = join(
      Deno.cwd(),
      "test",
      "browser-npm",
      "ngff-zarr.bundle.js",
    );
    try {
      const response = await serveFile(req, filePath);
      // Ensure correct MIME type for ES modules
      response.headers.set("Content-Type", "application/javascript");
      return response;
    } catch {
      return new Response("Not Found", { status: 404 });
    }
  }

  // Serve the browser bundle file
  if (url.pathname === "/ngff-zarr-browser.bundle.js") {
    const filePath = join(
      Deno.cwd(),
      "test",
      "browser-npm",
      "ngff-zarr-browser.bundle.js",
    );
    try {
      const response = await serveFile(req, filePath);
      // Ensure correct MIME type for ES modules
      response.headers.set("Content-Type", "application/javascript");
      return response;
    } catch {
      return new Response("Not Found", { status: 404 });
    }
  }

  return new Response("Not Found", { status: 404 });
}

console.log(`Test server running at http://localhost:${PORT}`);
console.log(`Listening on http://0.0.0.0:${PORT}/ (http://localhost:${PORT}/)`);

try {
  const server = Deno.serve({ port: PORT }, handler);
  await server.finished;
} catch (error) {
  console.error("Failed to start server:", error);
  Deno.exit(1);
}
