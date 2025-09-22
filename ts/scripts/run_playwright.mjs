#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

import { spawn } from "child_process";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import process from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Function to run a command with proper error handling
function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: "inherit",
      shell: true,
      ...options,
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve(code);
      } else {
        reject(new Error(`Command failed with exit code ${code}`));
      }
    });

    child.on("error", (err) => {
      reject(err);
    });
  });
}

async function main() {
  try {
    console.log("📦 Installing Playwright dependencies...");

    // Check if we need to install dependencies
    const testDir = join(__dirname, "../test/browser-npm");
    const projectRoot = join(__dirname, "..");

    // Note: We're now using deno bundle instead of npm package
    console.log("📋 Using bundle instead of npm package...");

    // Install npm dependencies first
    await runCommand("npm", ["install"], { cwd: testDir });

    // Install Playwright browsers
    await runCommand("npx", ["playwright", "install"], { cwd: testDir });

    // Start the test server in the background
    console.log("🚀 Starting test server...");
    const serverProcess = spawn(
      "deno",
      ["run", "--allow-all", "test/browser/server.ts"],
      {
        cwd: projectRoot,
        stdio: ["pipe", "pipe", "pipe"],
        detached: false,
      },
    );

    // Handle server process termination
    serverProcess.on("exit", (code, signal) => {
      if (code !== null && code !== 0) {
        console.error(`[Server] Server process exited with code ${code}`);
      }
      if (signal) {
        console.log(`[Server] Server process terminated by signal ${signal}`);
      }
    });

    // Log server output for debugging
    serverProcess.stdout.on("data", (data) => {
      console.log(`[Server] ${data.toString().trim()}`);
    });

    serverProcess.stderr.on("data", (data) => {
      console.error(`[Server Error] ${data.toString().trim()}`);
    });

    // Wait for server to be ready by checking if port 3000 is responding
    const waitForServer = async () => {
      for (let i = 0; i < 30; i++) {
        // Wait up to 30 seconds
        try {
          // Use a simple HTTP check with Node.js http module instead of fetch
          const http = await import("http");
          const response = await new Promise((resolve, reject) => {
            const req = http.request(
              "http://localhost:3000/bundle-test",
              (res) => {
                resolve(res);
              },
            );
            req.on("error", reject);
            req.setTimeout(2000, () => {
              req.destroy();
              reject(new Error("Request timeout"));
            });
            req.end();
          });

          if (response.statusCode === 200) {
            console.log("✅ Test server is ready!");
            // Wait a bit longer to ensure server is fully stable
            await new Promise((resolve) => setTimeout(resolve, 2000));
            return true;
          }
        } catch (_error) {
          // Server not ready yet, continue waiting
        }
        console.log(`⏳ Waiting for server... (${i + 1}/30)`);
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      throw new Error("Server failed to start within 30 seconds");
    };

    await waitForServer();

    console.log("🎭 Running Playwright tests...");

    try {
      // Run Playwright tests
      const playwrightArgs = process.argv.slice(2);
      await runCommand("npx", ["playwright", "test", ...playwrightArgs], {
        cwd: testDir,
        env: {
          ...process.env,
          PLAYWRIGHT_PROJECT_ROOT: projectRoot,
        },
      });

      console.log("✅ All tests completed successfully!");
    } finally {
      // Clean up server process
      if (serverProcess && !serverProcess.killed) {
        console.log("🛑 Shutting down test server...");
        serverProcess.kill("SIGTERM");
        // Give it a moment to shut down gracefully
        setTimeout(() => {
          if (!serverProcess.killed) {
            serverProcess.kill("SIGKILL");
          }
        }, 3000);
      }
    }
  } catch (error) {
    console.error("❌ Test execution failed:", error.message);
    process.exit(1);
  }
}

main();
