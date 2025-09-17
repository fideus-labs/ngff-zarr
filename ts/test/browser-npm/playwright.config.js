// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT
// @ts-check

import process from "node:process";

/**
 * @see https://playwright.dev/docs/test-configuration
 */
const config = {
  testDir: "./",
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: false,
  /* Retry on CI only */
  retries: 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 2 : undefined, // Use default number of workers (CPU cores) locally, 2 on CI
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: "list",
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: "http://localhost:3000",
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: "on-first-retry",
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      use: {
        channel: "chrome",
      },
    },

    {
      name: "firefox",
      use: {
        browserName: "firefox",
      },
    },

    {
      name: "webkit",
      use: {
        browserName: "webkit",
      },
    },

    /* Test against mobile viewports. */
    {
      name: "Mobile Chrome",
      use: {
        browserName: "chromium",
        viewport: { width: 375, height: 667 },
        isMobile: true,
        hasTouch: true,
      },
    },
    {
      name: "Mobile Safari",
      use: {
        browserName: "webkit",
        viewport: { width: 375, height: 667 },
        isMobile: true,
        hasTouch: true,
      },
    },
  ],
  /* Run your local dev server before starting the tests */
  // webServer: {
  //   command: "cd ../.. && deno run --allow-all test/browser/server.ts",
  //   url: "http://localhost:3000",
  //   reuseExistingServer: true,
  //   timeout: 120000,
  // },
};

export default config;
