#!/usr/bin/env -S deno run --allow-all
// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

import { parseArgs } from "@std/cli/parse-args";

const args = parseArgs(Deno.args, {
  boolean: ["help", "check", "fmt", "lint", "test"],
  string: ["target"],
  default: {
    target: "all",
  },
});

if (args.help) {
  console.log(`
Usage: deno run --allow-all scripts/build.ts [options]

Options:
  --help          Show this help message
  --check         Run type checking
  --fmt           Run formatting
  --lint          Run linting
  --test          Run tests
  --target <t>    Build target (all, check, fmt, lint, test, npm)

Examples:
  deno run --allow-all scripts/build.ts --check --fmt --lint --test
  deno run --allow-all scripts/build.ts --target npm
`);
  Deno.exit(0);
}

async function runCommand(cmd: string[], description: string) {
  console.log(`🔨 ${description}...`);
  const process = new Deno.Command(cmd[0], {
    args: cmd.slice(1),
    stdout: "inherit",
    stderr: "inherit",
  });

  const result = await process.output();

  if (!result.success) {
    console.error(`❌ ${description} failed`);
    Deno.exit(1);
  }

  console.log(`✅ ${description} completed`);
}

async function main() {
  const tasks: Array<{
    condition: boolean;
    cmd: string[];
    description: string;
  }> = [
    {
      condition: args.check || args.target === "all" || args.target === "check",
      cmd: ["deno", "check", "src/mod.ts"],
      description: "Type checking",
    },
    {
      condition: args.fmt || args.target === "all" || args.target === "fmt",
      cmd: ["deno", "fmt", "--check"],
      description: "Format checking",
    },
    {
      condition: args.lint || args.target === "all" || args.target === "lint",
      cmd: ["deno", "lint"],
      description: "Linting",
    },
    {
      condition: args.test || args.target === "all" || args.target === "test",
      cmd: ["deno", "test", "--allow-all", "--ignore=test/browser-npm/"],
      description: "Running tests",
    },
    {
      condition: args.target === "npm",
      cmd: ["deno", "run", "--allow-all", "scripts/build_npm.ts"],
      description: "Building npm package",
    },
  ];

  for (const task of tasks) {
    if (task.condition) {
      await runCommand(task.cmd, task.description);
    }
  }

  console.log("🎉 Build completed successfully!");
}

if (import.meta.main) {
  await main();
}
