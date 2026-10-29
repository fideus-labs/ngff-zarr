// SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
// SPDX-License-Identifier: MIT

/**
 * Assertion helpers with the call signatures of `@std/assert`, built on
 * `node:assert/strict`.
 *
 * The suite was written against `@std/assert`; keeping its signatures means a
 * test reads the same under Vitest and the port touched only import lines.
 */

import assert_ from "node:assert/strict";

/** Any constructor producing an Error, as `@std/assert` accepts. */
type ErrorClass<E extends Error = Error> = abstract new (...args: any[]) => E;

// node:assert's overloads take either no message or a definite string, so an
// optional `msg` is forwarded only when given.

export function assert(expr: unknown, msg?: string): asserts expr {
  if (msg === undefined) assert_.ok(expr);
  else assert_.ok(expr, msg);
}

export function assertFalse(expr: unknown, msg?: string): void {
  assert(!expr, msg);
}

export function assertEquals<T>(actual: T, expected: T, msg?: string): void {
  if (msg === undefined) assert_.deepStrictEqual(actual, expected);
  else assert_.deepStrictEqual(actual, expected, msg);
}

export function assertNotEquals<T>(actual: T, expected: T, msg?: string): void {
  if (msg === undefined) assert_.notDeepStrictEqual(actual, expected);
  else assert_.notDeepStrictEqual(actual, expected, msg);
}

export function assertExists<T>(
  actual: T,
  msg?: string,
): asserts actual is NonNullable<T> {
  if (actual === undefined || actual === null) {
    assert_.fail(
      msg ??
        `Expected actual: "${String(actual)}" to not be null or undefined.`,
    );
  }
}

export function assertStringIncludes(
  actual: string,
  expected: string,
  msg?: string,
): void {
  assert(
    actual.includes(expected),
    msg ?? `Expected actual: "${actual}" to contain: "${expected}".`,
  );
}

export function assertMatch(
  actual: string,
  expected: RegExp,
  msg?: string,
): void {
  if (msg === undefined) assert_.match(actual, expected);
  else assert_.match(actual, expected, msg);
}

export function assertInstanceOf<
  T extends abstract new (...args: any[]) => unknown,
>(
  actual: unknown,
  expectedType: T,
  msg?: string,
): asserts actual is InstanceType<T> {
  assert(
    actual instanceof expectedType,
    msg ??
      `Expected object to be an instance of "${expectedType.name}" but was "${
        (actual as { constructor?: { name?: string } })?.constructor?.name
      }".`,
  );
}

/** Absolute tolerance, matching `@std/assert`'s default. */
const DEFAULT_TOLERANCE = 1e-7;

export function assertAlmostEquals(
  actual: number,
  expected: number,
  tolerance: number = DEFAULT_TOLERANCE,
  msg?: string,
): void {
  if (Object.is(actual, expected)) return;
  const delta = Math.abs(expected - actual);
  if (delta <= tolerance) return;
  assert_.fail(
    msg ??
      `Expected actual: "${actual}" to be close to "${expected}": delta "${delta}" is greater than "${tolerance}".`,
  );
}

function checkThrown<E extends Error>(
  thrown: unknown,
  ErrorClass: ErrorClass<E> | undefined,
  msgIncludes: string | undefined,
  msg: string | undefined,
): void {
  if (ErrorClass !== undefined) {
    assert(
      thrown instanceof ErrorClass,
      msg ??
        `Expected error to be instance of "${ErrorClass.name}", but was "${
          (thrown as { constructor?: { name?: string } })?.constructor?.name
        }".`,
    );
  }
  if (msgIncludes !== undefined) {
    const message = thrown instanceof Error ? thrown.message : String(thrown);
    assert(
      message.includes(msgIncludes),
      msg ??
        `Expected error message to include "${msgIncludes}", but got "${message}".`,
    );
  }
}

/**
 * Assert that `fn` throws. With `ErrorClass` the thrown value must be an
 * instance of it; with `msgIncludes` its message must contain that text.
 * Returns the thrown value.
 */
export function assertThrows<E extends Error = Error>(
  fn: () => unknown,
  ErrorClass?: ErrorClass<E>,
  msgIncludes?: string,
  msg?: string,
): E {
  let thrown: unknown;
  let didThrow = false;
  try {
    fn();
  } catch (error) {
    thrown = error;
    didThrow = true;
  }
  if (!didThrow) assert_.fail(msg ?? "Expected function to throw.");
  checkThrown(thrown, ErrorClass, msgIncludes, msg);
  return thrown as E;
}

/** Async counterpart of {@link assertThrows}: `fn` must reject. */
export async function assertRejects<E extends Error = Error>(
  fn: () => Promise<unknown>,
  ErrorClass?: ErrorClass<E>,
  msgIncludes?: string,
  msg?: string,
): Promise<E> {
  let thrown: unknown;
  let didThrow = false;
  try {
    await fn();
  } catch (error) {
    thrown = error;
    didThrow = true;
  }
  if (!didThrow) assert_.fail(msg ?? "Expected function to reject.");
  checkThrown(thrown, ErrorClass, msgIncludes, msg);
  return thrown as E;
}
