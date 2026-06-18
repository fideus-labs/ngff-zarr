---
type: reference
title: Cross-Language Parity Guarantee
created: 2026-06-03
tags:
  - validation
  - ome-zarr
  - parity
  - cross-language
  - testing
related:
  - '[[overview]]'
  - '[[rule-reference]]'
  - '[[api]]'
---

<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->

# Cross-Language Parity Guarantee

The Python and TypeScript ports expose an **identical, byte-stable**
structural-validation surface for OME-Zarr v0.4. This page describes the
contract and the tests that lock it. For the rules, see [[rule-reference]]; for
usage, see [[api]].

## The contract

Both ports must agree on four observable dimensions:

1. **Rule identifiers** — the same `SpecRule` string values, in the same
   canonical declaration/iteration order.
2. **Identifier format** — every value is lower-kebab-case, matching the
   anchored pattern `[a-z]+(-[a-z]+)*`.
3. **Levels** — `ValidationLevel` has exactly `{ "schema_only", "strict" }`,
   with `strict` as the default applied when no options are passed.
4. **Fail-fast evaluation order** — each image/multiscales orchestrator
   (`validate_structural` / `validateStructural`) evaluates rules in the same
   canonical order, so the same metadata yields the same first violation in
   both languages.

Because both test suites assert these facts against the **same literal
identifier list**, adding, removing, renaming, or reordering a rule — or
changing the evaluation order, the kebab format, or the level set/default — in
only one language makes that language's suite diverge from the shared manifest
and fails CI. Such changes must therefore be deliberate, mirrored edits in both
ports.

## The locked identifier manifest

The canonical `SpecRule` set, in evaluation order:

1. `axis-count`
2. `axis-type`
3. `axis-order`
4. `scale-length-mismatch`
5. `global-coord-transform-after-per-level`
6. `dataset-order-highest-to-lowest`
7. `omero-channel-color-format`
8. `axis-orientation-consistent-type`
9. `axis-orientation-completeness`
10. `plate-row-index-consistency`
11. `well-acquisition-missing`

Each suite pins this list as a `CANONICAL_SPEC_RULE_IDS` literal — byte-identical
between the two languages so the tests are line-for-line comparable.

## The parity tests

| Language   | Test file |
| ---------- | --------- |
| Python     | `py/test/test_structural_validation_parity.py` |
| TypeScript | `ts/test/structural_validation_parity_test.ts` |

Each suite independently locks:

- **Set equality** — the full set of `SpecRule` values equals the canonical
  manifest (no more, no fewer).
- **Declaration/iteration order** — the ordered list of values equals the
  manifest in exact order (Python iterates the `Enum`; TypeScript reads
  `Object.values(SpecRule)`).
- **No silent aliases** — the value set size equals the manifest length, so no
  two members may share a value.
- **Lower-kebab-case format** — every value matches the anchored pattern
  (`re.fullmatch(r"[a-z]+(-[a-z]+)*", value)` in Python; the equivalent
  `/^[a-z]+(-[a-z]+)*$/` in TypeScript).
- **Level set and default** — `ValidationLevel` has exactly `schema_only` and
  `strict`, and `strict` is the default.
- **Fail-fast evaluation order** — driving the orchestrator with metadata that
  violates the earliest rule plus every later rule, then repairing exactly one
  rule per stage across ten stages, the rule raised at each stage builds a
  sequence equal to a shared `EXPECTED_EVALUATION_ORDER`. This proves every rule
  is evaluated strictly before all rules after it; a final, fully-repaired
  metadata is accepted with no violation.

The two HCS rules (`plate-row-index-consistency`, `well-acquisition-missing`)
are deliberately absent from the image orchestrator's evaluation order; they are
dispatched by the separate plate/well orchestrators (see [[rule-reference]]).

## Sanctioned TypeScript-only adaptations

The two ports are equivalent on the wire, with two documented, behavior-neutral
departures in the TypeScript suite:

- **Member spelling** follows each language's convention — TypeScript uses a
  PascalCase const object (`SpecRule.AxisCount`, `ValidationLevel.Strict`),
  Python an `UPPER_SNAKE` `str, Enum` (`SpecRule.AXIS_COUNT`,
  `ValidationLevel.STRICT`). The `.value` strings are identical, so the
  cross-language assertions hold.
- **Axis-name labels and the default check** — TypeScript's ordering fixtures
  use valid `SupportedDims` members in place of the Python helper's free-form
  axis names, and assert the `strict` default *observably* (because the
  TypeScript options type is a bare interface with no constructor) rather than
  through a constructable options object. Neither alters the observed rule set
  or evaluation order.

Internal message-fidelity helpers in the TypeScript port (rendering whole-number
scales with a trailing `.0`, Python-style quoted name lists, and `repr`-style
strings) exist purely to keep `ValidationError` message bytes identical across
languages; they are not part of the public surface.
