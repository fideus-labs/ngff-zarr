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
structural-validation surface for OME-Zarr v0.4 and v0.5. This page describes
the contract and the tests that lock it. For the rules, see [[rule-reference]];
for usage, see [[api]].

## The contract

Both ports must agree on six observable dimensions:

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
5. **The RFC-3 version set** — the four axis rules take a `version` and are
   inert for the versions that adopt the RFC-3 free-form axis model. Both ports
   must treat exactly the same version strings as RFC-3, or the same metadata
   validates in one language and not the other.
6. **The RFC-4 orientation version set** — the three orientation rules gate the
   opposite way: they are normative at exactly the versions that adopt RFC-4
   (and when no version is given), and inert below. Both ports must treat the
   same version strings as RFC-4 for the same reason.

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
4. `axis-names-unique`
5. `scale-length-mismatch`
6. `global-coord-transform-after-per-level`
7. `dataset-order-highest-to-lowest`
8. `omero-channel-color-format`
9. `axis-orientation-anatomical-type`
10. `axis-orientation-on-non-space`
11. `axis-orientation-unique-axis`
12. `zarr-format`
13. `ome-namespace`
14. `plate-row-index-consistency`
15. `well-acquisition-missing`
16. `node-type-required`
17. `node-name-required`
18. `node-name-unique`
19. `node-id-format`
20. `node-id-unique`
21. `node-nodes-xor-path`
22. `path-type-known`
23. `coordinate-system-id-required`
24. `reference-id-required`
25. `reference-path-required`
26. `label-value-required`
27. `label-color-format`

Entries 12–13 are the v0.5 namespacing rules; they fire only for v0.5 metadata
and are inert for v0.4. Entries 16–27 are the RFC-8 node/collection rules,
dispatched by the separate `validate_collection` orchestrator. Each suite pins this list as a `CANONICAL_SPEC_RULE_IDS`
literal — byte-identical between the two languages so the tests are
line-for-line comparable.

The versions that adopt the RFC-3 axis model are pinned the same way, as a
`CANONICAL_RFC3_VERSIONS` literal:

1. `0.9.dev1`
2. `0.9.dev3`

Rules 1–3 (and the spatial arm of 3) are inert at those versions and enforced
at every other. Rule 4, `axis-names-unique`, is never inert: RFC-3 *adds* it,
and ngff-zarr applies it at all versions as a strictness choice (see
[[rule-reference]]).

The versions at which the RFC-4 orientation rules are normative are pinned the
same way, as a `CANONICAL_RFC4_VERSIONS` literal:

1. `0.9.dev1`
2. `0.9.dev3`

Rules 9–11 gate the opposite way from the RFC-3 set: they are enforced at
those versions and when no version is given, and inert at every earlier
version, where RFC-4 has no normative status (see [[rule-reference]]).

The versions that store the RFC-8 node model are pinned as a
`CANONICAL_RFC8_VERSIONS` literal:

1. `0.9.dev3`

Rules 16–22 gate like the RFC-4 set: enforced at those versions and when no
version is given, inert at every other supported version, 0.9.dev1 included
(RFC-8 lands at 0.9.dev3).

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
- **Fail-fast evaluation order** — driving the orchestrator with v0.4 metadata
  that violates the earliest rule plus every later rule, then repairing exactly
  one rule per stage across eleven stages, the rule raised at each stage builds
  a sequence equal to a shared `EXPECTED_EVALUATION_ORDER`. This proves every
  rule is evaluated strictly before all rules after it; a final, fully-repaired
  metadata is accepted with no violation.
- **RFC-3 version set** — the axis rules are inert at exactly the versions in
  `CANONICAL_RFC3_VERSIONS` and enforced at every other supported version and
  when no version is given, asserted through the public orchestrator rather
  than through the internal predicate.
- **RFC-4 orientation version set** — the orientation rules are enforced at
  exactly the versions in `CANONICAL_RFC4_VERSIONS` (and when no version is
  given) and inert at every other supported version, asserted the same way,
  once per orientation rule.
- **RFC-8 collection version set** — the node/collection rules are enforced
  at exactly the versions in `CANONICAL_RFC8_VERSIONS` (and when no version
  is given) and inert at every other supported version, asserted through
  `validate_collection` / `validateCollection`.

Sixteen of the twenty-seven rules never appear in `EXPECTED_EVALUATION_ORDER`,
each for its own reason. The two v0.5 namespacing rules (`zarr-format`,
`ome-namespace`) run last in the image orchestrator but are inert for the v0.4
metadata the order test exercises. The two HCS rules
(`plate-row-index-consistency`, `well-acquisition-missing`) and the twelve
RFC-8 rules (`node-type-required` through `label-color-format`) are
deliberately absent; they are dispatched by the separate plate/well and
collection orchestrators (see [[rule-reference]]), and the RFC-8 rules' own
fail-fast order is pinned by the shared `rfc8_collection_cases.json` fixture
both suites drive.

## Sanctioned TypeScript-only adaptations

The two ports are equivalent on the wire, with two documented, behavior-neutral
departures in the TypeScript suite:

- **Member spelling** follows each language's convention — TypeScript uses a
  PascalCase const object (`SpecRule.AxisCount`, `ValidationLevel.Strict`),
  Python an `UPPER_SNAKE` `str, Enum` (`SpecRule.AXIS_COUNT`,
  `ValidationLevel.STRICT`). The `.value` strings are identical, so the
  cross-language assertions hold.
- **The default check** — TypeScript asserts the `strict` default *observably*,
  because its options type is a bare interface with no constructor, rather than
  through a constructable options object. This alters neither the observed rule
  set nor the evaluation order. The ordering fixtures themselves are identical
  in both ports: `Axis.name` is `AxisName`, so TypeScript uses the same
  free-form axis names as Python.

Internal message-fidelity helpers in the TypeScript port (rendering whole-number
scales with a trailing `.0`, Python-style quoted name lists, and `repr`-style
strings) exist purely to keep `ValidationError` message bytes identical across
languages; they are not part of the public surface.
