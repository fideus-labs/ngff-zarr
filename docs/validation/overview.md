---
type: reference
title: Structural Validation Overview
created: 2026-06-03
tags:
  - validation
  - ome-zarr
  - structural-validation
related:
  - '[[rule-reference]]'
  - '[[api]]'
  - '[[parity]]'
---

<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->

# Structural Validation Overview

**Structural validation** enforces the OME-Zarr v0.4 specification MUSTs that a
JSON Schema cannot express. It is layered conceptually on top of *schema*
validation: where schema validation answers "is this shaped like OME-Zarr?",
structural validation answers "does this obey the spec's structural
invariants?" — axis counts and ordering, coordinate-transformation arity,
finest-to-coarsest dataset ordering, OMERO channel color format, RFC 4
anatomical orientation, and HCS plate/well consistency.

The rules operate on the already-parsed metadata object (the `Metadata`,
`Plate`, and `Well` dataclasses in Python; their equivalents in TypeScript), so
they run even when the optional JSON Schema validator is not installed. Each
rule has a stable, lower-kebab-case identifier (a [[rule-reference|SpecRule]])
that is part of the observable surface — reused verbatim in error messages,
logs, tests, and across both language ports.

See [[rule-reference]] for the full rule table, [[api]] for how to invoke
validation from Python and TypeScript, and [[parity]] for the cross-language
equivalence guarantee.

## Why JSON Schema cannot express these MUSTs

A JSON Schema validates the *shape* of one node at a time. The OME-Zarr v0.4
MUSTs enforced here are instead cross-field, ordering, positional, and
conditional constraints that span multiple nodes:

- **Cross-field** — `scale-length-mismatch` requires every `scale` and
  `translation` vector to have exactly one entry per axis, i.e. a length equal
  to `len(axes)`. That target lives in a sibling array, and JSON Schema's
  `minItems`/`maxItems` are constants that cannot reference another node's size.
- **Ordering / relational** — `dataset-order-highest-to-lowest` requires each
  multiscale level's spatial scale to be no smaller than the previous level's,
  a pairwise comparison across adjacent elements of the `datasets` array.
  Schema keywords validate array elements independently of one another.
- **Positional / stateful** — `global-coord-transform-after-per-level` requires
  exactly one `scale` per dataset and forbids a `scale` after a `translation`;
  an element's validity depends on the type and position of earlier elements,
  which `prefixItems`/`contains` cannot capture.
- **Cross-object lookup** — `plate-row-index-consistency` requires a well's
  `rowIndex` to equal the position of the row named in its `path` within
  `plate.rows`, a derived lookup across objects.
- **Conditional** — `well-acquisition-missing` only applies when the plate
  declares more than one acquisition.

These imperative, multi-node dependencies are exactly why the rules are
implemented as ordinary code layered on top of schema validation, rather than
as schema keywords.

## Validation levels

Validation runs at one of two levels, expressed by the `ValidationLevel` enum.
There are exactly two members, and `strict` is the default:

| Level (`ValidationLevel`) | Wire value     | Behavior |
| ------------------------- | -------------- | -------- |
| `STRICT` (default)        | `"strict"`     | Runs every structural rule in canonical spec-MUST order, failing fast on the first violation. |
| `SCHEMA_ONLY`             | `"schema_only"`| Runs no structural rule and returns immediately. Schema (shape) validation remains the separate concern of the schema validator on the raw attribute dict. |

The default is applied whenever no options are passed: in Python,
`ValidateOptions()` constructs with `level=ValidationLevel.STRICT`; in
TypeScript, `validateStructural(metadata)` resolves the level to
`ValidationLevel.Strict` when none is supplied.

### Explicitly out of scope

Deep, store-reading checks are intentionally **out of scope for both levels** —
neither level reads array chunks. For example, confirming that each dataset's
array actually exists on disk with the expected shape and dtype is *not* part of
structural validation. The rules validate metadata only.

## Scope and orchestrators

Structural validation is dispatched through three fail-fast entry points:

- **Image / multiscales** (including OMERO color and RFC 4 orientation) —
  `validate_structural` (Python) / `validateStructural` (TypeScript).
- **HCS plate** — `validate_plate` / `validatePlate`.
- **HCS well** — `validate_well` / `validateWell` (validated in the context of
  its parent plate, whose acquisition declarations govern the well rules).

Each entry point raises a `ValidationError` on the first violated rule, in
canonical spec order; later rules do not run. A `ValidationError` carries the
offending `SpecRule`, a human-readable `message`, and an optional dotted-segment
`location` (e.g. `multiscales[0].datasets[2].coordinateTransformations[0]`). Its
string form is:

```text
Spec rule [<rule>] violated: <message>
```

Continue to [[rule-reference]] for every rule, or [[api]] to start using
validation.
