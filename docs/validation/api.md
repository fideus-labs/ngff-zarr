---
type: reference
title: Structural Validation API
created: 2026-06-03
tags:
  - validation
  - ome-zarr
  - api
  - python
  - typescript
related:
  - '[[overview]]'
  - '[[rule-reference]]'
  - '[[parity]]'
---

<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->

# Structural Validation API

How to invoke structural validation from Python and TypeScript. For the rules
themselves see [[rule-reference]], and for the levels and scope see
[[overview]].

On read, validation is **opt-in** and runs in two stages for v0.4+ metadata:
schema (shape) validation first, then — at the default `strict`
[[overview|level]] — the structural rules. Both stages are gated by the same
flag (`validate=True` in Python, `{ validate: true }` in TypeScript); the
flag defaults to off, so nothing extra runs unless you ask for it.

## Python

```python
from ngff_zarr import (
    validate_structural,
    ValidationLevel,
    ValidateOptions,
    ValidationError,
    SpecRule,
)
```

`validate_structural(metadata, options=None, version=None)` runs the
image/multiscales rules. When `options` is `None` it uses `ValidateOptions()`,
i.e. `ValidationLevel.STRICT`. `version` is the OME-Zarr version the metadata
declares; the axis rules are inert for the versions that adopt the RFC-3 axis
model (see [[parity]]), so omitting it holds every store to the v0.4 axis
caps. A `ValidationError` carries `.rule` (a `SpecRule`),
`.message` (str), and `.location` (`str | None`); `str(exc)` is
`Spec rule [<rule>] violated: <message>`.

**Validate on read** with `from_ome_zarr(store, validate=True)` (the `validate`
keyword defaults to `False`). The `ValidationError` propagates unchanged:

```python
from ngff_zarr import from_ome_zarr, ValidationError

try:
    multiscales = from_ome_zarr("image.ome.zarr", validate=True)
except ValidationError as exc:
    print(f"rule={exc.rule.value} location={exc.location}: {exc.message}")
```

**Validate an already-parsed metadata object** directly, choosing the level:

```python
from ngff_zarr import (
    validate_structural,
    ValidateOptions,
    ValidationLevel,
    ValidationError,
)

try:
    validate_structural(
        multiscales.metadata,
        ValidateOptions(level=ValidationLevel.STRICT),  # the default
    )
except ValidationError as exc:
    print(exc)  # "Spec rule [<rule>] violated: <message>"
```

For HCS metadata, the companion entry points `validate_plate(plate)` and
`validate_well(plate, well)` follow the same `strict`-by-default contract.
`ValidationLevel.SCHEMA_ONLY` makes any orchestrator return immediately without
running a structural rule. Because `ValidationLevel` and `SpecRule` derive from
`str`, you can compare directly: `ValidationLevel.STRICT == "strict"` and
`SpecRule.AXIS_COUNT == "axis-count"`.

## TypeScript

```typescript
import {
  fromOmeZarr,
  validateStructural,
  ValidationError,
  ValidationLevel,
  SpecRule,
} from "@fideus-labs/ngff-zarr";
```

`validateStructural(metadata, options?, version?)` runs the image/multiscales
rules. When `options` (or its `level`) is omitted, the level resolves to
`ValidationLevel.Strict`. `version` is the OME-Zarr version the metadata
declares; the axis rules are inert for the versions that adopt the RFC-3 axis
model (see [[parity]]), so omitting it holds every store to the v0.4 axis
caps. A `ValidationError` carries a readonly `rule`
(`SpecRule`) and an optional `location` (`string`); its `message` is
`Spec rule [<rule>] violated: <message>`.

**Validate an already-parsed metadata object** directly — this is the typed
path, where `instanceof ValidationError` exposes `rule` and `location`:

```typescript
import { validateStructural, ValidationError } from "@fideus-labs/ngff-zarr";

try {
  validateStructural(metadata); // throws on the first violated SpecRule
} catch (err) {
  if (err instanceof ValidationError) {
    console.error(err.rule, err.location, err.message);
  }
}
```

**Validate on read** with `fromOmeZarr(store, { validate: true })` (the
`validate` option defaults to `false`). For a v0.4+ store this runs schema
validation first, then the strict structural rules:

```typescript
import { fromOmeZarr } from "@fideus-labs/ngff-zarr";

try {
  const multiscales = await fromOmeZarr(url, { validate: true });
  // ... use multiscales.images / multiscales.metadata
} catch (err) {
  console.error(err instanceof Error ? err.message : String(err));
}
```

> **Note.** `fromOmeZarr` wraps any read failure — including a structural
> `ValidationError` — as `Error("Failed to read OME-Zarr: <message>")`. The
> original `rule`/`location` are therefore not recoverable via `instanceof`
> on the read path; only the prefixed message survives. When you need the typed
> `SpecRule` and `location`, call `validateStructural` on the parsed metadata
> directly (the typed path above).

For HCS metadata, `validatePlate(plate)` and `validateWell(plate, well)` are
the plate/well counterparts. Pass `{ level: ValidationLevel.SchemaOnly }` to
skip the structural rules.

## Public surface summary

| Symbol | Python (`ngff_zarr`) | TypeScript (`@fideus-labs/ngff-zarr`) |
| ------ | -------------------- | ------------------------------------- |
| Image orchestrator | `validate_structural` | `validateStructural` |
| Plate orchestrator | `validate_plate` | `validatePlate` |
| Well orchestrator | `validate_well` | `validateWell` |
| Rule identifiers | `SpecRule` | `SpecRule` |
| Levels | `ValidationLevel` | `ValidationLevel` |
| Options | `ValidateOptions` | `ValidateOptions` |
| Error type | `ValidationError` | `ValidationError` |
| Validate on read | `from_ome_zarr(store, validate=True)` | `fromOmeZarr(store, { validate: true })` |

The identifier set, level set, default, and evaluation order are identical
across both ports and locked by tests — see [[parity]].
