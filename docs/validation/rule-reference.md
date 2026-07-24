---
type: reference
title: Structural Validation Rule Reference
created: 2026-06-03
tags:
  - validation
  - ome-zarr
  - structural-validation
  - rules
related:
  - '[[overview]]'
  - '[[api]]'
  - '[[parity]]'
---

<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->

# Structural Validation Rule Reference

This is the canonical table of OME-Zarr structural rules. Each rule has a
stable, lower-kebab-case identifier (the `SpecRule` value) that is identical
across the Python and TypeScript ports — see [[parity]] for the guarantee. Most
rules enforce OME-Zarr v0.4 MUSTs; the two v0.5 namespacing rules
(`zarr-format`, `ome-namespace`) fire only for v0.5 metadata and are inert (a
no-op) for v0.4. For the conceptual background and the two validation levels,
see [[overview]]; for invocation, see [[api]].

Location strings are dotted-segment, JSON-Pointer-style identifiers of the
offending metadata node, emitted byte-for-byte identically in both languages.

## Canonical rule table

The rules are listed in canonical spec-MUST evaluation order — the same order
the `SpecRule` enum declares them and the orchestrators evaluate them.

| #  | Identifier | Applies to | What it checks | Spec MUST | Example location |
| -- | ---------- | ---------- | -------------- | --------- | ---------------- |
| 1  | `axis-count` | images/multiscales | The number of axes is within the closed range 2–5. | v0.4: between 2 and 5 axes, inclusive. | `multiscales[0].axes` |
| 2  | `axis-type` | images/multiscales | At most one `time` axis and at most one `channel` axis. | v0.4: ≤ 1 time axis and ≤ 1 channel axis. | `multiscales[0].axes` |
| 3  | `axis-order` | images/multiscales | Axes ordered time → channel → space; at most 3 space axes; the space-axis names are the length-N suffix of `(z, y, x)`. | v0.4: axes ordered time, then channel, then space, with spatial names a suffix of `(z, y, x)`. | `multiscales[0].axes[1]` |
| 4  | `scale-length-mismatch` | images/multiscales | Every `scale`/`translation` vector length equals the axis count, for both the global and per-dataset transforms. | v0.4: each scale/translation has exactly one entry per axis. | `multiscales[0].datasets[2].coordinateTransformations[0]` |
| 5  | `global-coord-transform-after-per-level` | images/multiscales | Exactly one `scale` per dataset, and a `translation` must follow — not precede — its `scale`. | v0.4: each dataset defines exactly one scale; a translation follows its scale. | `multiscales[0].datasets[1].coordinateTransformations` |
| 6  | `dataset-order-highest-to-lowest` | images/multiscales | Datasets ordered finest → coarsest; the spatial scale must not decrease as the level index rises. | v0.4: multiscale datasets ordered from highest to lowest resolution. | `multiscales[0].datasets[2]` |
| 7  | `omero-channel-color-format` | OMERO | Each OMERO channel `color` is exactly six hexadecimal digits (RGB). | v0.4: OMERO channel color is 6 hex digits. | `multiscales[0].omero.channels[0].color` |
| 8  | `axis-orientation-anatomical-type` | RFC 4 orientation | Every declared spatial-axis `orientation` has `type` `anatomical`. | RFC 4: an orientation's `type` is `anatomical`. | `multiscales[0].axes` |
| 9  | `axis-orientation-on-non-space` | RFC 4 orientation | An `orientation` is declared only on `space` axes, never on a non-spatial axis. | RFC 4: orientation applies to spatial axes only. | `multiscales[0].axes[0]` |
| 10 | `axis-orientation-unique-axis` | RFC 4 orientation | No two spatial axes declare orientations describing the same anatomical axis. | RFC 4: each spatial axis describes a distinct anatomical axis. | `multiscales[0].axes` |
| 11 | `zarr-format` | images/multiscales (v0.5) | A v0.5 entry implies a Zarr v3 store; a `zarr_format` value that leaked into the entry must be exactly `3`. Inert for v0.4. | v0.5: metadata is backed by a Zarr v3 store (`zarr_format == 3`). | `multiscales[0]` |
| 12 | `ome-namespace` | images/multiscales (v0.5) | A v0.5 entry must not retain a group-level `ome` or `multiscales` wrapper key — the `ome` namespace wraps the group attributes, not each entry. Inert for v0.4. | v0.5: multiscales live under the top-level `ome` namespace, with `version` hoisted to `ome.version`. | `multiscales[0]` |
| 13 | `plate-row-index-consistency` | HCS plate | Each well's `path` is `<row>/<column>`, naming declared row/column entries, with `rowIndex`/`columnIndex` equal to those entries' positions. | v0.4: well `rowIndex`/`columnIndex` match the named row/column positions in `plate.rows`/`plate.columns`. | `plate.wells[3]` |
| 14 | `well-acquisition-missing` | HCS well | When the plate declares more than one acquisition, every well image references one via `acquisition`. | v0.4: with multiple acquisitions, each well image references an acquisition. | `well.images[0].acquisition` |

## Evaluation order and orchestrators

The rules are evaluated fail-fast: the first violation raises a
`ValidationError` and later rules do not run. Three orchestrators dispatch the
rules:

- **`validate_structural` / `validateStructural`** evaluates rules **1–12** (the
  image/multiscales rules, including the OMERO color check, the RFC 4
  orientation checks, and the two v0.5 namespacing rules). Rules 3 and 5 each
  have two internal enforcement points that share a single identifier — axis
  class-ordering then spatial-name ordering for `axis-order`; per-dataset
  scale-count then transform-ordering for
  `global-coord-transform-after-per-level`. The v0.5 namespacing rules (11 and
  12) run last and are inert for v0.4 input. The orientation rules
  `axis-orientation-on-non-space` (9) and `axis-orientation-unique-axis` (10)
  fire only for specific orientation shapes, so the linear fail-fast cascade for
  a v0.4 metadata is a 10-step sequence ending at
  `axis-orientation-anatomical-type` (identifiers 1–8).
- **`validate_plate` / `validatePlate`** evaluates rule **13**
  (`plate-row-index-consistency`).
- **`validate_well` / `validateWell`** evaluates rule **14**
  (`well-acquisition-missing`), in the context of its parent plate.

The two HCS rules (13 and 14) are deliberately absent from the image
orchestrator's order; they operate on separate metadata objects. The exact
fail-fast sequence is locked by the parity tests described in [[parity]].

## Rule categories at a glance

- **Image / multiscales** — `axis-count`, `axis-type`, `axis-order`,
  `scale-length-mismatch`, `global-coord-transform-after-per-level`,
  `dataset-order-highest-to-lowest`.
- **OMERO** — `omero-channel-color-format`.
- **RFC 4 orientation** — `axis-orientation-anatomical-type`,
  `axis-orientation-on-non-space`, `axis-orientation-unique-axis`.
- **v0.5 namespacing** — `zarr-format`, `ome-namespace` (inert for v0.4).
- **HCS plate / well** — `plate-row-index-consistency`,
  `well-acquisition-missing`.
