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
no-op) for v0.4. The three RFC 4 orientation rules (9–11) are normative from
OME-Zarr 0.9.dev1, which incorporates RFC-4 through `ome/ngff-spec#190` (and
from 0.9.dev3, which extends it): they are inert when the caller declares 0.4,
0.5 or 0.6, where RFC 4 has no normative status, and stay on when no version
is declared — a strictness choice, like `axis-names-unique` below 0.9.dev1.
The ten RFC 8 node/collection rules (16–25) are normative from OME-Zarr
0.9.dev3, the version that stores the RFC-8 node model. For the conceptual background and the two
validation levels, see [[overview]]; for invocation, see [[api]].

Location strings are dotted-segment, JSON-Pointer-style identifiers of the
offending metadata node, emitted byte-for-byte identically in both languages.

## Canonical rule table

The rules are listed in canonical spec-MUST evaluation order — the same order
the `SpecRule` enum declares them and the orchestrators evaluate them.

| #  | Identifier | Applies to | What it checks | Spec MUST | Example location |
| -- | ---------- | ---------- | -------------- | --------- | ---------------- |
| 1  | `axis-count` | images/multiscales | The number of axes is within the closed range 2–5. Inert for 0.9.dev1. | v0.4: between 2 and 5 axes, inclusive. | `multiscales[0].axes` |
| 2  | `axis-type` | images/multiscales | At most one `time` axis and at most one `channel` axis. Inert for 0.9.dev1. | v0.4: ≤ 1 time axis and ≤ 1 channel axis. | `multiscales[0].axes` |
| 3  | `axis-order` | images/multiscales | Axes ordered time → channel → space; 2 or 3 space axes; the space-axis names are the length-N suffix of `(z, y, x)`. Inert for 0.9.dev1; a v0.6 `array` coordinate system is exempt from the two-space-axis floor, but not from the three-axis cap or the suffix check. | v0.4: axes ordered time, then channel, then space, with 2 or 3 `space` axes whose names are a suffix of `(z, y, x)`. | `multiscales[0].axes[1]` |
| 4  | `axis-names-unique` | images/multiscales | No two axes share a `name`. Never inert. | RFC-3 rule 5: axis names MUST NOT be repeated within a dataset. No released schema states it, so below 0.9.dev1 this is a strictness choice, not a spec MUST of those versions. | `multiscales[0].axes[2]` |
| 5 | `scale-length-mismatch` | images/multiscales | Every `scale`/`translation` vector length equals the axis count, for both the global and per-dataset transforms. At v0.6 the multiscale-level transforms map between named coordinate systems, so only the per-dataset ones are measured. | v0.4: each scale/translation has exactly one entry per axis. | `multiscales[0].datasets[2].coordinateTransformations[0]` |
| 6 | `global-coord-transform-after-per-level` | images/multiscales | Exactly one `scale` per dataset, and a `translation` must follow — not precede — its `scale`. | v0.4: each dataset defines exactly one scale; a translation follows its scale. | `multiscales[0].datasets[1].coordinateTransformations` |
| 7 | `dataset-order-highest-to-lowest` | images/multiscales | Datasets ordered finest → coarsest; the spatial scale must not decrease as the level index rises. | v0.4: multiscale datasets ordered from highest to lowest resolution. | `multiscales[0].datasets[2]` |
| 8 | `omero-channel-color-format` | OMERO | Each OMERO channel `color` is exactly six hexadecimal digits (RGB). | v0.4: OMERO channel color is 6 hex digits. | `multiscales[0].omero.channels[0].color` |
| 9 | `axis-orientation-anatomical-type` | RFC 4 orientation | Every declared spatial-axis `orientation` has `type` `anatomical`. Inert below 0.9.dev1 when the caller declares a version; kept on when none is given. | RFC 4 (normative from 0.9.dev1): an orientation's `type` is `anatomical`. No released spec below 0.9.dev1 adopts RFC 4, so enforcement without a declared version is a strictness choice, not a spec MUST of those versions. | `multiscales[0].axes` |
| 10| `axis-orientation-on-non-space` | RFC 4 orientation | An `orientation` is declared only on `space` axes, never on a non-spatial axis. Inert below 0.9.dev1 when the caller declares a version; kept on when none is given. | RFC 4 (normative from 0.9.dev1): orientation applies to spatial axes only. Same below-0.9.dev1 status as rule 9. | `multiscales[0].axes[0]` |
| 11| `axis-orientation-unique-axis` | RFC 4 orientation | No two spatial axes declare orientations describing the same anatomical axis. Inert below 0.9.dev1 when the caller declares a version; kept on when none is given. | RFC 4 (normative from 0.9.dev1): each spatial axis describes a distinct anatomical axis. Same below-0.9.dev1 status as rule 9. | `multiscales[0].axes` |
| 12| `zarr-format` | images/multiscales (v0.5) | A v0.5 entry implies a Zarr v3 store; a `zarr_format` value that leaked into the entry must be exactly `3`. Inert for v0.4. | v0.5: metadata is backed by a Zarr v3 store (`zarr_format == 3`). | `multiscales[0]` |
| 13| `ome-namespace` | images/multiscales (v0.5) | A v0.5 entry must not retain a group-level `ome` or `multiscales` wrapper key — the `ome` namespace wraps the group attributes, not each entry. Inert for v0.4. | v0.5: multiscales live under the top-level `ome` namespace, with `version` hoisted to `ome.version`. | `multiscales[0]` |
| 14| `plate-row-index-consistency` | HCS plate | Each well's `path` is `<row>/<column>`, naming declared row/column entries, with `rowIndex`/`columnIndex` equal to those entries' positions. | v0.4: well `rowIndex`/`columnIndex` match the named row/column positions in `plate.rows`/`plate.columns`. | `plate.wells[3]` |
| 15| `well-acquisition-missing` | HCS well | When the plate declares more than one acquisition, every well image references one via `acquisition`. | v0.4: with multiple acquisitions, each well image references an acquisition. | `well.images[0].acquisition` |
| 16| `node-type-required` | RFC-8 nodes | Every node of a 0.9.dev3 document declares a non-empty string `type`. Enforced when the document declares 0.9.dev3 or no version; inert below. | RFC-8 (normative from 0.9.dev3): a node's `type` MUST be a string identifying the node type. | `ome.nodes[1]` |
| 17| `node-name-required` | RFC-8 nodes | Every node declares a non-empty string `name` for human-readable display. Same gating as rule 16. | RFC-8: a node's `name` MUST be a non-empty string. | `ome.nodes[1]` |
| 18| `node-name-unique` | RFC-8 nodes | Sibling nodes within one collection carry distinct names; nodes of different collections may share one. Same gating as rule 16. | RFC-8: names MUST be unique within the enclosing collection. | `ome.nodes[2]` |
| 19| `node-id-format` | RFC-8 nodes | Every declared `id` (node ids today, reference ids when references land) matches `[a-zA-Z0-9-_.]+`. Same gating as rule 16. | RFC-8: an id MUST be a string matching `[a-zA-Z0-9-_.]+`. | `ome.nodes[0].id` |
| 20| `node-id-unique` | RFC-8 nodes | Ids are unique within the JSON document (path-referenced nodes live in their own documents). Same gating as rule 16. | RFC-8: ids MUST be unique within the JSON document. | `ome.nodes[3].id` |
| 21| `node-nodes-xor-path` | RFC-8 collections | A collection carries exactly one of `nodes` and `path`; unknown and extension node types are exempt. Same gating as rule 16. | RFC-8: either `nodes` or `path` MUST be present on a collection, but not both. | `ome.nodes[1]` |
| 22| `path-type-known` | RFC-8 paths | A path `type` is `zarr`, `json`, or a prefixed extension identifier (`prefix:name`), which passes as opaque. Same gating as rule 16. | RFC-8: unprefixed path types are reserved for the core specification, which defines `zarr` and `json`. | `ome.nodes[1].path.type` |
| 23| `coordinate-system-id-required` | RFC-8 coordinate systems | Every coordinate system declared under a node's `attributes.coordinateSystems` carries an `id` (the `name` becomes an optional label). Same gating as rule 16. | RFC-8: coordinate systems are referenced by id, not by name. | `ome.nodes[0].attributes.coordinateSystems[1]` |
| 24| `reference-id-required` | RFC-8 references | Every top-level transformation `input`/`output` under `attributes.coordinateTransformations` is an object carrying an `id`; wrapped transforms inside a sequence are exempt. Same gating as rule 16. | RFC-8: transformation inputs and outputs are References. | `ome.nodes[0].attributes.coordinateTransformations[0].input` |
| 25| `reference-path-required` | RFC-8 references | A reference whose `id` is not declared in this document (as a node id or coordinate-system id) carries a `path` locating its document; what the path points at is not resolved. Same gating as rule 16. | RFC-8: for external references, the path field MUST be present. | `ome.nodes[0].attributes.coordinateTransformations[0].output` |

## Evaluation order and orchestrators

The rules are evaluated fail-fast: the first violation raises a
`ValidationError` and later rules do not run. Three orchestrators dispatch the
rules:

- **`validate_structural` / `validateStructural`** evaluates rules **1–13** (the
  image/multiscales rules, including the OMERO color check, the RFC 4
  orientation checks, and the two v0.5 namespacing rules). Rules 3 and 6 each
  have two internal enforcement points that share a single identifier — axis
  class-ordering then spatial-name ordering for `axis-order`; per-dataset
  scale-count then transform-ordering for
  `global-coord-transform-after-per-level`. The v0.5 namespacing rules (12 and
  13) run last and are inert for v0.4 input, and the orientation rules (9–11)
  are inert when the caller declares a version below 0.9.dev1. The orientation
  rules `axis-orientation-on-non-space` (10) and `axis-orientation-unique-axis`
  (11) fire only for specific axis shapes, so the linear fail-fast cascade for
  a v0.4-shaped metadata validated with no declared version is an 11-step
  sequence ending at `axis-orientation-anatomical-type`.
- **`validate_plate` / `validatePlate`** evaluates rule **14**
  (`plate-row-index-consistency`).
- **`validate_well` / `validateWell`** evaluates rule **15**
  (`well-acquisition-missing`), in the context of its parent plate.
- **`validate_collection` / `validateCollection`** evaluates rules **16–25**
  (the RFC-8 node/collection rules) against the raw `ome` document of an
  OME-Zarr 0.9.dev3 store or standalone JSON file, walking the node tree
  depth-first once per rule. The rules are enforced when the document
  declares 0.9.dev3 or no version, and inert when an earlier version is
  declared (RFC-8 lands at 0.9.dev3; see `is_rfc8_node_model`).

The HCS rules (14 and 15) and the RFC-8 rules (16–25) are deliberately absent
from the image orchestrator's order; they operate on separate metadata
objects. The exact fail-fast sequence is locked by the parity tests described
in [[parity]].

A fourth dispatcher sits on the write path: `_gate_axis_model` /
`gateAxisModel` refuses to serialize an axis model the target version cannot
express. It is **not** the same pass as `validate_structural`. It runs rules
**1–4** only, and it applies them to *every* coordinate system the target
version serializes, where the structural orchestrator reads one flat axis list.
It sees only what will be written: a v0.4/v0.5 target writes a single flat
`axes` and drops the coordinate systems, so a system the downgrade discards is
not gated. Python reaches that set by gating after `Metadata.to_version`;
TypeScript by checking the target version in `axisViews`.

## Rule categories at a glance

- **Image / multiscales** — `axis-count`, `axis-type`, `axis-order`,
  `axis-names-unique`, `scale-length-mismatch`,
  `global-coord-transform-after-per-level`,
  `dataset-order-highest-to-lowest`.
- **OMERO** — `omero-channel-color-format`.
- **RFC 4 orientation** (normative from 0.9.dev1) —
  `axis-orientation-anatomical-type`, `axis-orientation-on-non-space`,
  `axis-orientation-unique-axis`.
- **v0.5 namespacing** — `zarr-format`, `ome-namespace` (inert for v0.4).
- **HCS plate / well** — `plate-row-index-consistency`,
  `well-acquisition-missing`.
- **RFC 8 nodes / collections** (normative from 0.9.dev3) —
  `node-type-required`, `node-name-required`, `node-name-unique`,
  `node-id-format`, `node-id-unique`, `node-nodes-xor-path`,
  `path-type-known`, `coordinate-system-id-required`,
  `reference-id-required`, `reference-path-required`.
