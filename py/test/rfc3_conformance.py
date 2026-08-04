# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""RFC-3 conformance driver against the official reference datasets.

In the spirit of the RFC-4 conformance driver (``run_conformance.py``): a
manifest of authored ground truth, one case per official RFC-3 sample dataset,
diffed against what ngff-zarr actually does. It prints ``[PASS]``/``[FAIL]`` per
case and a summary line; the exit code is 0 iff every case conforms.

The official RFC-3 data is *generated* by ``clbarnes/ome-zarr-rfc3-data``
(``uv run main.py`` -> ``data/*.ome.zarr``); the files are not checked in, so
this driver is opt-in and never part of the default unit-test suite (which uses
the deterministic synthetic fixtures in ``test_rfc3_axes.py``). It downloads
nothing -- the caller supplies the directory::

    python test/rfc3_conformance.py --data-dir /path/to/ome-zarr-rfc3-data/data

Each case records the declared version, the Zarr format, the axes/shape, and the
result of: reading (``from_ngff_zarr``), structural validation
(``validate_structural``), reading every referenced array, and the
scale/translation-length check. A read failure is classified so the distinct
concerns are not conflated:

- ``version-string`` -- the reference data tags itself ``0.5+rfc3``, a string
  no specification defines and the reader's ``NgffVersion`` does not accept;
- ``rfc3-metadata`` -- the reader rejects a valid RFC-3 dataset (e.g. an axis
  with no ``type``);
- ``malformed-data`` -- the reference data is itself malformed/incomplete;
- ``storage`` -- an unrelated Zarr/storage-access error.

When the only faithful-read failure is that version string, a labelled retagged
pass (a temp copy declaring ``1.0-DEV``) characterises the axis behaviour
separately, so version and axis concerns stay distinct. ``1.0-DEV`` is the
version that adopts RFC-3: a 6-D axis model is conformant there, and rejected
at 0.5.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

VERSION_STRING = "version-string"
RFC3_METADATA = "rfc3-metadata"
MALFORMED_DATA = "malformed-data"
STORAGE = "storage"

#: Revision of clbarnes/ome-zarr-rfc3-data whose generated output the manifest
#: below describes (axes, shapes, dtypes). The data is regenerated, not
#: versioned; against another revision a shape or axis mismatch may be an
#: upstream change rather than malformed data.
RFC3_DATA_REVISION = "503323514c4c"

# Authored ground truth for the datasets produced by clbarnes/ome-zarr-rfc3-data
# (main.py, at RFC3_DATA_REVISION). ``axes`` is the exact declared axis list,
# in order; a missing ``type`` key is preserved because it is the RFC-3 point
# under test.
MANIFEST: list[dict[str, Any]] = [
    {
        "id": "ecg_1d",
        "file": "ecg_1d.ome.zarr",
        "version": "0.5+rfc3",
        "axes": [{"name": "t", "type": "time", "unit": "seconds"}],
        "ndim": 1,
        # scipy.datasets.electrocardiogram: 5 minutes sampled at 360 Hz.
        "shape": (108000,),
        "rfc3_reason": "1-D data was disallowed before RFC-3 (min 2 axes).",
    },
    {
        "id": "astronaut_xcy",
        "file": "astronaut_xcy.ome.zarr",
        "version": "0.5+rfc3",
        "axes": [
            {"name": "x", "type": "space"},
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space"},
        ],
        "ndim": 3,
        # imageio's 512x512 RGB astronaut, transposed (y, x, c) -> (x, c, y).
        "shape": (512, 3, 512),
        "rfc3_reason": "channel-before-space ordering was required before RFC-3.",
    },
    {
        "id": "ramp_6d",
        "file": "ramp_6d.ome.zarr",
        "version": "0.5+rfc3",
        "axes": [{"name": n} for n in "abcdef"],
        "ndim": 6,
        # A zero border around an 8^6 linear ramp, 16 samples per axis.
        "shape": (16,) * 6,
        "rfc3_reason": "max 5 axes, and only one untyped axis, before RFC-3.",
    },
]


def _read_group_attrs(path: Path) -> tuple[dict, int | None]:
    """Return (ome-attributes, zarr_format) from a store's group metadata."""
    zarr_json = path / "zarr.json"  # Zarr v3
    if zarr_json.is_file():
        doc = json.loads(zarr_json.read_text())
        attrs = doc.get("attributes", doc)
        return attrs.get("ome", attrs), 3
    zmetadata = path / ".zmetadata"  # Zarr v2, consolidated (authoritative)
    if zmetadata.is_file():
        doc = json.loads(zmetadata.read_text())
        attrs = doc.get("metadata", {}).get(".zattrs", {})
        return attrs.get("ome", attrs), 2
    zattrs = path / ".zattrs"  # Zarr v2
    if zattrs.is_file():
        attrs = json.loads(zattrs.read_text())
        return attrs.get("ome", attrs), 2
    return {}, None


def _classify_read_failure(message: str) -> str:
    if "is not a valid NgffVersion" in message:
        return VERSION_STRING
    if "missing required field 'type'" in message:
        return RFC3_METADATA
    return STORAGE


def _check_read_result(multiscales, expected: dict) -> dict[str, Any]:
    """Verify a successfully-read dataset against the manifest expectations."""
    import numpy as np
    from ngff_zarr import validate_structural

    result: dict[str, Any] = {}
    image = multiscales.images[0]
    result["dims"] = tuple(image.dims)
    result["shape"] = tuple(image.data.shape)
    result["dtype"] = str(image.data.dtype)

    problems: list[str] = []
    classification: str | None = None

    expected_names = tuple(ax["name"] for ax in expected["axes"])
    if tuple(image.dims) != expected_names:
        problems.append(f"axis order {tuple(image.dims)} != expected {expected_names}")
        classification = MALFORMED_DATA
    if image.data.ndim != expected["ndim"]:
        problems.append(f"ndim {image.data.ndim} != expected {expected['ndim']}")
        classification = MALFORMED_DATA
    if tuple(image.data.shape) != tuple(expected["shape"]):
        problems.append(
            f"shape {tuple(image.data.shape)} != expected {tuple(expected['shape'])}"
        )
        classification = MALFORMED_DATA

    try:
        for img in multiscales.images:
            np.asarray(img.data)
        result["arrays_read"] = "ok"
    except Exception as exc:
        result["arrays_read"] = f"fail: {type(exc).__name__}: {exc}"
        problems.append("arrays could not be read")
        classification = STORAGE  # a chunk/store error, not an RFC-3 concern

    ndim = image.data.ndim
    bad = []
    for i, dataset in enumerate(getattr(multiscales.metadata, "datasets", [])):
        for transform in getattr(dataset, "coordinateTransformations", []) or []:
            vec = getattr(transform, "scale", None) or getattr(
                transform, "translation", None
            )
            if vec is not None and len(vec) != ndim:
                bad.append(f"dataset[{i}]={len(vec)}")
    result["scale_translation_lengths"] = "ok" if not bad else f"mismatch: {bad}"
    if bad:
        problems.append("scale/translation length mismatch")
        classification = classification or RFC3_METADATA

    try:
        # Validate at the RFC-3 version. Downgrading to 0.4 would re-apply the
        # axis-count, axis-type and axis-order rules that RFC-3 lifts, so every
        # case here would fail by construction.
        validate_structural(
            multiscales.metadata.to_version("0.4"), version=RFC3_VERSION
        )
        result["validate_structural"] = "ok"
    except Exception as exc:
        result["validate_structural"] = f"fail: {type(exc).__name__}: {exc}"
        problems.append(f"validate_structural: {exc}")
        classification = classification or RFC3_METADATA

    result["problems"] = problems
    result["classification"] = classification
    return result


def _retag_attrs_version(doc: dict, version: str) -> None:
    """Rewrite the OME-Zarr version inside a group-attributes dict, in place."""
    if isinstance(doc.get("ome"), dict):  # v0.5 namespace on a v2 store
        doc["ome"]["version"] = version
    for multiscale in doc.get("multiscales", []):  # v0.4 per-entry version
        multiscale["version"] = version


def _patch_group_version(root: Path, version: str) -> None:
    """Rewrite the OME-Zarr version in a store's group metadata (v2 or v3)."""
    zarr_json = root / "zarr.json"  # Zarr v3
    if zarr_json.is_file():
        doc = json.loads(zarr_json.read_text())
        doc.setdefault("attributes", {}).setdefault("ome", {})["version"] = version
        zarr_json.write_text(json.dumps(doc))
        return
    zattrs = root / ".zattrs"  # Zarr v2
    if zattrs.is_file():
        doc = json.loads(zattrs.read_text())
        _retag_attrs_version(doc, version)
        zattrs.write_text(json.dumps(doc))
    zmetadata = root / ".zmetadata"  # consolidated copy of the v2 attributes
    if zmetadata.is_file():
        doc = json.loads(zmetadata.read_text())
        attrs = doc.get("metadata", {}).get(".zattrs")
        if isinstance(attrs, dict):
            _retag_attrs_version(attrs, version)
            zmetadata.write_text(json.dumps(doc))


#: The version an RFC-3 dataset declares. 6-D axes are conformant at 1.0-DEV
#: and rejected at 0.5, so retagging makes the reference data valid, not just
#: readable.
RFC3_VERSION = "1.0-DEV"


def _version_normalized(
    path: Path, expected: dict, axis_problems: tuple[str, ...] | list[str] = ()
) -> dict[str, Any]:
    """Copy the (small) dataset, retag it as RFC-3, and re-check the axes.

    ``axis_problems`` are declared-axis mismatches against the manifest. The
    retag rewrites the version string only, so they still apply here.
    """
    from ngff_zarr import from_ngff_zarr

    out: dict[str, Any] = {"normalized_version": RFC3_VERSION}
    tmp = Path(tempfile.mkdtemp())
    try:
        copy = tmp / path.name
        shutil.copytree(path, copy)
        _patch_group_version(copy, RFC3_VERSION)
        try:
            multiscales = from_ngff_zarr(str(copy), validate=False)
            out["read"] = "ok"
            out.update(_check_read_result(multiscales, expected))
            if axis_problems:
                out["problems"] = list(axis_problems) + out.get("problems", [])
                out["classification"] = out.get("classification") or MALFORMED_DATA
            out["status"] = "fail" if out["problems"] else "pass"
        except Exception as exc:
            out["read"] = f"fail: {type(exc).__name__}: {exc}"
            out["classification"] = _classify_read_failure(str(exc))
            out["status"] = "fail"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def run_case(data_dir: Path, case: dict) -> dict[str, Any]:
    """Read one reference dataset and diff it against its manifest entry."""
    from ngff_zarr import from_ngff_zarr

    path = data_dir / case["file"]
    report: dict[str, Any] = {"id": case["id"], "path": str(path)}
    if not path.exists():
        report.update(status="skip", reason="dataset not found")
        return report

    try:
        ome_attrs, report["zarr_format"] = _read_group_attrs(path)
    except json.JSONDecodeError as exc:
        report.update(
            status="fail",
            classification=MALFORMED_DATA,
            problems=[f"group metadata is not valid JSON: {exc}"],
        )
        return report
    except OSError as exc:
        report.update(
            status="fail",
            classification=STORAGE,
            problems=[f"group metadata could not be read: {exc}"],
        )
        return report
    report["declared_version"] = ome_attrs.get("version")
    report["declared_axes"] = (ome_attrs.get("multiscales") or [{}])[0].get("axes", [])

    # The declared metadata itself must match the authored manifest (version and
    # the full axis list, not just names), else the dataset is malformed data.
    meta_problems: list[str] = []
    axis_problems: list[str] = []
    if report["declared_version"] != case["version"]:
        meta_problems.append(
            f"version {report['declared_version']!r} != manifest {case['version']!r}"
        )
    if report["declared_axes"] != case["axes"]:
        axis_problems.append(
            f"declared axes {report['declared_axes']} != manifest {case['axes']}"
        )
    meta_problems += axis_problems

    try:
        multiscales = from_ngff_zarr(str(path), validate=False)
        report["read"] = "ok"
        checks = _check_read_result(multiscales, case)
        report.update(checks)
        report["problems"] = meta_problems + checks["problems"]
        if report["problems"]:
            report["status"] = "fail"
            report["classification"] = checks["classification"] or MALFORMED_DATA
        else:
            report["status"] = "pass"
        return report
    except Exception as exc:
        report["read"] = f"fail: {type(exc).__name__}: {exc}"
        report["status"] = "fail"
        report["classification"] = _classify_read_failure(str(exc))
        if report["classification"] == VERSION_STRING:
            report["version_normalized"] = _version_normalized(
                path, case, axis_problems
            )
        return report


def run_data_dir(data_dir: Path) -> list[dict[str, Any]]:
    """Run every manifest case against ``data_dir``; return one report each."""
    return [run_case(data_dir, case) for case in MANIFEST]


def _print(report: dict[str, Any]) -> None:
    status = report.get("status", "?").upper()
    print(f"[{status}] {report['id']}")
    print(
        f"       version={report.get('declared_version')} "
        f"zarr=v{report.get('zarr_format')} axes={report.get('declared_axes')}"
    )
    print(f"       read={report.get('read')}")
    if "dims" in report:
        print(
            f"       dims={report['dims']} shape={report['shape']} "
            f"dtype={report['dtype']}"
        )
        print(f"       validate_structural={report.get('validate_structural')}")
        print(f"       arrays_read={report.get('arrays_read')}")
        print(f"       scale_translation={report.get('scale_translation_lengths')}")
    for problem in report.get("problems", []):
        print(f"       - {problem}")
    if "classification" in report:
        print(f"       classification={report['classification']}")
    if "version_normalized" in report:
        print(f"       version_normalized={report['version_normalized']}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print per-case reports and the summary line.

    ``--data-dir`` (default: ``$RFC3_DATA_DIR``) locates the generated
    datasets and ``--json`` also writes the reports to a file. Returns exit
    status 0 iff every manifest case passes; a skipped dataset is not a pass.
    """
    parser = argparse.ArgumentParser(description="RFC-3 conformance driver.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("RFC3_DATA_DIR"),
        help="directory of generated datasets (defaults to $RFC3_DATA_DIR)",
    )
    parser.add_argument("--json", type=Path, help="also write a JSON report")
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("provide --data-dir or set RFC3_DATA_DIR")

    reports = run_data_dir(args.data_dir)
    print(
        f"RFC-3 conformance driver -- {len(reports)} cases "
        f"(manifest @ ome-zarr-rfc3-data {RFC3_DATA_REVISION})\n" + "=" * 72
    )
    for report in reports:
        _print(report)
    passed = sum(1 for r in reports if r.get("status") == "pass")
    failed = sum(1 for r in reports if r.get("status") == "fail")
    skipped = sum(1 for r in reports if r.get("status") == "skip")
    print("=" * 72)
    print(
        f"Total {len(reports)}  ·  {passed} passed  ·  {failed} failed  ·  "
        f"{skipped} skipped"
    )
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2, default=str))
    # Success only when every manifest case conforms (no failures, no skips).
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
