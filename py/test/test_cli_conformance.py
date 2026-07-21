# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for the ``ngff-zarr conformance`` RFC 4 verdict output."""

import json
import zipfile
from pathlib import Path
from typing import Any

from ngff_zarr.cli import main
from ngff_zarr.rfc4_conformance import conformance_report

Axis = dict[str, Any]


def _write_zarr(tmp_path: Path, axes: list[Axis]) -> str:
    """Write a minimal v3 ``zarr.json`` carrying the given axes; return its path."""
    metadata = {"attributes": {"ome": {"multiscales": [{"axes": axes}]}}}
    (tmp_path / "zarr.json").write_text(json.dumps(metadata))
    return str(tmp_path)


def _write_zattrs(tmp_path: Path, axes: list[Axis]) -> str:
    """Write a minimal v2 ``.zattrs`` carrying the given axes; return its dir path."""
    metadata = {"multiscales": [{"axes": axes}]}
    (tmp_path / ".zattrs").write_text(json.dumps(metadata))
    return str(tmp_path)


def _write_zarr_zip(tmp_path: Path, axes: list[Axis]) -> str:
    """Pack a minimal v3 ``zarr.json`` into a ``.zip``; return the archive path."""
    metadata = {"attributes": {"ome": {"multiscales": [{"axes": axes}]}}}
    archive = tmp_path / "image.ome.zarr.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("zarr.json", json.dumps(metadata))
    return str(archive)


def _space(name: str, value: str) -> Axis:
    return {
        "name": name,
        "type": "space",
        "orientation": {"type": "anatomical", "value": value},
    }


_LPS = (
    _space("z", "inferior-to-superior"),
    _space("y", "anterior-to-posterior"),
    _space("x", "right-to-left"),
)


def test_conformance_report_valid(tmp_path):
    path = _write_zarr(
        tmp_path,
        [
            _space("z", "inferior-to-superior"),
            _space("y", "anterior-to-posterior"),
            _space("x", "right-to-left"),
        ],
    )
    report = conformance_report(path)
    assert report["rfc4_valid"] is True
    assert report["violations"] == []
    assert report["axes"] == {
        "z": "inferior-to-superior",
        "y": "anterior-to-posterior",
        "x": "right-to-left",
    }


def test_conformance_report_orientation_on_non_space(tmp_path):
    path = _write_zarr(
        tmp_path,
        [
            {
                "name": "t",
                "type": "time",
                "orientation": {"type": "anatomical", "value": "inferior-to-superior"},
            },
            _space("z", "inferior-to-superior"),
            _space("y", "anterior-to-posterior"),
            _space("x", "right-to-left"),
        ],
    )
    report = conformance_report(path)
    assert report["rfc4_valid"] is False
    assert "axis-orientation-on-non-space" in report["violations"]


def test_conformance_report_duplicate_anatomical_axis(tmp_path):
    path = _write_zarr(
        tmp_path,
        [
            _space("z", "left-to-right"),
            _space("y", "anterior-to-posterior"),
            _space("x", "right-to-left"),
        ],
    )
    report = conformance_report(path)
    assert report["rfc4_valid"] is False
    assert "axis-orientation-unique-axis" in report["violations"]


def test_conformance_report_out_of_vocabulary(tmp_path):
    path = _write_zarr(
        tmp_path,
        [
            _space("z", "LPS"),
            _space("y", "anterior-to-posterior"),
            _space("x", "right-to-left"),
        ],
    )
    report = conformance_report(path)
    assert report["rfc4_valid"] is False
    assert report["violations"] == ["orientation-schema-invalid"]


def test_conformance_report_reads_v2_zattrs(tmp_path):
    """A v2 ``.zattrs`` layout (no ``attributes``/``ome`` wrapper) is read too."""
    path = _write_zattrs(tmp_path, list(_LPS))
    report = conformance_report(path)
    assert report["rfc4_valid"] is True
    assert report["axes"] == {
        "z": "inferior-to-superior",
        "y": "anterior-to-posterior",
        "x": "right-to-left",
    }


def test_conformance_report_reads_zip(tmp_path):
    """An OME-Zarr packed as ``.zip`` is read from its archived metadata."""
    path = _write_zarr_zip(tmp_path, list(_LPS))
    report = conformance_report(path)
    assert report["rfc4_valid"] is True
    assert report["axes"]["x"] == "right-to-left"


def test_conformance_report_empty_dir_is_invalid(tmp_path):
    """A directory without OME-Zarr metadata is invalid, not vacuously valid."""
    report = conformance_report(str(tmp_path))
    assert report["rfc4_valid"] is False
    assert report["violations"] == ["input-not-ome-zarr"]
    assert report["axes"] == {}


def test_conformance_report_zip_without_metadata_is_invalid(tmp_path):
    """A ``.zip`` carrying no root metadata is classified invalid, not valid."""
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("not-metadata.txt", "nothing here")
    report = conformance_report(str(archive))
    assert report["rfc4_valid"] is False
    assert report["violations"] == ["input-not-ome-zarr"]


def test_conformance_report_non_list_axes_is_invalid(tmp_path):
    """``axes`` present but not a list is a read failure, not empty-and-valid."""
    metadata = {"attributes": {"ome": {"multiscales": [{"axes": "oops"}]}}}
    (tmp_path / "zarr.json").write_text(json.dumps(metadata))
    report = conformance_report(str(tmp_path))
    assert report["rfc4_valid"] is False
    assert report["violations"] == ["input-not-ome-zarr"]


def test_conformance_report_malformed_json_is_invalid(tmp_path):
    """Unparsable metadata is reported as unreadable, never as valid."""
    (tmp_path / "zarr.json").write_text("{ this is not json")
    report = conformance_report(str(tmp_path))
    assert report["rfc4_valid"] is False
    assert report["violations"] == ["input-unreadable"]


def test_conformance_report_axis_missing_name_does_not_crash(tmp_path):
    """A raw axis without a name yields a verdict instead of crashing the CLI."""
    metadata = {
        "attributes": {
            "ome": {
                "multiscales": [
                    {
                        "axes": [
                            {
                                "type": "space",
                                "orientation": {
                                    "type": "anatomical",
                                    "value": "left-to-right",
                                },
                            }
                        ]
                    }
                ]
            }
        }
    }
    (tmp_path / "zarr.json").write_text(json.dumps(metadata))
    report = conformance_report(str(tmp_path))
    assert report["format"] == "ome-zarr"
    assert report["rfc4_valid"] is False
    # The unnamed axis is dropped from the display map but is still surfaced as a
    # violation by validate_rfc4_orientation, classified per the conformance
    # contract as a schema-level failure.
    assert report["axes"] == {}
    assert report["violations"] == ["orientation-schema-invalid"]


def test_conformance_cli_prints_json(tmp_path, monkeypatch, capsys):
    path = _write_zarr(
        tmp_path,
        [
            _space("z", "inferior-to-superior"),
            _space("y", "anterior-to-posterior"),
            _space("x", "right-to-left"),
        ],
    )
    monkeypatch.setattr("sys.argv", ["ngff-zarr", "conformance", path])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["rfc4_valid"] is True
    assert report["format"] == "ome-zarr"
