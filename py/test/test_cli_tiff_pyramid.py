# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""CLI metadata overrides reach the output of a reused TIFF pyramid (#799)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


@pytest.mark.parametrize("axes", ["YX", "TYX"])
def test_cli_overrides_apply_to_reused_tiff_pyramid(tmp_path, axes):
    lead = (2,) * (len(axes) - 2)
    input_path = tmp_path / "pyramid.ome.tiff"
    with tifffile.TiffWriter(input_path, ome=True) as tif:
        tif.write(
            np.zeros((*lead, 256, 256), np.uint8),
            metadata={"axes": axes},
            tile=(128, 128),
            subifds=1,
        )
        tif.write(np.zeros((*lead, 128, 128), np.uint8), tile=(128, 128), subfiletype=1)
    output_path = tmp_path / "out.ome.zarr"

    cmd = [
        sys.executable,
        "-c",
        "from ngff_zarr.cli import main; main()",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "--scale",
        "x",
        "2.5",
        "y",
        "2.5",
        "--translation",
        "x",
        "10",
        "--units",
        "x",
        "micrometer",
        "y",
        "micrometer",
        "--name",
        "renamed",
        "--orientation",
        "LPS",
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    zarr_json = json.loads((output_path / "zarr.json").read_text())
    multiscales = zarr_json["attributes"]["ome"]["multiscales"][0]
    assert multiscales["name"] == "renamed"
    by_name = {axis["name"]: axis for axis in multiscales["axes"]}
    assert list(by_name) == [*("t" * len(lead)), "y", "x"]
    assert by_name["x"]["unit"] == by_name["y"]["unit"] == "micrometer"
    assert by_name["x"]["orientation"]["value"] == "right-to-left"
    transforms = [d["coordinateTransformations"] for d in multiscales["datasets"]]
    # Non-spatial axes keep the base scale and translation on every level.
    scale, translation = [1.0] * len(lead), [0.0] * len(lead)
    assert [t[0]["scale"] for t in transforms] == [
        [*scale, 2.5, 2.5],
        [*scale, 5.0, 5.0],
    ]
    # The reused level's half-pixel shift is computed from the overridden base.
    assert [t[1]["translation"] for t in transforms] == [
        [*translation, 0.0, 10.0],
        [*translation, 1.25, 11.25],
    ]
