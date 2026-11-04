# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""CLI metadata overrides on OME-Zarr input, which keeps its pyramid."""

import numpy as np
import pytest
import zarr
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr
from ngff_zarr.cli import main
from packaging import version

requires_zarr_v3 = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="OME-Zarr v0.5 output requires zarr-python >= 3.0.0b1",
)


def _write_pyramid(tmp_path, input_version="0.4"):
    image = to_ngff_image(
        np.zeros((3, 64, 64), np.uint8),
        dims=("c", "y", "x"),
        scale={"y": 0.5, "x": 0.5},
        translation={"y": 1.0, "x": 1.0},
    )
    multiscales = to_multiscales(
        image, scale_factors=[{"y": 2, "x": 2}, {"y": 4, "x": 4}]
    )
    source = tmp_path / "source.ome.zarr"
    to_ome_zarr(source, multiscales, version=input_version)
    return source


@requires_zarr_v3
@pytest.mark.parametrize("input_version", ["0.4", "0.5", "0.6"])
def test_overrides_reach_every_level_of_ome_zarr_input(tmp_path, input_version):
    source = _write_pyramid(tmp_path, input_version)
    output = tmp_path / "output.ome.zarr"

    main(
        ["-q", "-i", str(source), "-o", str(output)]
        + ["--dims", "z", "y", "x", "--scale", "x", "2", "--translation", "y", "5"]
        + ["--units", "y", "millimeter", "x", "millimeter", "--orientation", "LPS"]
        + ["--name", "renamed"]
    )

    result = from_ome_zarr(output)
    assert result.metadata.name == "renamed"
    # Each level keeps its half-pixel offset, rescaled with the first level.
    assert [(level.scale, level.translation) for level in result.images] == [
        ({"z": 1.0, "y": 0.5, "x": 2.0}, {"z": 0.0, "y": 5.0, "x": 1.0}),
        ({"z": 1.0, "y": 1.0, "x": 4.0}, {"z": 0.0, "y": 5.25, "x": 2.0}),
        ({"z": 1.0, "y": 2.0, "x": 8.0}, {"z": 0.0, "y": 5.75, "x": 4.0}),
    ]
    base = result.images[0]
    assert base.axes_units == {"z": None, "y": "millimeter", "x": "millimeter"}
    assert {dim: o.value.value for dim, o in base.axes_orientations.items()} == {
        "z": "inferior-to-superior",
        "y": "anterior-to-posterior",
        "x": "right-to-left",
    }


@requires_zarr_v3
def test_dims_count_mismatch_on_ome_zarr_input_exits(tmp_path, capsys):
    source = _write_pyramid(tmp_path)
    output = tmp_path / "output.ome.zarr"

    with pytest.raises(SystemExit):
        main(["-q", "-i", str(source), "-o", str(output), "--dims", "y", "x"])
    assert "number of dims" in capsys.readouterr().out
    assert not output.exists()
