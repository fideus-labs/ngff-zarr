# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""``ngff-zarr transform-points`` against the RFC-5 test stores' expected answers."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ngff_zarr.cli import main

pytest.importorskip("itk")

_GENERATOR = Path(__file__).parents[1] / "examples" / "rfc5_test_stores.py"


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("rfc5_test_stores", _GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("rfc5")
    return out, module.main(out)


def test_every_store_maps_its_points_to_the_expected_answer(stores, capsys):
    out, manifest = stores
    assert len(manifest["cases"]) == 4
    for case in manifest["cases"]:
        main(
            [
                "transform-points",
                str(out / case["store"]),
                str(out / manifest["points"]),
            ]
        )
        result = json.loads(capsys.readouterr().out)
        assert result["dims"] == manifest["dims"]
        np.testing.assert_allclose(
            result["points"],
            case["expected"],
            atol=manifest["tolerance"],
            err_msg=case["description"],
        )


def test_points_of_the_wrong_width_are_refused():
    # Three 2-value points read against three dims would regroup into two wrong 3-value points.
    from ngff_zarr.transform_points import transform_points
    from ngff_zarr.v06.zarr_metadata import Identity

    with pytest.raises(ValueError, match="rows of 3"):
        transform_points(Identity(), [[1, 2], [3, 4], [5, 6]], ["z", "y", "x"])


def test_a_field_with_an_orientation_maps_the_points_it_maps_without_one():
    # The orientation names the anatomy; RFC-5 places the field's samples by
    # scale and translation alone. A field converted from an oriented ITK image
    # carries one, and was refused.
    from dataclasses import replace

    from ngff_zarr import NgffImage
    from ngff_zarr.rfc4 import RAS
    from ngff_zarr.transform_points import transform_points
    from ngff_zarr.v06.zarr_metadata import Displacements

    values = np.random.default_rng(2).normal(size=(3, 6, 7, 8))
    field = NgffImage(
        data=values,
        dims=("c", "z", "y", "x"),
        scale={"c": 1.0, "z": 2.0, "y": 1.5, "x": 0.5},
        translation={"c": 0.0, "z": 1.0, "y": -2.0, "x": 3.0},
        axes_types={"c": "displacement"},
    )
    oriented = replace(field, axes_orientations=RAS)
    points = [[4.0, 1.0, 5.0], [9.5, 3.25, 4.0]]
    expected = transform_points(
        Displacements(path="f"), points, ["z", "y", "x"], fields={"f": field}
    )
    got = transform_points(
        Displacements(path="f"), points, ["z", "y", "x"], fields={"f": oriented}
    )
    np.testing.assert_allclose(got, expected)
    assert not np.allclose(expected, points)


def test_a_transformation_from_a_scene_coordinate_system_maps_its_points(
    tmp_path, capsys
):
    # Its input names no image path: the axes come from the scene's own system,
    # where looking the input up among the images raised a KeyError.
    import dask.array as da
    import ngff_zarr as nz
    from ngff_zarr import CoordinateSystemIdentifier
    from ngff_zarr.v06.zarr_metadata import Affine, Axis, CoordinateSystem

    image = nz.to_multiscales(
        nz.NgffImage(
            data=da.zeros((4, 5, 6), dtype=np.float32),
            dims=("z", "y", "x"),
            scale={"z": 2.0, "y": 1.0, "x": 0.5},
            translation={"z": 0.0, "y": 0.0, "x": 0.0},
            name="fixed",
        ),
        scale_factors=[],
    )
    world = CoordinateSystem(
        name="world", axes=[Axis(name=dim, type="space") for dim in ("z", "y", "x")]
    )
    shift = Affine(
        affine=[[1, 0, 0, 1.0], [0, 1, 0, -2.0], [0, 0, 1, 0.5]],
        input=CoordinateSystemIdentifier(name="world"),
        output=CoordinateSystemIdentifier(
            path="fixed", name=image.metadata.intrinsic_coordinate_system.name
        ),
    )
    store = tmp_path / "scene.ome.zarr"
    nz.NgffScene(
        images={"fixed": image},
        coordinateSystems=[world],
        coordinateTransformations=[shift],
    ).to_ome_zarr(str(store))
    points = tmp_path / "points.json"
    points.write_text(json.dumps([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))

    main(["transform-points", str(store), str(points)])
    result = json.loads(capsys.readouterr().out)
    assert result["dims"] == ["z", "y", "x"]
    np.testing.assert_allclose(result["points"], [[1.0, -2.0, 0.5], [2.0, 0.0, 3.5]])

    # A transformation the scene does not have is a usage error, not a traceback
    # (a negative index picked one from the end).
    for index in ("1", "-1"):
        with pytest.raises(SystemExit):
            main(["transform-points", str(store), str(points), "--transform", index])
        assert "out of range" in capsys.readouterr().err


def test_a_float32_field_in_an_ozx_scene_maps_its_points(stores, tmp_path, capsys):
    # The compact form a registration stores: float32 values, zipped. A float32
    # field made a float ITK transform, which itk.transform_from_dict refused.
    import ngff_zarr as nz

    out, manifest = stores
    case = next(c for c in manifest["cases"] if "every 1 voxel" in c["description"])
    scene = nz.NgffScene.from_ome_zarr(str(out / case["store"]))
    path = scene.coordinateTransformations[0].path
    field = nz.from_ome_zarr(str(out / case["store"] / path))
    field.images[0].data = field.images[0].data.astype(np.float32)
    zipped = tmp_path / "scene.ozx"
    store = tmp_path / "scene.ome.zarr"
    nz.to_ome_zarr(str(store / path), field, version="0.6")
    scene.to_ome_zarr(str(store), overwrite=False)
    nz.write_store_to_zip(str(store), str(zipped), version="0.6")

    main(["transform-points", str(zipped), str(out / manifest["points"])])
    result = json.loads(capsys.readouterr().out)
    np.testing.assert_allclose(result["points"], case["expected"], atol=1e-4)


@pytest.mark.parametrize("dims", [("z", "y", "x"), ("x", "y", "z"), ("y", "z", "x")])
def test_points_follow_the_order_of_dims(dims):
    # ITK orders a point x, y, z by name: reversing dims holds for z, y, x only,
    # and gave other numbers, silently, in any other order.
    from ngff_zarr.transform_points import transform_points
    from ngff_zarr.v06.zarr_metadata import Affine

    shift = {"x": 1.0, "y": 2.0, "z": 3.0}
    affine = Affine(
        affine=[
            [float(row == col) for col in range(3)] + [shift[dim]]
            for row, dim in enumerate(dims)
        ]
    )
    np.testing.assert_allclose(
        transform_points(affine, [[0.0, 0.0, 0.0]], list(dims)),
        [[shift[dim] for dim in dims]],
    )
