# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Scenes: images that share a spatial relationship (OME-Zarr 0.6, GH #563)."""

import json
from pathlib import Path

import numpy as np
import packaging.version
import pytest
import zarr
from ngff_zarr import (
    CoordinateSystem,
    CoordinateSystemIdentifier,
    NgffMultiscales,
    NgffScene,
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ome_zarr,
)
from ngff_zarr._zarrista_utils import create_zarrista_group
from ngff_zarr.rfc9_zip import write_store_to_zip
from ngff_zarr.scene import _scene_from_dict
from ngff_zarr.v06.zarr_metadata import (
    Axis,
    Bijection,
    Displacements,
    MapAxis,
    TransformSequence,
    Translation,
)

pytest.importorskip("jsonschema", reason="schema checks need the validate extra")

from jsonschema import ValidationError
from ngff_zarr.validate import validate as validate_ngff

FIXTURES = Path(__file__).parent / "fixtures" / "scene"


def _tile(seed: int):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 255, (16, 16), dtype=np.uint8)
    image = to_ngff_image(
        data,
        dims=["y", "x"],
        scale={"y": 0.5, "x": 0.5},
        axes_units={"y": "micrometer", "x": "micrometer"},
    )
    return data, to_multiscales(image, scale_factors=[2])


def _world() -> CoordinateSystem:
    return CoordinateSystem(
        name="world",
        axes=[
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ],
    )


def _to_world(path: str, offset: tuple[float, float]) -> Translation:
    return Translation(
        translation=list(offset),
        input=CoordinateSystemIdentifier(path=path, name="intrinsic"),
        output=CoordinateSystemIdentifier(name="world"),
        name=f"{path} to world",
    )


def _tiles_scene(paths=("tile_0", "tile_1")):
    pixels, images, transforms = {}, {}, []
    for index, path in enumerate(paths):
        pixels[path], images[path] = _tile(index)
        transforms.append(_to_world(path, (0.0, 8.0 * index)))
    scene = NgffScene(
        images=images,
        coordinateTransformations=transforms,
        coordinateSystems=[_world()],
    )
    return pixels, scene


def _root_document(store: Path) -> dict:
    return json.loads((store / "zarr.json").read_text())


def test_scene_round_trip(tmp_path):
    pixels, scene = _tiles_scene()
    store = tmp_path / "tiles.ome.zarr"
    scene.to_ome_zarr(store)

    document = _root_document(store)
    ome = document["attributes"]["ome"]
    assert ome["version"] == "0.6"
    assert ome["scene"]["coordinateSystems"][0]["name"] == "world"
    assert ome["scene"]["coordinateTransformations"][1] == {
        "type": "translation",
        "translation": [0.0, 8.0],
        "name": "tile_1 to world",
        "input": {"path": "tile_1", "name": "intrinsic"},
        "output": {"name": "world"},
    }
    validate_ngff(document["attributes"], version="0.6", model="scene")
    assert "tile_0/scale0/image" in document["consolidated_metadata"]["metadata"]

    read = NgffScene.from_ome_zarr(store, validate=True)
    assert list(read.images) == ["tile_0", "tile_1"]
    for path, data in pixels.items():
        np.testing.assert_array_equal(read.images[path].images[0].data, data)
        assert read.images[path].metadata.coordinateSystems[0].name == "intrinsic"
    assert read.coordinateSystems == [_world()]
    assert read.coordinateTransformations == scene.coordinateTransformations


@pytest.mark.skipif(
    packaging.version.parse(zarr.__version__).major < 3,
    reason="Zarr v3 stores need zarr-python >= 3",
)
def test_nested_image_paths_open_with_zarr_python(tmp_path):
    pixels, scene = _tiles_scene(("sample/instrument1", "sample/instrument2"))
    store = tmp_path / "scene.ome.zarr"
    to_ome_zarr(store, scene)

    group = zarr.open_group(store, mode="r")
    assert isinstance(group["sample"], zarr.Group)
    assert "multiscales" in group["sample/instrument2"].attrs["ome"]
    np.testing.assert_array_equal(
        group["sample/instrument1/scale0/image"][:], pixels["sample/instrument1"]
    )
    assert list(from_ome_zarr(store).images) == list(pixels)


def test_scene_in_ozx_archive_reads(tmp_path):
    pixels, scene = _tiles_scene()
    store = tmp_path / "tiles.ome.zarr"
    to_ome_zarr(store, scene)
    archive = tmp_path / "tiles.ozx"
    with pytest.raises(ValueError, match="written to a directory path"):
        to_ome_zarr(archive, scene)
    write_store_to_zip(store, archive, version="0.6")

    read = from_ome_zarr(archive)
    np.testing.assert_array_equal(
        read.images["tile_1"].images[0].data, pixels["tile_1"]
    )
    assert read.coordinateTransformations[1].input.path == "tile_1"


def test_overwrite_false_keeps_root_attributes(tmp_path):
    _, scene = _tiles_scene()
    store = tmp_path / "tiles.ome.zarr"
    create_zarrista_group(store, {"myorg:note": "kept"}, 3)
    to_ome_zarr(store, scene, overwrite=False)
    attrs = _root_document(store)["attributes"]
    assert attrs["myorg:note"] == "kept"
    assert "scene" in attrs["ome"]


def _tile_0_to(name: str, path: str | None = None) -> Translation:
    return Translation(
        translation=[0.0, 0.0],
        input=CoordinateSystemIdentifier(path="tile_0", name="intrinsic"),
        output=CoordinateSystemIdentifier(path=path, name=name),
    )


@pytest.mark.parametrize(
    ("transforms", "match"),
    [
        ([], "at least one coordinate transformation"),
        ([_to_world("tile_9", (0.0, 0.0))], "does not contain"),
        ([_tile_0_to("universe")], "does not declare"),
        ([_tile_0_to("physical", path="tile_1")], "which declares"),
        (
            [
                Translation(
                    translation=[0.0, 0.0],
                    input=CoordinateSystemIdentifier(path="tile_0"),
                    output=CoordinateSystemIdentifier(name="world"),
                )
            ],
            "must name a coordinate system",
        ),
        ([_to_world("tile_0", (0.0, 0.0))], "unconnected groups"),
    ],
)
def test_write_refuses_unresolved_or_disconnected_scenes(tmp_path, transforms, match):
    _, scene = _tiles_scene()
    scene.coordinateTransformations = transforms
    store = tmp_path / "tiles.ome.zarr"
    with pytest.raises(ValueError, match=match):
        to_ome_zarr(store, scene)
    assert not store.exists()


def test_write_refuses_duplicate_or_empty_system_names(tmp_path):
    _, scene = _tiles_scene()
    scene.coordinateSystems = [_world(), _world()]
    with pytest.raises(ValueError, match="non-empty and unique"):
        to_ome_zarr(tmp_path / "tiles.ome.zarr", scene)


def test_scene_axis_model_follows_the_version(tmp_path):
    _, scene = _tiles_scene()
    scene.coordinateSystems[0].axes = [Axis(name=n, type="space") for n in "abcdef"]
    with pytest.raises(ValueError, match=r"scene.coordinateSystems\[0\].axes"):
        to_ome_zarr(tmp_path / "v06.ome.zarr", scene)
    to_ome_zarr(tmp_path / "dev1.ome.zarr", scene, version="0.9.dev1")
    read = from_ome_zarr(tmp_path / "dev1.ome.zarr", validate=True)
    assert [axis.name for axis in read.coordinateSystems[0].axes] == list("abcdef")


def test_write_refuses_vectors_that_do_not_span_their_systems(tmp_path):
    _, scene = _tiles_scene()
    scene.coordinateTransformations[0].translation = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="3 translation values for the 2 axes"):
        to_ome_zarr(tmp_path / "tiles.ome.zarr", scene)


def test_write_refuses_transforms_the_reader_rejects(tmp_path):
    _, scene = _tiles_scene()
    scene.coordinateTransformations[0] = MapAxis(
        mapAxis=[0, 1, 2],
        input=CoordinateSystemIdentifier(path="tile_0", name="intrinsic"),
        output=CoordinateSystemIdentifier(name="world"),
    )
    with pytest.raises(ValueError, match="cannot read back.*mapAxis length 3"):
        to_ome_zarr(tmp_path / "tiles.ome.zarr", scene)


def _displacement_field():
    import dask.array as da
    from ngff_zarr import AxisType, NgffImage

    field = NgffImage(
        data=da.zeros((2, 16, 16), dtype=np.float32),
        dims=("c", "y", "x"),
        scale={"c": 1.0, "y": 0.5, "x": 0.5},
        translation={"c": 0.0, "y": 0.0, "x": 0.0},
        axes_types={"c": AxisType.Displacement},
    )
    return to_multiscales(field, scale_factors=[])


def test_scene_with_a_displacement_field_between_two_images(tmp_path):
    from ngff_zarr import from_ome_zarr

    _, scene = _tiles_scene()
    field_path = "coordinateTransformations/dfield"
    warp = Displacements(
        path=field_path,
        interpolation="linear",
        input=CoordinateSystemIdentifier(path="tile_0", name="intrinsic"),
        output=CoordinateSystemIdentifier(path="tile_1", name="intrinsic"),
    )
    scene.coordinateTransformations.append(warp)
    store = tmp_path / "registered.ome.zarr"

    with pytest.raises(ValueError, match="write those first"):
        to_ome_zarr(store, scene)
    with pytest.raises(ValueError, match="which the store does not hold"):
        to_ome_zarr(store, scene, overwrite=False)
    assert not store.exists()

    to_ome_zarr(store / field_path, _displacement_field(), version="0.6")
    to_ome_zarr(store, scene, overwrite=False)
    document = _root_document(store)
    consolidated = document["consolidated_metadata"]["metadata"]
    assert "coordinateTransformations" in consolidated
    assert f"{field_path}/scale0/image" in consolidated

    read = from_ome_zarr(store, validate=True)
    assert read.coordinateTransformations[2] == warp
    field = from_ome_zarr(store / read.coordinateTransformations[2].path)
    assert [axis.type for axis in field.metadata.coordinateSystems[0].axes] == [
        "displacement",
        "space",
        "space",
    ]


def test_write_refuses_paths_outside_the_scene(tmp_path):
    _, scene = _tiles_scene(("../tile_0", "tile_1"))
    with pytest.raises(ValueError, match="relative path below the scene group"):
        to_ome_zarr(tmp_path / "tiles.ome.zarr", scene)


def test_write_refuses_versions_before_0_6(tmp_path):
    _, scene = _tiles_scene()
    with pytest.raises(ValueError, match="0.6"):
        to_ome_zarr(tmp_path / "tiles.ome.zarr", scene, version="0.5")


def test_read_validates_references_against_the_store(tmp_path):
    _, scene = _tiles_scene()
    store = tmp_path / "tiles.ome.zarr"
    to_ome_zarr(store, scene, consolidate_metadata=False)
    document = _root_document(store)
    del document["attributes"]["ome"]["scene"]["coordinateSystems"]
    (store / "zarr.json").write_text(json.dumps(document))

    read = from_ome_zarr(store)
    assert read.coordinateSystems is None
    with pytest.raises(ValueError, match="does not declare"):
        from_ome_zarr(store, validate=True)

    document["attributes"]["ome"]["scene"]["coordinateTransformations"] = []
    (store / "zarr.json").write_text(json.dumps(document))
    with pytest.raises(ValidationError):
        from_ome_zarr(store, validate=True)


@pytest.mark.parametrize(
    "path", ["../tile_0", "..\\tile_0", "%2e%2e/tile_0", "/tile_0"]
)
def test_read_refuses_image_paths_outside_the_scene(tmp_path, path):
    _, scene = _tiles_scene()
    store = tmp_path / "tiles.ome.zarr"
    to_ome_zarr(store, scene, consolidate_metadata=False)
    document = _root_document(store)
    transforms = document["attributes"]["ome"]["scene"]["coordinateTransformations"]
    transforms[0]["input"]["path"] = path
    (store / "zarr.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="relative path below the scene group"):
        from_ome_zarr(store)


def test_the_kind_option_selects_what_a_store_holds(tmp_path):
    _, multiscales = _tile(0)
    image = tmp_path / "image.ome.zarr"
    to_ome_zarr(image, multiscales, version="0.6")
    assert isinstance(from_ome_zarr(image), NgffMultiscales)
    assert isinstance(from_ome_zarr(image, kind="multiscales"), NgffMultiscales)
    with pytest.raises(ValueError, match="holds no scene"):
        NgffScene.from_ome_zarr(image)

    _, scene = _tiles_scene()
    store = tmp_path / "tiles.ome.zarr"
    to_ome_zarr(store, scene)
    assert isinstance(from_ome_zarr(store), NgffScene)
    assert isinstance(from_ome_zarr(store, kind="scene"), NgffScene)
    with pytest.raises(ValueError, match="holds a scene"):
        from_ome_zarr(store, kind="multiscales")


def test_a_path_into_a_scene_reads_that_image(tmp_path):
    pixels, scene = _tiles_scene()
    store = tmp_path / "tiles.ome.zarr"
    to_ome_zarr(store, scene)
    image = from_ome_zarr(store / "tile_1")
    assert isinstance(image, NgffMultiscales)
    np.testing.assert_array_equal(image.images[0].data, pixels["tile_1"])


def _upstream(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())["attributes"]


def test_upstream_stitching_example():
    attrs = _upstream("scene_stitching.json")
    validate_ngff(attrs, version="0.6", model="scene")
    systems, transforms = _scene_from_dict(attrs["ome"]["scene"], "0.6")
    assert [system.name for system in systems] == ["world"]
    assert [axis.unit for axis in systems[0].axes] == ["micrometer", "micrometer"]
    assert [transform.input.path for transform in transforms] == [
        f"tile_{index}" for index in range(4)
    ]
    assert all(transform.input.name == "physical" for transform in transforms)
    assert all(transform.output.name == "world" for transform in transforms)
    assert transforms[3].translation == [276, 348]


def test_upstream_registration_example():
    attrs = _upstream("scene_registration.json")
    validate_ngff(attrs, version="0.6", model="scene")
    systems, transforms = _scene_from_dict(attrs["ome"]["scene"], "0.6")
    assert systems is None
    (bijection,) = transforms
    assert isinstance(bijection, Bijection)
    assert bijection.input == CoordinateSystemIdentifier(
        path="JRC2018F", name="physical"
    )
    assert bijection.output == CoordinateSystemIdentifier(path="FCWB", name="physical")
    assert isinstance(bijection.forward, TransformSequence)
    field = bijection.forward.transformations[0]
    assert isinstance(field, Displacements)
    assert field.path == "coordinateTransformations/dfield"
    assert field.interpolation == "linear"
