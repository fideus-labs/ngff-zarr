# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Rotation and affine matrices are written to, and read from, Zarr arrays."""

import json
import shutil
import sys
from pathlib import Path

import ngff_zarr as nz
import numpy as np
import pytest
import zarr
from ngff_zarr._matrix_transform_arrays import (
    externalize_matrix_transforms,
    resolve_matrix_transforms,
    write_matrix_arrays,
)
from ngff_zarr.v06.zarr_metadata import (
    Affine,
    Axis,
    Bijection,
    ByDimension,
    ByDimensionItem,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Rotation,
    Scale,
    TransformSequence,
)
from packaging import version

pytestmark = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="OME-Zarr v0.6 requires zarr-python >= 3.0.0b1",
)

#: Entries a JSON round trip through a lossy tool would round: no short
#: decimal spells them, and -0.0 compares equal to 0.0 unless the bits are
#: checked.
AFFINE = [
    [0.1, 1 / 3, 2 / 3, np.pi],
    [1e-17, 1.0, 0.30000000000000004, np.e],
    [-0.0, 0.0, 1.0, -7.5],
]

_angle = 0.1
ROTATION = [
    [np.cos(_angle), -np.sin(_angle), 0.0],
    [np.sin(_angle), np.cos(_angle), 0.0],
    [0.0, 0.0, 1.0],
]


def _bits(matrix) -> bytes:
    return np.asarray(matrix, dtype=np.float64).tobytes()


def _multiscales(transforms):
    image = nz.to_ngff_image(np.zeros((4, 4, 4), np.float32), dims=["z", "y", "x"])
    multiscales = nz.to_multiscales(image, scale_factors=[])
    intrinsic = multiscales.metadata.intrinsic_coordinate_system.name
    multiscales.metadata.coordinateSystems.append(
        CoordinateSystem(
            name="output", axes=[Axis(name=d, type="space") for d in "zyx"]
        )
    )
    for transform in transforms:
        transform.input = CoordinateSystemIdentifier(name=intrinsic)
        transform.output = CoordinateSystemIdentifier(name="output")
    multiscales.metadata.coordinateTransformations = transforms
    return multiscales


def _written_transforms(store: Path) -> list:
    root = json.loads((store / "zarr.json").read_text())
    return root["attributes"]["ome"]["multiscales"][0]["coordinateTransformations"]


def _array_document(store: Path, path: str) -> dict:
    return json.loads((store / path / "zarr.json").read_text())


@pytest.mark.parametrize("ngff_version", ["0.6", "0.9.dev1"])
def test_matrices_are_written_as_float64_arrays(tmp_path, ngff_version):
    store = tmp_path / "image.ome.zarr"
    multiscales = _multiscales(
        [
            Affine(affine=AFFINE, name="to_output"),
            TransformSequence(
                name="rotate",
                transformations=[
                    Scale(scale=[1.0, 2.0, 3.0]),
                    Rotation(rotation=ROTATION),
                ],
            ),
        ]
    )

    nz.to_ome_zarr(str(store), multiscales, version=ngff_version)

    affine, sequence = _written_transforms(store)
    assert affine["path"] == "coordinateTransformations/to_output"
    assert "affine" not in affine
    rotation = sequence["transformations"][1]
    assert rotation == {
        "type": "rotation",
        "path": "coordinateTransformations/rotation",
    }

    document = _array_document(store, "coordinateTransformations/to_output")
    assert document["data_type"] == "float64"
    assert document["shape"] == [3, 4]
    assert document["chunk_grid"]["configuration"]["chunk_shape"] == [3, 4]
    assert [codec["name"] for codec in document["codecs"]] == ["bytes"]
    assert _array_document(store, "coordinateTransformations/rotation")["shape"] == [
        3,
        3,
    ]
    group = _array_document(store, "coordinateTransformations")
    assert group["node_type"] == "group"

    stored = zarr.open_array(str(store / "coordinateTransformations/to_output"))
    assert _bits(stored[...]) == _bits(AFFINE)

    # The caller's transforms are serialized, not edited.
    assert multiscales.metadata.coordinateTransformations[0].path is None
    assert multiscales.metadata.coordinateTransformations[0].affine == AFFINE


def test_matrices_round_trip_bit_for_bit(tmp_path):
    store = tmp_path / "image.ome.zarr"
    multiscales = _multiscales(
        [
            Affine(affine=AFFINE, name="to_output"),
            Bijection(
                name="rotate",
                forward=Rotation(rotation=ROTATION),
                inverse=Rotation(rotation=np.asarray(ROTATION).T.tolist()),
            ),
        ]
    )

    nz.to_ome_zarr(str(store), multiscales, version="0.6")
    imported = nz.from_ome_zarr(str(store), validate=True)

    affine, bijection = imported.metadata.coordinateTransformations
    assert affine.path == "coordinateTransformations/to_output"
    assert _bits(affine.affine) == _bits(AFFINE)
    assert all(type(value) is float for row in affine.affine for value in row)
    assert bijection.forward.path == "coordinateTransformations/rotation"
    assert _bits(bijection.forward.rotation) == _bits(ROTATION)
    assert bijection.inverse.path == "coordinateTransformations/rotation_1"
    assert _bits(bijection.inverse.rotation) == _bits(np.asarray(ROTATION).T)

    # A rewrite of what was read keeps each matrix at the path it was read from.
    copy_store = tmp_path / "copy.ome.zarr"
    nz.to_ome_zarr(str(copy_store), imported, version="0.6")
    assert _written_transforms(copy_store) == _written_transforms(store)
    reread = nz.from_ome_zarr(str(copy_store), validate=True)
    assert _bits(reread.metadata.coordinateTransformations[0].affine) == _bits(AFFINE)


def test_ozx_archive_holds_the_matrix_arrays(tmp_path):
    archive = tmp_path / "image.ozx"
    nz.to_ome_zarr(
        str(archive),
        _multiscales([Affine(affine=AFFINE, name="to_output")]),
        version="0.6",
    )

    (affine,) = nz.from_ome_zarr(str(archive)).metadata.coordinateTransformations
    assert affine.path == "coordinateTransformations/to_output"
    assert _bits(affine.affine) == _bits(AFFINE)


def test_generated_paths():
    transforms = [
        {"type": "affine", "affine": AFFINE, "name": "to_output"},
        {
            "type": "affine",
            "affine": AFFINE,
            "path": "coordinateTransformations/affine",
        },
        {"type": "affine", "affine": AFFINE},
        {"type": "affine", "affine": AFFINE, "name": "not a node name"},
        {"type": "affine", "affine": AFFINE, "name": "__reserved"},
        {
            "type": "byDimension",
            "transformations": [
                {
                    "transformation": {"type": "rotation", "rotation": [[1.0]]},
                    "inputAxes": [0],
                    "outputAxes": [0],
                }
            ],
        },
        {"type": "affine", "affine": AFFINE, "name": "to_output"},
    ]

    arrays = externalize_matrix_transforms(transforms)

    paths = [
        transforms[0]["path"],
        transforms[1]["path"],
        transforms[2]["path"],
        transforms[3]["path"],
        transforms[4]["path"],
        transforms[5]["transformations"][0]["transformation"]["path"],
        transforms[6]["path"],
    ]
    assert paths == [
        "coordinateTransformations/to_output",
        # A path the transform names is kept, and no generated path takes it.
        "coordinateTransformations/affine",
        "coordinateTransformations/affine_1",
        "coordinateTransformations/affine_2",
        "coordinateTransformations/affine_3",
        "coordinateTransformations/rotation",
        "coordinateTransformations/to_output_1",
    ]
    assert sorted(arrays) == sorted(paths)
    assert not any(
        kind in transform for transform in transforms for kind in ("affine", "rotation")
    )


def test_a_path_without_a_matrix_is_left_to_the_store():
    transforms = [{"type": "rotation", "path": "elsewhere/rotation"}]

    assert externalize_matrix_transforms(transforms) == {}
    assert transforms == [{"type": "rotation", "path": "elsewhere/rotation"}]


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        ({"type": "affine"}, "holds no matrix and names no path"),
        ({"type": "affine", "affine": [[1.0, 0.0], [1.0]]}, "rectangular matrix"),
        ({"type": "rotation", "rotation": [1.0, 0.0]}, "2D matrix"),
        ({"type": "affine", "affine": AFFINE, "path": "../outside"}, "'..'"),
        ({"type": "affine", "affine": AFFINE, "path": "/absolute"}, "relative path"),
    ],
)
def test_unwritable_matrices_are_refused(transform, message):
    with pytest.raises(ValueError, match=message):
        externalize_matrix_transforms([transform])


def test_one_path_cannot_hold_two_matrices():
    transforms = [
        {"type": "affine", "affine": AFFINE, "path": "shared"},
        {"type": "affine", "affine": np.eye(3, 4).tolist(), "path": "shared"},
    ]
    with pytest.raises(ValueError, match="different matrices"):
        externalize_matrix_transforms(transforms)


def test_a_refused_matrix_leaves_the_store_untouched(tmp_path):
    store = tmp_path / "image.ome.zarr"
    multiscales = _multiscales([Affine(affine=AFFINE, path="../outside")])

    with pytest.raises(ValueError, match="'..'"):
        nz.to_ome_zarr(str(store), multiscales, version="0.6")
    assert not store.exists()


def test_a_path_only_matrix_written_elsewhere_is_read(tmp_path):
    """A store another tool wrote, with the matrix only in an array, reads."""
    store = tmp_path / "image.ome.zarr"
    nz.to_ome_zarr(
        str(store), _multiscales([Affine(affine=AFFINE, name="t")]), version="0.6"
    )
    shutil.rmtree(store / "coordinateTransformations")
    write_matrix_arrays(str(store), {"params/t": np.asarray(AFFINE)})
    root = json.loads((store / "zarr.json").read_text())
    root.pop("consolidated_metadata", None)
    entry = root["attributes"]["ome"]["multiscales"][0]
    entry["coordinateTransformations"][0]["path"] = "params/t"
    (store / "zarr.json").write_text(json.dumps(root))

    (affine,) = nz.from_ome_zarr(
        str(store), validate=True
    ).metadata.coordinateTransformations

    assert affine.path == "params/t"
    assert _bits(affine.affine) == _bits(AFFINE)


def test_a_missing_matrix_array_is_reported(tmp_path):
    store = tmp_path / "image.ome.zarr"
    nz.to_ome_zarr(
        str(store), _multiscales([Affine(affine=AFFINE, name="t")]), version="0.6"
    )
    shutil.rmtree(store / "coordinateTransformations")

    with pytest.raises(ValueError, match="coordinateTransformations/t"):
        nz.from_ome_zarr(str(store))


def test_resolution_is_relative_to_the_multiscales_group(tmp_path):
    write_matrix_arrays(str(tmp_path), {"image/params/r": np.asarray(ROTATION)})
    raw = [
        {
            "type": "sequence",
            "transformations": [{"type": "rotation", "path": "params/r"}],
        },
        # Malformed entries are left for the transform parser to report.
        {"type": "bijection", "forward": None, "inverse": "not a transform"},
    ]

    resolved = resolve_matrix_transforms(raw, str(tmp_path), subpath="image")

    rotation = resolved[0]["transformations"][0]
    assert rotation["path"] == "params/r"
    assert _bits(rotation["rotation"]) == _bits(ROTATION)
    assert resolved[1] == raw[1]
    # The raw attributes are not edited.
    assert "rotation" not in raw[0]["transformations"][0]


def test_a_matrix_array_that_is_not_2d_is_refused_on_read(tmp_path):
    write_matrix_arrays(str(tmp_path), {"params/v": np.asarray([1.0, 0.0, 0.0])})

    with pytest.raises(ValueError, match="2D matrix"):
        resolve_matrix_transforms(
            [{"type": "rotation", "path": "params/v"}], str(tmp_path)
        )


@pytest.mark.parametrize("path", ["../outside/r", "/outside/r", "params/./r"])
def test_a_matrix_path_outside_the_group_is_refused_on_read(tmp_path, path):
    """A crafted path cannot read an array beside the multiscales group."""
    write_matrix_arrays(str(tmp_path), {"outside/r": np.asarray(ROTATION)})

    with pytest.raises(ValueError, match="relative path inside the multiscales"):
        resolve_matrix_transforms(
            [{"type": "rotation", "path": path}], str(tmp_path), subpath="image"
        )


def _rewrite_inline(store: Path, index: int, matrix) -> None:
    """Store transform ``index`` the way earlier releases did: inline, no array."""
    root = json.loads((store / "zarr.json").read_text())
    root.pop("consolidated_metadata", None)
    entry = root["attributes"]["ome"]["multiscales"][0]
    transform = entry["coordinateTransformations"][index]
    shutil.rmtree(store / transform.pop("path"))
    transform[transform["type"]] = matrix
    (store / "zarr.json").write_text(json.dumps(root))


def test_in_place_upgrade_moves_inline_matrices_into_arrays(tmp_path):
    """A 0.6 store written with inline matrices is upgraded to the array form."""
    store = tmp_path / "image.ome.zarr"
    nz.to_ome_zarr(
        str(store),
        _multiscales(
            [
                Affine(affine=AFFINE, name="inline"),
                Rotation(rotation=ROTATION, name="stored"),
            ]
        ),
        version="0.6",
    )
    _rewrite_inline(store, 0, AFFINE)
    stored_chunk = store / "coordinateTransformations" / "stored" / "c" / "0" / "0"
    stored_mtime = stored_chunk.stat().st_mtime_ns

    nz.upgrade_ome_zarr(str(store), version="0.9.dev1")

    inline, stored = _written_transforms(store)
    assert inline["path"] == "coordinateTransformations/inline"
    assert "affine" not in inline
    assert stored["path"] == "coordinateTransformations/stored"
    # The array the source already held is left as it was.
    assert stored_chunk.stat().st_mtime_ns == stored_mtime
    upgraded = nz.from_ome_zarr(str(store), validate=True)
    affine, rotation = upgraded.metadata.coordinateTransformations
    assert _bits(affine.affine) == _bits(AFFINE)
    assert _bits(rotation.rotation) == _bits(ROTATION)


def test_in_place_upgrade_refuses_an_occupied_matrix_path(tmp_path):
    """An unrelated array at the generated path is neither replaced nor lost."""
    store = tmp_path / "image.ome.zarr"
    nz.to_ome_zarr(
        str(store), _multiscales([Affine(affine=AFFINE, name="inline")]), version="0.6"
    )
    _rewrite_inline(store, 0, AFFINE)
    write_matrix_arrays(str(store), {"coordinateTransformations/inline": np.eye(2)})
    root_before = (store / "zarr.json").read_bytes()

    with pytest.raises(ValueError, match="coordinateTransformations/inline"):
        nz.upgrade_ome_zarr(str(store), version="0.9.dev1")

    assert (store / "zarr.json").read_bytes() == root_before
    unrelated = zarr.open_array(str(store / "coordinateTransformations/inline"))
    np.testing.assert_array_equal(unrelated[...], np.eye(2))


def test_by_dimension_items_are_externalized(tmp_path):
    store = tmp_path / "image.ome.zarr"
    multiscales = _multiscales(
        [
            ByDimension(
                transformations=[
                    ByDimensionItem(
                        transformation=Rotation(rotation=[[0.0, -1.0], [1.0, 0.0]]),
                        inputAxes=[1, 2],
                        outputAxes=[1, 2],
                    ),
                    ByDimensionItem(
                        transformation=Scale(scale=[2.0]),
                        inputAxes=[0],
                        outputAxes=[0],
                    ),
                ]
            )
        ]
    )

    nz.to_ome_zarr(str(store), multiscales, version="0.6")

    (by_dimension,) = _written_transforms(store)
    item = by_dimension["transformations"][0]["transformation"]
    assert item == {"type": "rotation", "path": "coordinateTransformations/rotation"}
    (imported,) = nz.from_ome_zarr(
        str(store), validate=True
    ).metadata.coordinateTransformations
    assert imported.transformations[0].transformation.rotation == [
        [0.0, -1.0],
        [1.0, 0.0],
    ]


def _append_level(store: Path, transforms):
    """A two-level multiscales over ``store``'s level 0, declaring ``transforms``."""
    read = nz.from_ome_zarr(str(store))
    multiscales = nz.to_multiscales(read.images[0], scale_factors=[2])
    multiscales.metadata.coordinateSystems = read.metadata.coordinateSystems
    multiscales.metadata.coordinateTransformations = transforms
    for transform in transforms:
        transform.input = CoordinateSystemIdentifier(name="intrinsic")
        transform.output = CoordinateSystemIdentifier(name="output")
    return multiscales


def test_an_append_writes_the_matrices_it_declares(tmp_path):
    store = tmp_path / "image.ome.zarr"
    nz.to_ome_zarr(
        str(store), _multiscales([Affine(affine=AFFINE, name="first")]), version="0.6"
    )
    replaced = np.eye(3, 4).tolist()
    multiscales = _append_level(
        store,
        [
            # The stored root names this array as a matrix, so it is replaced.
            Affine(affine=replaced, name="first"),
            Rotation(rotation=ROTATION, name="second"),
        ],
    )

    nz.to_ome_zarr(
        str(store), multiscales, version="0.6", overwrite=False, start_level=1
    )

    appended = nz.from_ome_zarr(str(store), validate=True)
    assert len(appended.images) == 2
    affine, rotation = appended.metadata.coordinateTransformations
    assert affine.path == "coordinateTransformations/first"
    assert _bits(affine.affine) == _bits(replaced)
    assert rotation.path == "coordinateTransformations/second"
    assert _bits(rotation.rotation) == _bits(ROTATION)


@pytest.mark.parametrize(
    ("path", "occupied"), [("extra", "extra"), ("extra/inner", "extra")]
)
def test_an_append_refuses_a_matrix_path_the_store_holds(tmp_path, path, occupied):
    """A retained array the stored root does not name as a matrix is kept."""
    store = tmp_path / "image.ome.zarr"
    nz.to_ome_zarr(
        str(store), _multiscales([Affine(affine=AFFINE, name="first")]), version="0.6"
    )
    write_matrix_arrays(str(store), {"extra": np.eye(2)})
    root_before = (store / "zarr.json").read_bytes()
    multiscales = _append_level(store, [Affine(affine=AFFINE, path=path)])

    with pytest.raises(ValueError, match=f"node at '{occupied}'"):
        nz.to_ome_zarr(
            str(store), multiscales, version="0.6", overwrite=False, start_level=1
        )

    assert (store / "zarr.json").read_bytes() == root_before
    np.testing.assert_array_equal(zarr.open_array(str(store / "extra"))[...], np.eye(2))


@pytest.mark.parametrize("path", ["scale0/image", "scale0", "scale0/image/m"])
def test_a_matrix_path_overlapping_a_dataset_is_refused(tmp_path, path):
    store = tmp_path / "image.ome.zarr"
    multiscales = _multiscales([Affine(affine=AFFINE, path=path)])

    with pytest.raises(ValueError, match="overlaps the dataset array 'scale0/image'"):
        nz.to_ome_zarr(str(store), multiscales, version="0.6")
    assert not store.exists()


def test_an_interrupted_append_lists_its_matrices(tmp_path, monkeypatch):
    """The kept root names the new matrix, so its consolidated block lists it."""
    writer = sys.modules["ngff_zarr.to_ngff_zarr"]

    store = tmp_path / "image.ome.zarr"
    nz.to_ome_zarr(
        str(store), _multiscales([Affine(affine=AFFINE, name="first")]), version="0.6"
    )
    multiscales = _append_level(store, [Rotation(rotation=ROTATION, name="second")])

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(writer, "_write_array_with_zarrista", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        nz.to_ome_zarr(
            str(store), multiscales, version="0.6", overwrite=False, start_level=1
        )

    root = json.loads((store / "zarr.json").read_text())
    assert (
        "coordinateTransformations/second"
        in (root["consolidated_metadata"]["metadata"])
    )
    kept = nz.from_ome_zarr(str(store), validate=True)
    assert len(kept.images) == 1
    (rotation,) = kept.metadata.coordinateTransformations
    assert _bits(rotation.rotation) == _bits(ROTATION)


def test_itk_conversion_points_a_path_only_matrix_to_the_reader():
    transform = Affine(affine=[], path="coordinateTransformations/to_output")

    with pytest.raises(ValueError, match="from_ome_zarr loads them"):
        nz.ngff_transform_to_itk_transform(transform, ["z", "y", "x"])
