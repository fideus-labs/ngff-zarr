# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pytest
import zarr
from ngff_zarr import (
    NgffMultiscales,
    from_ngff_zarr,
    from_ome_zarr,
    to_multiscales,
    to_ngff_image,
    to_ngff_zarr,
    to_ome_zarr,
    validate,
)
from packaging import version

zarr_version = version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

# OME-Zarr v0.5 and v0.6 require a Zarr v3 store, which zarr-python only writes
# at >= 3.0.0b1. The CI matrix exercises legs pinned to zarr 2.x (v0.4 only),
# where ``to_ngff_zarr(..., version="0.5")`` raises; skip those tests there.
requires_zarr_v3 = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"),
    reason="OME-Zarr v0.5 and v0.6 require zarr-python >= 3.0.0b1",
)


def check_valid_ngff(multiscale: NgffMultiscales, tmp_path: Path):
    store = tmp_path / "test.ome.zarr"
    version = "0.4"
    to_ngff_zarr(store, multiscale, version=version)
    format_kwargs = {}
    if version and zarr_version_major >= 3:
        format_kwargs = {"zarr_format": 2} if version == "0.4" else {"zarr_format": 3}
    root = zarr.open_group(store, mode="r", **format_kwargs)

    validate(root.attrs.asdict())
    # Need to add NGFF metadata property
    # validate(ngff, strict=True)

    from_ngff_zarr(store, validate=True, version=version)


def test_y_x_valid_ngff(tmp_path):
    array = np.random.random((32, 16))
    multiscale = to_multiscales(array, [2, 4])

    check_valid_ngff(multiscale, tmp_path)


def test_validate_0_1():
    test_store = Path(__file__).parent / "data" / "input" / "v01" / "6001251.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.1")
    # The read path normalizes byte order to the host's: zarr-python keeps the
    # explicit stored order character ("<"/">"), zarrista decodes straight to
    # native ("="); both are the native dtype.
    assert multiscales.images[0].data.dtype.isnative


def test_validate_0_1_no_version():
    test_store = Path(__file__).parent / "data" / "input" / "v01" / "6001251.zarr"
    from_ngff_zarr(test_store, validate=True, version="0.1")


def test_validate_0_2():
    test_store = Path(__file__).parent / "data" / "input" / "v02" / "6001240.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.2")
    print(multiscales)


def test_validate_0_2_no_version():
    test_store = Path(__file__).parent / "data" / "input" / "v02" / "6001240.zarr"
    from_ngff_zarr(test_store, validate=True)


def test_validate_0_3():
    test_store = Path(__file__).parent / "data" / "input" / "v03" / "9528933.zarr"
    multiscales = from_ngff_zarr(test_store, validate=True, version="0.3")
    print(multiscales)


def test_validate_0_3_no_version():
    test_store = Path(__file__).parent / "data" / "input" / "v03" / "9528933.zarr"
    from_ngff_zarr(test_store, validate=True)


def _write_valid_2d_store_v05(store: Path) -> Path:
    """Write a valid 2D ``(y, x)`` two-level v0.5 multiscales to a store."""
    array = np.random.random((32, 16)).astype("float32")
    to_ngff_zarr(store, to_multiscales(array, [2]), version="0.5")
    return store


@requires_zarr_v3
def test_validate_v05_wrapped_instance(tmp_path):
    # The public ``validate`` entry point checks a v0.5 store against the v0.5
    # image schema, which requires the ``ome`` namespace wrapper. The full root
    # attributes (``{"ome": {...}}``) validate; the unwrapped ``ome`` content
    # does not (it lacks the required top-level ``ome`` wrapper), proving the
    # vendored v0.5 schema -- not the v0.4 schema -- is what runs.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_2d_store_v05(tmp_path / "v05.ome.zarr")
    root_attrs = zarr.open_group(store, mode="r").attrs.asdict()

    validate(root_attrs, version="0.5", model="image")
    with pytest.raises(jsonschema.ValidationError):
        validate(root_attrs["ome"], version="0.5", model="image")


@requires_zarr_v3
def test_validate_v05_schema_active_on_read_path(tmp_path):
    # The read path validates the ``ome``-wrapped instance against the v0.5
    # image schema. A duplicate multiscale entry violates the schema's
    # ``uniqueItems`` constraint -- a pure schema concern the structural rules
    # (which inspect only ``multiscales[0]``) do not check -- so it is rejected
    # under ``validate=True`` and read silently under ``validate=False``.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_2d_store_v05(tmp_path / "v05.ome.zarr")
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    ome["multiscales"] = [ome["multiscales"][0], ome["multiscales"][0]]
    root.attrs["ome"] = ome

    with pytest.raises(jsonschema.ValidationError):
        from_ngff_zarr(store, validate=True)

    multiscales = from_ngff_zarr(store, validate=False)
    assert multiscales is not None


@requires_zarr_v3
def test_read_path_schema_instance_by_version(tmp_path):
    # The v0.4 and v0.5 image schemas share identical multiscale definitions and
    # differ only in the ``ome`` wrapper, so the rewire is behavior-neutral on
    # realistic inputs; this spy locks the wiring directly. The read path must
    # hand the schema validator the ``ome``-wrapped instance tagged version
    # "0.5" for a v0.5 store, and the bare root tagged "0.4" for a v0.4 store.
    pytest.importorskip("jsonschema")
    from unittest import mock

    def _schema_call(store):
        with mock.patch("ngff_zarr.validate.validate") as spy:
            from_ngff_zarr(store, validate=True)
        assert spy.call_count >= 1
        args, kwargs = spy.call_args
        instance = args[0] if args else kwargs["ngff_dict"]
        version_arg = args[1] if len(args) > 1 else kwargs["version"]
        return instance, version_arg

    # v0.5: wrapped under ``ome``, tagged "0.5".
    v05_instance, v05_version = _schema_call(
        _write_valid_2d_store_v05(tmp_path / "v05.ome.zarr")
    )
    assert "ome" in v05_instance
    assert "multiscales" in v05_instance["ome"]
    assert v05_version == "0.5"

    # v0.4: bare root with top-level ``multiscales``, tagged "0.4".
    store_v04 = tmp_path / "v04.ome.zarr"
    array = np.random.random((32, 16)).astype("float32")
    to_ngff_zarr(store_v04, to_multiscales(array, [2]), version="0.4")
    v04_instance, v04_version = _schema_call(store_v04)
    assert "multiscales" in v04_instance
    assert "ome" not in v04_instance
    assert v04_version == "0.4"


def _write_valid_3d_store_v06(store: Path) -> Path:
    """Write a valid 3D ``(z, y, x)`` two-level v0.6 multiscales to a store."""
    array = np.random.random((4, 8, 8)).astype("float32")
    to_ngff_zarr(store, to_multiscales(array, [2]), version="0.6")
    return store


@requires_zarr_v3
def test_validate_v06_resolves_split_schema_refs(tmp_path):
    # From v0.6 the image schema reaches coordinate systems and coordinate
    # transformations through the absolute ``$id`` URLs of sibling schema
    # files. Validation is offline, so every bundled file has to be in the
    # reference registry; without them this raised an unresolvable-reference
    # error on the writer's own output rather than validating it.
    pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "v06.ome.zarr")
    root_attrs = zarr.open_group(store, mode="r").attrs.asdict()

    validate(root_attrs, version="0.6", model="image")


@requires_zarr_v3
def test_validate_v06_accepts_the_on_disk_version_string(tmp_path):
    # A v0.6 store records the upstream pre-release tag the bundled schemas
    # carry. That string has no ``spec`` tree of its own, so it has to resolve
    # to the tree of the release it leads to.
    pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "v06.ome.zarr")
    root_attrs = zarr.open_group(store, mode="r").attrs.asdict()
    on_disk_version = root_attrs["ome"]["version"]
    assert on_disk_version.startswith("0.6")
    assert on_disk_version != "0.6"

    validate(root_attrs, version=on_disk_version, model="image")


@requires_zarr_v3
def test_validate_v06_rejects_an_earlier_prerelease_tag(tmp_path):
    # The bundled 0.6 schemas pin ``ome.version`` to the pre-release they were
    # published with, and the schema API checks a document as given. The
    # reader is the lenient one: see the warning test below.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "image.ome.zarr")
    root_attrs = zarr.open_group(str(store), mode="r").attrs.asdict()
    root_attrs["ome"]["version"] = "0.6.dev4"

    with pytest.raises(jsonschema.ValidationError, match="0.6rc0"):
        validate(root_attrs, version="0.6", model="image")


@requires_zarr_v3
def test_v06_write_refuses_a_top_level_transform_without_references(tmp_path):
    # From 0.6 a multiscale-level transform maps between two named coordinate
    # systems and the schema requires both references. The writer refuses the
    # model up front rather than producing a store its own validated reader
    # rejects; with the references present the store validates.
    from ngff_zarr.v06.zarr_metadata import CoordinateSystemIdentifier, Scale

    array = np.random.random((4, 8, 8)).astype("float32")
    multiscales = to_multiscales(array, [2])
    multiscales.metadata.coordinateTransformations = [Scale(scale=[2.0, 2.0, 2.0])]

    with pytest.raises(ValueError, match="names no input coordinate system"):
        to_ome_zarr(tmp_path / "refused.ome.zarr", multiscales, version="0.6")

    intrinsic = multiscales.metadata.intrinsic_coordinate_system.name
    multiscales.metadata.coordinateTransformations = [
        Scale(
            scale=[2.0, 2.0, 2.0],
            input=CoordinateSystemIdentifier(name=intrinsic),
            output=CoordinateSystemIdentifier(name=intrinsic),
        )
    ]
    store = tmp_path / "written.ome.zarr"
    to_ome_zarr(store, multiscales, version="0.6")
    pytest.importorskip("jsonschema")
    validate(zarr.open_group(str(store), mode="r").attrs.asdict(), version="0.6")


@requires_zarr_v3
def test_read_warns_on_a_superseded_0_6_tag_and_validates_the_rest(tmp_path):
    # A store tagged with an earlier 0.6 pre-release differs from a valid store
    # in that string alone. The validating reader says so and checks the rest
    # of the document with the tag substituted, rather than fail on the one
    # thing ``upgrade_ome_zarr`` exists to rewrite.
    pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "image.ome.zarr")
    root = zarr.open_group(str(store), mode="r+")
    ome = dict(root.attrs["ome"])
    ome["version"] = "0.6.dev4"
    root.attrs["ome"] = ome

    with pytest.warns(UserWarning, match="superseded 0.6 pre-release tag '0.6.dev4'"):
        multiscales = from_ome_zarr(store, validate=True)
    assert len(multiscales.images) == 2


@requires_zarr_v3
def test_read_does_not_substitute_a_tag_no_release_wrote(tmp_path):
    # Only the tags earlier releases wrote are substituted. A plain ``0.6``,
    # which a stricter writer might record, is checked as given, and the
    # schema rejects it, so it is not passed off as the vendored pre-release.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "image.ome.zarr")
    root = zarr.open_group(str(store), mode="r+")
    ome = dict(root.attrs["ome"])
    ome["version"] = "0.6"
    root.attrs["ome"] = ome

    with pytest.raises(jsonschema.ValidationError, match="'0.6' is not one of"):
        from_ome_zarr(store, validate=True)


@requires_zarr_v3
def test_read_still_rejects_a_defect_behind_a_superseded_0_6_tag(tmp_path):
    # The substitution covers the tag and nothing else.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "image.ome.zarr")
    root = zarr.open_group(str(store), mode="r+")
    ome = dict(root.attrs["ome"])
    ome["version"] = "0.6.dev4"
    del ome["multiscales"][0]["coordinateSystems"]
    root.attrs["ome"] = ome

    with (
        pytest.warns(UserWarning, match="superseded"),
        pytest.raises(jsonschema.ValidationError),
    ):
        from_ome_zarr(store, validate=True)


@requires_zarr_v3
def test_validate_v06_rejects_invalid_metadata(tmp_path):
    # Resolving the references must not turn validation into a no-op: dropping
    # a required property still fails.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "v06.ome.zarr")
    root_attrs = zarr.open_group(store, mode="r").attrs.asdict()
    del root_attrs["ome"]["multiscales"][0]["coordinateSystems"]

    with pytest.raises(jsonschema.ValidationError):
        validate(root_attrs, version="0.6", model="image")


@requires_zarr_v3
def test_validate_v06_schema_active_on_read_path(tmp_path):
    # The v0.6 read path validated nothing while the split-schema references
    # were unresolvable. As for v0.5, a duplicate multiscale entry violates the
    # schema's ``uniqueItems`` constraint -- a pure schema concern the
    # structural rules (which inspect only ``multiscales[0]``) do not check --
    # so it is rejected under ``validate=True`` and read silently otherwise.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "v06.ome.zarr")
    assert from_ngff_zarr(store, validate=True) is not None

    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    ome["multiscales"] = [ome["multiscales"][0], ome["multiscales"][0]]
    root.attrs["ome"] = ome

    with pytest.raises(jsonschema.ValidationError):
        from_ngff_zarr(store, validate=True)

    multiscales = from_ngff_zarr(store, validate=False)
    assert multiscales is not None


@pytest.mark.parametrize(
    "ngff_version",
    [
        "0.4",
        pytest.param("0.5", marks=requires_zarr_v3),
        pytest.param("0.6", marks=requires_zarr_v3),
    ],
)
def test_validate_strict_image_schema(ngff_version, tmp_path):
    # Every ``strict_image`` schema wraps its base schema by absolute ``$id``
    # URL, and the pre-0.6 ones omit ``$schema`` entirely. Both have to be
    # handled for the strict models to run at all.
    pytest.importorskip("jsonschema")

    store = tmp_path / "strict.ome.zarr"
    array = np.random.random((4, 8, 8)).astype("float32")
    to_ngff_zarr(store, to_multiscales(array, [2]), version=ngff_version)
    root_attrs = zarr.open_group(store, mode="r").attrs.asdict()

    validate(root_attrs, version=ngff_version, model="image", strict=True)


def test_load_schema_rejects_unbundled_version():
    # The version reaches the loader straight from a store's own metadata, so
    # it is matched against the bundled directory names rather than joined onto
    # the path as given. An unbundled version names the ones that are bundled
    # instead of surfacing a filesystem path.
    from ngff_zarr.validate import load_schema

    for unbundled in ("0.7", "latest", "", "../../../../etc", "/etc"):
        with pytest.raises(ValueError) as excinfo:
            load_schema(version=unbundled)
        message = str(excinfo.value)
        assert "0.4" in message
        assert "spec/" not in message


@requires_zarr_v3
def test_read_path_rejects_a_forged_version_string(tmp_path):
    # ``from_ngff_zarr(store, version="0.6")`` bypasses version detection, so a
    # store's own ``ome.version`` is what selects the schema. A forged value
    # must be rejected by name, not resolved as a path.
    pytest.importorskip("jsonschema")

    store = _write_valid_3d_store_v06(tmp_path / "v06.ome.zarr")
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    ome["version"] = "../../../../../../etc"
    root.attrs["ome"] = ome

    with pytest.raises(ValueError, match="No JSON Schema is bundled"):
        from_ngff_zarr(store, validate=True, version="0.6")


@pytest.mark.parametrize("values", [[2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
def test_a_transform_that_does_not_span_its_axes_is_refused(tmp_path, values):
    # The 0.4 and 0.5 models validate nothing of their own and the bundled
    # schemas constrain what these vectors hold, not how many: a scale of two
    # values over three axes was written, read back, and passed validate(),
    # while no consumer could apply it. Short and long both.
    import dataclasses

    from ngff_zarr.v04.zarr_metadata import Scale as ScaleV04

    array = np.random.random((4, 8, 8)).astype("float32")
    multiscales = to_multiscales(array, [])
    metadata = multiscales.metadata.to_version("0.4")
    metadata.coordinateTransformations = [ScaleV04(scale=list(values))]

    with pytest.raises(
        ValueError, match="does not span its axes|values for the 3 axes"
    ):
        to_ome_zarr(
            tmp_path / "refused.ome.zarr",
            dataclasses.replace(multiscales, metadata=metadata),
            version="0.4",
        )


def test_a_dataset_level_transform_is_checked_too(tmp_path):
    # The mainstream path: every store carries dataset-level transforms, and
    # they went unchecked at the same versions.
    import dataclasses

    from ngff_zarr.v04.zarr_metadata import Scale as ScaleV04

    array = np.random.random((4, 8, 8)).astype("float32")
    multiscales = to_multiscales(array, [])
    metadata = multiscales.metadata.to_version("0.4")
    metadata.datasets[0].coordinateTransformations = [ScaleV04(scale=[2.0, 2.0])]

    with pytest.raises(ValueError, match="dataset"):
        to_ome_zarr(
            tmp_path / "refused.ome.zarr",
            dataclasses.replace(multiscales, metadata=metadata),
            version="0.4",
        )


@requires_zarr_v3
def test_a_scale_nested_in_a_sequence_is_checked(tmp_path):
    # The mainstream 0.6 representation: a dataset's scale and translation sit
    # inside one TransformSequence, whose own attributes hold no vector. A gate
    # that reads only the outer transform checks nothing at the version where
    # every store is written this way.
    import dataclasses

    array = np.random.random((4, 8, 8)).astype("float32")
    multiscales = to_multiscales(array, [])
    metadata = multiscales.metadata.to_version("0.6")
    metadata.datasets[0].coordinateTransformations[0].transformations[0].scale = [
        2.0,
        2.0,
    ]

    with pytest.raises(ValueError, match="transformations\\[0\\]"):
        to_ome_zarr(
            tmp_path / "refused.ome.zarr",
            dataclasses.replace(multiscales, metadata=metadata),
            version="0.6",
        )


@requires_zarr_v3
def test_a_transform_spans_the_systems_it_names(tmp_path):
    # From 0.6 a transform maps between the coordinate systems it names, whose
    # arity need not be the intrinsic one: here a spatial system beside a
    # channel axis, which is what a field transform declares. Measured against
    # the intrinsic axes the scale is short by one and the store is refused.
    import dataclasses

    from ngff_zarr.v06.zarr_metadata import (
        Axis,
        CoordinateSystem,
        CoordinateSystemIdentifier,
        Scale,
    )

    image = to_ngff_image(
        np.zeros((2, 4, 5, 6), dtype=np.float32),
        dims=["c", "z", "y", "x"],
        scale={"c": 1.0, "z": 2.0, "y": 1.5, "x": 1.0},
        translation={"c": 0.0, "z": 5.0, "y": -3.0, "x": 7.0},
    )
    multiscales = to_multiscales(image, scale_factors=[], cache=False)
    metadata = multiscales.metadata.to_version("0.6")
    spatial = [Axis(name=name, type="space", unit=None) for name in ("z", "y", "x")]
    metadata = dataclasses.replace(
        metadata,
        coordinateSystems=[
            *metadata.coordinateSystems,
            CoordinateSystem(name="phys", axes=spatial),
        ],
        coordinateTransformations=[
            Scale(
                input=CoordinateSystemIdentifier(name="phys"),
                output=CoordinateSystemIdentifier(name="phys"),
                scale=[2.0, 2.0, 2.0],
            )
        ],
    )

    store = tmp_path / "named.ome.zarr"
    to_ome_zarr(
        store, dataclasses.replace(multiscales, metadata=metadata), version="0.6"
    )

    back = from_ome_zarr(store)
    entry = back.metadata.coordinateTransformations[0]
    assert entry.scale == [2.0, 2.0, 2.0]
    assert (entry.input.name, entry.output.name) == ("phys", "phys")
