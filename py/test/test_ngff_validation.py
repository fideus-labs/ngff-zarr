# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
import pytest
import zarr
from ngff_zarr import (
    NgffMultiscales,
    from_ngff_zarr,
    to_multiscales,
    to_ngff_zarr,
    validate,
)
from packaging import version

zarr_version = version.parse(zarr.__version__)
zarr_version_major = zarr_version.major

# OME-Zarr v0.5 requires a Zarr v3 store, which zarr-python only writes at
# >= 3.0.0b1. The CI matrix exercises legs pinned to zarr 2.x (v0.4 only), where
# ``to_ngff_zarr(..., version="0.5")`` raises; skip the v0.5-writing tests there.
requires_zarr_v3 = pytest.mark.skipif(
    zarr_version < version.parse("3.0.0b1"),
    reason="OME-Zarr v0.5 requires zarr-python >= 3.0.0b1",
)


def check_valid_ngff(multiscale: NgffMultiscales):
    store = zarr.storage.MemoryStore()
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


def test_y_x_valid_ngff():
    array = np.random.random((32, 16))
    multiscale = to_multiscales(array, [2, 4])

    check_valid_ngff(multiscale)


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


def _write_valid_2d_store_v05() -> zarr.storage.MemoryStore:
    """Write a valid 2D ``(y, x)`` two-level v0.5 multiscales to a store."""
    store = zarr.storage.MemoryStore()
    array = np.random.random((32, 16)).astype("float32")
    to_ngff_zarr(store, to_multiscales(array, [2]), version="0.5")
    return store


@requires_zarr_v3
def test_validate_v05_wrapped_instance():
    # The public ``validate`` entry point checks a v0.5 store against the v0.5
    # image schema, which requires the ``ome`` namespace wrapper. The full root
    # attributes (``{"ome": {...}}``) validate; the unwrapped ``ome`` content
    # does not (it lacks the required top-level ``ome`` wrapper), proving the
    # vendored v0.5 schema -- not the v0.4 schema -- is what runs.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_2d_store_v05()
    root_attrs = zarr.open_group(store, mode="r").attrs.asdict()

    validate(root_attrs, version="0.5", model="image")
    with pytest.raises(jsonschema.ValidationError):
        validate(root_attrs["ome"], version="0.5", model="image")


@requires_zarr_v3
def test_validate_v05_schema_active_on_read_path():
    # The read path validates the ``ome``-wrapped instance against the v0.5
    # image schema. A duplicate multiscale entry violates the schema's
    # ``uniqueItems`` constraint -- a pure schema concern the structural rules
    # (which inspect only ``multiscales[0]``) do not check -- so it is rejected
    # under ``validate=True`` and read silently under ``validate=False``.
    jsonschema = pytest.importorskip("jsonschema")

    store = _write_valid_2d_store_v05()
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
def test_read_path_schema_instance_by_version():
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
    v05_instance, v05_version = _schema_call(_write_valid_2d_store_v05())
    assert "ome" in v05_instance
    assert "multiscales" in v05_instance["ome"]
    assert v05_version == "0.5"

    # v0.4: bare root with top-level ``multiscales``, tagged "0.4".
    store_v04 = zarr.storage.MemoryStore()
    array = np.random.random((32, 16)).astype("float32")
    to_ngff_zarr(store_v04, to_multiscales(array, [2]), version="0.4")
    v04_instance, v04_version = _schema_call(store_v04)
    assert "multiscales" in v04_instance
    assert "ome" not in v04_instance
    assert v04_version == "0.4"
