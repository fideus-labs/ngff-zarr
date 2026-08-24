# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Reader-integration tests for strict structural validation.

These exercise the image read path
(:meth:`~ngff_zarr.v04.zarr_metadata.Metadata._from_zarr_attrs` via
:func:`~ngff_zarr.from_ngff_zarr`) rather than the rule functions in isolation:

* a valid array round-tripped through :func:`~ngff_zarr.to_ngff_zarr` and read
  back with ``validate=True`` passes both the schema and structural layers;
* an on-disk store whose ``.zattrs`` violates one structural rule (a ``scale``
  vector whose length disagrees with the axis count -- a v0.4 MUST that the JSON
  Schema cannot express) raises
  :class:`~ngff_zarr.structural_validation.ValidationError` under
  ``validate=True`` but reads silently under the default ``validate=False``.

``validate=True`` runs the JSON-Schema pass first, so these tests require the
optional ``jsonschema`` dependency.
"""

import numpy as np
import pytest
import zarr
from ngff_zarr import from_ngff_zarr, to_multiscales, to_ngff_zarr
from ngff_zarr.structural_validation import SpecRule, ValidationError
from packaging import version

pytest.importorskip("jsonschema", reason="jsonschema required for validate=True")

# OME-Zarr v0.5 requires a Zarr v3 store, which zarr-python only writes at
# >= 3.0.0b1. The CI matrix exercises legs pinned to zarr 2.x (v0.4 only), where
# ``to_ngff_zarr(..., version="0.5")`` raises; skip the v0.5-writing tests there.
requires_zarr_v3 = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="OME-Zarr v0.5 requires zarr-python >= 3.0.0b1",
)


def _write_valid_2d_store(dest) -> str:
    """Write a valid 2D ``(y, x)`` two-level multiscales to directory *dest*."""
    store = str(dest)
    array = np.random.random((32, 16)).astype("float32")
    multiscales = to_multiscales(array, [2])
    to_ngff_zarr(store, multiscales, version="0.4")
    return store


def test_valid_roundtrip_passes_strict_validation(tmp_path):
    store = _write_valid_2d_store(tmp_path / "valid.zarr")
    multiscales = from_ngff_zarr(store, validate=True)
    assert multiscales is not None


def _write_valid_2d_store_v05(dest) -> str:
    """Write a valid 2D ``(y, x)`` two-level v0.5 multiscales to *dest*."""
    store = str(dest)
    array = np.random.random((32, 16)).astype("float32")
    multiscales = to_multiscales(array, [2])
    to_ngff_zarr(store, multiscales, version="0.5")
    return store


@requires_zarr_v3
def test_valid_v05_roundtrip_passes_strict_validation(tmp_path):
    # A clean v0.5 store passes strict validation: the reader hoists
    # ``ome.version`` so the structural pass sees v0.5, and the new namespacing
    # rules accept it (no leaked group-level key, no contradictory zarr_format).
    store = _write_valid_2d_store_v05(tmp_path / "valid_v05.zarr")
    multiscales = from_ngff_zarr(store, validate=True)
    assert multiscales is not None


@requires_zarr_v3
def test_v05_ome_namespace_violation_raises_only_when_validating(tmp_path):
    store = _write_valid_2d_store_v05(tmp_path / "valid_v05.zarr")
    # Leak the group-level ``ome`` wrapper key into the multiscale entry, where
    # it must never appear -- a double-namespaced / malformed v0.5 document.
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    ome["multiscales"][0]["ome"] = {"version": "0.5"}
    root.attrs["ome"] = ome

    # validate=True surfaces the structural violation as a ValidationError.
    with pytest.raises(ValidationError) as exc_info:
        from_ngff_zarr(store, validate=True)
    assert exc_info.value.rule == SpecRule.OME_NAMESPACE

    # The same store reads cleanly when validation is disabled.
    multiscales = from_ngff_zarr(store, validate=False)
    assert multiscales is not None


@pytest.mark.parametrize(
    "version", ["0.4", pytest.param("0.5", marks=requires_zarr_v3)]
)
def test_extra_passthrough_is_never_serialized(version, tmp_path):
    # ``Metadata.extra`` is a read-side validation aid (it captures leaked
    # group-level keys for the v0.5 namespacing rules). It must never reach the
    # written store: the write path serializes the metadata dataclass via
    # ``asdict``, which would otherwise emit an ``extra: {}`` key on every entry.
    # ``_pop_metadata_optionals`` drops it; this guards that drop in both layouts.
    store = str(tmp_path / "store.zarr")
    array = np.random.random((32, 16)).astype("float32")
    multiscales = to_multiscales(array, [2])
    to_ngff_zarr(store, multiscales, version=version)

    attrs = zarr.open_group(store, mode="r").attrs.asdict()
    entry = (
        attrs["ome"]["multiscales"][0] if version == "0.5" else attrs["multiscales"][0]
    )
    assert "extra" not in entry, (
        f"v{version}: 'extra' leaked into the written multiscale entry: {sorted(entry)}"
    )


def test_clean_roundtrip_read_has_empty_extra(tmp_path):
    # ``Metadata.extra`` captures only leaked / mis-namespaced keys. A clean
    # library-authored store must therefore read back with an empty ``extra`` --
    # in particular the serializer-written JSON-LD ``@type`` (``"ngff:Image"``)
    # is a recognized entry field, not a leak, so it must not appear in ``extra``.
    # (v0.4 is asserted directly: its returned Metadata carries ``extra``; the
    # v0.5 conversion drops ``extra`` from its returned object, so this read-side
    # contract is the v0.4 parser's to keep.)
    store = _write_valid_2d_store(tmp_path / "valid.zarr")
    multiscales = from_ngff_zarr(store, validate=True)
    assert multiscales.metadata.extra == {}


@requires_zarr_v3
def test_v05_namespacing_validated_when_ome_version_null(tmp_path):
    # A malformed v0.5 store with ``ome.version: null`` (JSON null -> Python
    # None) must still be validated with the v0.5 namespacing rules. The v0.5
    # read path floors a missing *or* null version to "0.5" before the structural
    # pass, so a leaked group-level wrapper key is still caught rather than
    # silently treated as v0.4 (where the rule is inert).
    store = _write_valid_2d_store_v05(tmp_path / "valid_v05.zarr")
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    ome["version"] = None  # malformed: explicit null version
    ome["multiscales"][0]["ome"] = {"version": "0.5"}  # leaked wrapper key
    root.attrs["ome"] = ome

    with pytest.raises(ValidationError) as exc_info:
        from_ngff_zarr(store, validate=True)
    assert exc_info.value.rule == SpecRule.OME_NAMESPACE


@requires_zarr_v3
def test_v05_zarr_format_violation_raises_only_when_validating(tmp_path):
    store = _write_valid_2d_store_v05(tmp_path / "valid_v05.zarr")
    # Mis-embed a non-3 ``zarr_format`` byte inside the multiscale entry; it
    # belongs on the enclosing Zarr v3 group, never on the entry, and a value
    # other than 3 contradicts the v0.5 => Zarr v3 requirement.
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    ome["multiscales"][0]["zarr_format"] = 2
    root.attrs["ome"] = ome

    with pytest.raises(ValidationError) as exc_info:
        from_ngff_zarr(store, validate=True)
    assert exc_info.value.rule == SpecRule.ZARR_FORMAT

    multiscales = from_ngff_zarr(store, validate=False)
    assert multiscales is not None


@requires_zarr_v3
def test_v05_namespacing_validated_when_ome_version_missing(tmp_path):
    # A malformed v0.5 store missing ``ome.version`` must still be validated with
    # the v0.5 namespacing rules. The store is routed to the v0.5 read path by
    # its ``ome`` key, and that path floors the spec version to "0.5" before the
    # structural pass, so a leaked group-level wrapper key is still caught --
    # rather than silently treated as v0.4 (where the rule is inert).
    store = _write_valid_2d_store_v05(tmp_path / "valid_v05.zarr")
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    del ome["version"]  # malformed: missing ome.version
    ome["multiscales"][0]["ome"] = {"version": "0.5"}  # leaked wrapper key
    root.attrs["ome"] = ome

    with pytest.raises(ValidationError) as exc_info:
        from_ngff_zarr(store, validate=True)
    assert exc_info.value.rule == SpecRule.OME_NAMESPACE


def test_scale_length_violation_raises_only_when_validating(tmp_path):
    store = _write_valid_2d_store(tmp_path / "valid.zarr")
    # Corrupt one dataset's scale to length 3 against the 2-axis (y, x) image.
    # This stays schema-valid (scale has >= 2 entries) but violates the
    # structural scale-length MUST.
    root = zarr.open_group(store, mode="r+")
    attrs = root.attrs.asdict()
    dataset0 = attrs["multiscales"][0]["datasets"][0]
    for transform in dataset0["coordinateTransformations"]:
        if transform.get("type") == "scale":
            transform["scale"] = [1.0, 1.0, 1.0]
    root.attrs["multiscales"] = attrs["multiscales"]

    # validate=True surfaces the structural violation as a ValidationError.
    with pytest.raises(ValidationError) as exc_info:
        from_ngff_zarr(store, validate=True)
    assert exc_info.value.rule == SpecRule.SCALE_LENGTH_MISMATCH

    # The same corrupted store reads cleanly when validation is disabled.
    multiscales = from_ngff_zarr(store, validate=False)
    assert multiscales is not None
