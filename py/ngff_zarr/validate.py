# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import json
from functools import cache
from typing import TYPE_CHECKING

from importlib_resources import files as file_resources
from packaging import version as packaging_version

if TYPE_CHECKING:
    from importlib_resources.abc import Traversable
    from referencing import Registry

NGFF_URI = "https://ngff.openmicroscopy.org"


@cache
def _bundled_versions() -> frozenset:
    """The versions that have a bundled ``spec/<version>/schemas`` tree."""
    spec = file_resources("ngff_zarr").joinpath("spec")
    return frozenset(
        entry.name
        for entry in spec.iterdir()
        if entry.is_dir() and entry.joinpath("schemas").is_dir()
    )


def _schemas_dir(version: str) -> "Traversable":
    """Locate the bundled ``schemas`` directory that holds ``version``.

    A pre-release shares the tree of the release it leads to: the bundled 0.6
    schemas carry the upstream ``0.6.dev4`` tag, so the ``"0.6.dev4"`` string a
    0.6 store records on disk resolves to ``spec/0.6`` just as ``"0.6"`` does.

    The version is matched against the bundled directory names rather than
    joined onto the path as given, because it reaches here straight from a
    store's own metadata.
    """
    available = _bundled_versions()
    name = str(version)
    if name not in available:
        try:
            name = packaging_version.parse(name).base_version
        except packaging_version.InvalidVersion:
            name = ""
        if name not in available:
            raise ValueError(
                f"No JSON Schema is bundled for OME-Zarr version {version!r}. "
                f"Bundled versions: {', '.join(sorted(available))}."
            )
    return file_resources("ngff_zarr").joinpath("spec").joinpath(name, "schemas")


def load_schema(
    version: str = "0.4", model: str = "image", strict: bool = False
) -> dict:
    strict_str = ""
    if strict:
        strict_str = "strict_"
    schema = _schemas_dir(version).joinpath(f"{strict_str}{model}.schema").read_text()
    return json.loads(schema)


@cache
def _schema_registry(version: str) -> "Registry":
    """Register every bundled schema for ``version`` under its own ``$id``.

    From 0.6 the spec splits axes, coordinate systems and coordinate
    transformations into their own files, which ``image.schema`` reaches by
    absolute ``$id`` URL. Validation is offline, so nothing dereferences those
    URLs: each sibling file has to be in the registry for the references to
    resolve. The pre-0.6 schemas keep the same layout, where the only
    cross-file references are ``strict_*`` wrappers around their base schema.
    """
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    resources = []
    for entry in _schemas_dir(version).iterdir():
        if not entry.name.endswith(".schema"):
            continue
        contents = json.loads(entry.read_text())
        # Some bundled schemas omit ``$schema``; they all predate 2020-12 draft
        # divergences, so the draft the validator runs is the right default.
        resource = Resource.from_contents(contents, default_specification=DRAFT202012)
        resources.append((contents.get("$id", NGFF_URI), resource))
    return Registry().with_resources(resources)


def validate(
    ngff_dict: dict, version: str = "0.4", model: str = "image", strict: bool = False
):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        raise ImportError(
            "jsonschema is required to validate NGFF metadata - install the ngff-zarr[validate] extra"
        )
    schema = load_schema(version=version, model=model, strict=strict)
    validator = Draft202012Validator(schema, registry=_schema_registry(version))
    validator.validate(ngff_dict)
