# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import json
import warnings
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from importlib_resources import files as file_resources
from packaging import version as packaging_version

from ._supported_versions import (
    V06_ONDISK_VERSION,
    V06_SUPERSEDED_TAGS,
    NgffVersion,
    is_v06_version,
)

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
    schemas carry the released ``0.6`` tag, and a pre-release string such as
    ``0.6rc0`` or ``0.6.dev4`` that a 0.6 store records on disk still resolves
    to ``spec/0.6`` through ``packaging``'s ``base_version``.

    The version is matched against the bundled directory names rather than
    joined onto the path as given, because it reaches here straight from a
    store's own metadata.
    """
    available = _bundled_versions()
    name = str(version)
    if name not in available:
        try:
            parsed = packaging_version.parse(name)
        except packaging_version.InvalidVersion:
            name = ""
        else:
            # Only a pre-release leads to a release. A post-release or a local
            # version shares a base version with one without being it.
            name = parsed.base_version if parsed.is_prerelease else ""
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
) -> None:
    """Validate OME-Zarr metadata against its bundled JSON Schema.

    Parameters
    ----------
    ngff_dict:
        The parsed group attributes to check.
    version:
        The OME-Zarr version whose bundled schemas to validate against. A
        pre-release resolves to the tree of the release it leads to.
    model:
        The schema to validate against: ``image``, ``label``, ``plate`` or
        ``well``.
    strict:
        Validate against the ``strict_*`` variant of ``model``.

    Raises
    ------
    ImportError
        When the optional ``[validate]`` extra is not installed.
    ValueError
        When no schema tree is bundled for ``version``.
    jsonschema.ValidationError
        When ``ngff_dict`` does not satisfy the schema.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        raise ImportError(
            "jsonschema is required to validate NGFF metadata - install the ngff-zarr[validate] extra"
        )
    schema = load_schema(version=version, model=model, strict=strict)
    validator = Draft202012Validator(schema, registry=_schema_registry(version))
    validator.validate(ngff_dict)


@dataclass(frozen=True)
class Problem:
    """One way a group of an OME-Zarr store breaks the specification."""

    #: The rule broken, or ``schema`` for the official JSON Schema.
    code: str
    message: str
    #: The group, as a path below the store root.
    path: str = ""

    def __str__(self) -> str:
        where = f" at '{self.path}'" if self.path else ""
        return f"[{self.code}]{where} {self.message}"


class ValidationError(ValueError):
    """The metadata of an OME-Zarr store breaks the specification.

    ``problems`` lists every way it does.
    """

    def __init__(self, problems: list[Problem]) -> None:
        self.problems = problems
        super().__init__("\n".join(str(problem) for problem in problems))


#: The JSON Schema each kind of OME-Zarr Metadata is checked against.
_SCHEMA_OF_KEY = {
    "multiscales": "image",
    "image-label": "label",
    "plate": "plate",
    "well": "well",
    "scene": "scene",
    "bioformats2raw.layout": "bf2raw",
}


@cache
def _validator(version: str, model: str):
    from jsonschema import Draft202012Validator

    return Draft202012Validator(
        load_schema(version, model), registry=_schema_registry(version)
    )


def schema_problems(
    attributes: dict, version: str, path: str = "", model: str | None = None
) -> list[Problem]:
    """Check a group's attributes against the JSON Schema of each kind of
    OME-Zarr Metadata they hold, at ``version``.

    ``model`` names the schema the group is read as (``image``, ``plate``,
    ``scene``): it is checked even when the attributes lack its key."""
    ome = attributes.get("ome", attributes)
    if not isinstance(ome, dict):
        return [Problem("schema", "'ome' is not a JSON object", path)]
    if "ome" in attributes and "version" not in ome:
        return [Problem("NGFF-GN-001", "the 'ome' metadata has no 'version'", path)]
    tag = ome.get("version")
    if isinstance(tag, str) and tag in V06_SUPERSEDED_TAGS:
        # Earlier releases wrote this 0.6 pre-release tag, which the bundled
        # schemas no longer list; the rest of the document is checked as 0.6.
        warnings.warn(
            f"OME-Zarr store carries the superseded 0.6 pre-release tag "
            f"{tag!r}; validating it as {V06_ONDISK_VERSION.value!r}. "
            "upgrade_ome_zarr(store, version='0.6') rewrites the tag.",
            stacklevel=3,
        )
        ome = {**ome, "version": V06_ONDISK_VERSION.value}
        attributes = {**attributes, "ome": ome}
        version = V06_ONDISK_VERSION.value
    tree = _schemas_dir(version)
    return [
        Problem("schema", f"{name}.schema, {error.json_path}: {error.message}", path)
        for key, name in _SCHEMA_OF_KEY.items()
        if (key in ome or name == model) and tree.joinpath(f"{name}.schema").is_file()
        for error in _validator(version, name).iter_errors(attributes)
    ]


def group_problems(
    attributes: dict, version: str, path: str = "", model: str | None = None
) -> list[Problem]:
    """Check a group's attributes against the JSON Schema and, once the schema
    accepts them, against the rules beyond it (``ngff_zarr.rules``)."""
    problems = schema_problems(attributes, version, path, model)
    if problems:
        return problems
    from .rules import rule_problems

    if is_v06_version(version):
        version = NgffVersion.V06.value
    return [
        Problem(code, message, path)
        for code, message in rule_problems(attributes, version)
    ]


def _group(node, name: str):
    """The group ``name`` below ``node``, or None."""
    # A name from the store's own metadata must stay below the group.
    if ".." in name.split("/"):
        return None
    try:
        child = node[name]
    except KeyError:
        return None
    return None if hasattr(child, "shape") else child


def store_problems(root, version: str) -> list[Problem]:
    """Check the group ``root`` and the label images its ``labels`` group lists."""
    problems = group_problems(root.attrs.asdict(), version)
    labels = _group(root, "labels")
    if labels is None:
        return problems
    attributes = labels.attrs.asdict()
    ome = attributes.get("ome", attributes)
    names = ome.get("labels") if isinstance(ome, dict) else None
    for name in names if isinstance(names, list) else []:
        label = _group(labels, name) if isinstance(name, str) else None
        if label is not None:
            # A label image implements the multiscales specification.
            problems += group_problems(
                label.attrs.asdict(), version, f"labels/{name}", model="image"
            )
    return problems
