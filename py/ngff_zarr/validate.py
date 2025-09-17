# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Dict
from pathlib import Path
import json
from packaging import version as packaging_version

from importlib_resources import files as file_resources

NGFF_URI = "https://ngff.openmicroscopy.org"


def load_schema(
    version: str = "0.4", model: str = "image", strict: bool = False
) -> Dict:
    strict_str = ""
    if strict:
        strict_str = "strict_"
    schema = (
        file_resources("ngff_zarr")
        .joinpath(
            Path("spec")
            / Path(version)
            / Path("schemas")
            / f"{strict_str}{model}.schema"
        )
        .read_text()
    )
    return json.loads(schema)


def validate(
    ngff_dict: Dict, version: str = "0.4", model: str = "image", strict: bool = False
):
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
    except ImportError:
        raise ImportError(
            "jsonschema is required to validate NGFF metadata - install the ngff-zarr[validate] extra"
        )
    schema = load_schema(version=version, model=model, strict=strict)
    registry = Registry().with_resource(
        NGFF_URI, resource=Resource.from_contents(schema)
    )
    if packaging_version.parse(version) >= packaging_version.parse("0.5"):
        version_schema = load_schema(version=version, model="_version")
        registry = registry.with_resource(
            NGFF_URI, resource=Resource.from_contents(version_schema)
        )
    validator = Draft202012Validator(schema, registry=registry)
    validator.validate(ngff_dict)
