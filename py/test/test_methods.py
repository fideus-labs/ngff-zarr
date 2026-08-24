# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Contract tests for the ``Methods`` enum and its derived helpers.

These lock in the public surface of :mod:`ngff_zarr.methods`: the exact set of
members and their ``.value`` strings (which are persisted verbatim in OME-Zarr
``multiscales.type`` metadata), the backwards-compatible ``methods`` /
``methods_values`` helpers, value-based lookup (used by the CLI and metadata
parser), and that every member is documented.
"""

import ast
import enum
import inspect
import textwrap

import ngff_zarr
from ngff_zarr.methods import Methods, methods, methods_values

# Name -> value mapping, in definition order. The values are written to disk in
# OME-Zarr metadata, so changing one is a breaking change and must be deliberate.
EXPECTED_MEMBERS = {
    "ITKWASM_GAUSSIAN": "itkwasm_gaussian",
    "ITKWASM_BIN_SHRINK": "itkwasm_bin_shrink",
    "ITKWASM_LABEL_IMAGE": "itkwasm_label_image",
    "ITK_GAUSSIAN": "itk_gaussian",
    "ITK_BIN_SHRINK": "itk_bin_shrink",
    "DASK_IMAGE_GAUSSIAN": "dask_image_gaussian",
    "DASK_IMAGE_MODE": "dask_image_mode",
    "DASK_IMAGE_NEAREST": "dask_image_nearest",
    "DASK_BIN_SHRINK": "dask_bin_shrink",
}


def test_methods_is_enum_exported_at_top_level():
    assert issubclass(Methods, enum.Enum)
    # The top-level re-export must be the very same object.
    assert ngff_zarr.Methods is Methods


def test_member_names_and_values_are_stable():
    """Members and their persisted value strings must not drift unexpectedly."""
    assert {member.name: member.value for member in Methods} == EXPECTED_MEMBERS
    # Order matters for CLI choices and is part of the public contract.
    assert [member.name for member in Methods] == list(EXPECTED_MEMBERS.keys())


def test_methods_helper_matches_enum():
    """``methods`` is the (name, value) pairs derived from the enum."""
    assert methods == [(member.name, member.value) for member in Methods]
    assert methods == list(EXPECTED_MEMBERS.items())


def test_methods_values_helper_matches_enum():
    """``methods_values`` (used for CLI ``--method`` choices) tracks the enum."""
    assert methods_values == [member.value for member in Methods]
    assert methods_values == list(EXPECTED_MEMBERS.values())


def test_value_lookup_roundtrips():
    """``Methods(value)`` resolves to the member, as cli.py / parse_metadata rely on."""
    for member in Methods:
        assert Methods(member.value) is member


def test_class_has_docstring():
    assert Methods.__doc__ is not None
    assert Methods.__doc__.strip()


def test_every_member_has_an_attribute_docstring():
    """Each member must be followed by a docstring (the gap that caused gh-227).

    Attribute docstrings are not available at runtime, so inspect the source AST:
    every ``NAME = "value"`` assignment in the class body must be immediately
    followed by a string-literal expression statement.
    """
    class_source = textwrap.dedent(inspect.getsource(Methods))
    class_def = ast.parse(class_source).body[0]
    assert isinstance(class_def, ast.ClassDef)

    body = class_def.body
    documented = set()
    for index, node in enumerate(body):
        is_member_assign = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        )
        if not is_member_assign:
            continue
        following = body[index + 1] if index + 1 < len(body) else None
        has_docstring = (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        )
        if has_docstring:
            documented.add(node.targets[0].id)

    member_names = {member.name for member in Methods}
    undocumented = member_names - documented
    assert not undocumented, f"Methods members missing a docstring: {undocumented}"
