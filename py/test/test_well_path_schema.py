# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
"""Test well image path validation rules from ome/ngff-spec#71.

Validates the relaxed path constraints:
- Paths may contain alphanumeric characters plus `-`, `_`, `.`
- Paths must not consist only of periods (`.`, `..`)
- Paths must not start with `__`
- Paths must not be empty
- Paths must not contain `/`
"""

import re

import pytest

# The well image path pattern from ome/ngff-spec#71 that will appear
# in a future spec version. We test the rules here directly so they
# are ready when the schema is updated.
WELL_IMAGE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
PERIODS_ONLY_PATTERN = re.compile(r"^\.+$")
DOUBLE_UNDERSCORE_PREFIX = "__"


def is_valid_well_image_path(path: str) -> bool:
    """Check whether a well image path satisfies the relaxed spec rules."""
    if not path:
        return False
    if not WELL_IMAGE_PATH_PATTERN.match(path):
        return False
    if PERIODS_ONLY_PATTERN.match(path):
        return False
    if path.startswith(DOUBLE_UNDERSCORE_PREFIX):
        return False
    return True


@pytest.mark.parametrize(
    "path",
    [
        "0",
        "1",
        "abc",
        "0_a-b.image",
        "my-image",
        "img_01",
        "test.v2",
        "a-b-c",
        "x_y_z",
        "image.1",
    ],
)
def test_valid_well_image_paths(path):
    """Well image paths with allowed characters should be accepted."""
    assert is_valid_well_image_path(path), f'Path "{path}" should be valid'


@pytest.mark.parametrize(
    "path",
    [
        ".",
        "..",
        "...",
    ],
)
def test_reject_periods_only_paths(path):
    """Paths consisting only of periods must be rejected."""
    assert not is_valid_well_image_path(path), f'Path "{path}" should be rejected'


def test_reject_double_underscore_prefix():
    """Paths starting with __ must be rejected."""
    assert not is_valid_well_image_path("__something")


@pytest.mark.parametrize(
    "path",
    [
        "!@#$%^&*()+={}[]",
        "path with spaces",
        "path/with/slashes",
    ],
)
def test_reject_forbidden_characters(path):
    """Paths with characters outside [A-Za-z0-9_.-] must be rejected."""
    assert not is_valid_well_image_path(path), f'Path "{path}" should be rejected'


def test_reject_empty_path():
    """Empty paths must be rejected."""
    assert not is_valid_well_image_path("")
