# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT

"""The RFC-5 metadata models are public under a version-scoped path.

There is no unambiguous top-level export: v06 and v09 each define their own
CoordinateSystem with different fields, so a bare ``ngff_zarr.CoordinateSystem``
would silently pick a spec version. Users import from the version they target.

Only released versions are re-exported from the package. The 0.9 model is a
development one that changes between releases, so it is reachable by its own
import path and nothing more.
"""

import ngff_zarr
from ngff_zarr.v06 import zarr_metadata as v06
from ngff_zarr.v09 import zarr_metadata as v09


def test_released_version_subpackages_are_reachable_from_the_package():
    from ngff_zarr.v04 import zarr_metadata as v04

    assert {"v04", "v06"} <= set(ngff_zarr.__all__)
    assert ngff_zarr.v04.zarr_metadata.Axis is v04.Axis
    assert ngff_zarr.v06.zarr_metadata.CoordinateSystem is v06.CoordinateSystem


def test_the_development_model_is_not_exported():
    assert "v09" not in ngff_zarr.__all__
    assert v09.CoordinateSystem is not None  # reachable by its own import path


def test_required_version_specific_names_stay_public():
    # A required export removed from a module must fail here, not only its own __all__ check.
    assert "CoordinateSystemIdentifier" in v06.__all__
    for name in ("Coordinates", "Displacements", "Dataset", "TransformSequence"):
        assert name in v09.__all__, f"v09 dropped {name} from its public surface"


def test_each_module_declares_its_public_surface():
    for name in v06.__all__:
        assert hasattr(v06, name), f"v06.__all__ names {name}, which is absent"
    for name in v09.__all__:
        assert hasattr(v09, name), f"v09.__all__ names {name}, which is absent"
    assert "CoordinateSystem" in v06.__all__
    assert "CoordinateSystem" in v09.__all__


def test_coordinate_system_is_two_distinct_models():
    # The reason there is no top-level export: same name, different class per spec version.
    assert v06.CoordinateSystem is not v09.CoordinateSystem
    assert v06.CoordinateSystem.__module__.endswith("v06.zarr_metadata")
    assert v09.CoordinateSystem.__module__.endswith("v09.zarr_metadata")


def test_field_transforms_are_shared_from_v06():
    # v09 reuses v06's field transforms rather than redefining them.
    assert v09.Displacements is v06.Displacements
    assert v09.Coordinates is v06.Coordinates
