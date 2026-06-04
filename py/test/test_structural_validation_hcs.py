# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Unit tests for the HCS structural rules in :mod:`ngff_zarr.structural_validation`.

Paired pass/fail tests for the two plate/well structural rules added this phase:

* :func:`~ngff_zarr.structural_validation.validate_plate_well_index_consistency`
  -- a small 2x2 plate (:func:`build_valid_plate`) is mutated in exactly one
  field so a well's recorded ``rowIndex``/``columnIndex`` (or its ``path``)
  disagrees with the row/column named in its ``path``.
* :func:`~ngff_zarr.structural_validation.validate_well_acquisition` -- a
  multi-acquisition plate whose well image omits its ``acquisition`` reference.

Each rule function is exercised directly so the violation is isolated to that
rule, asserting the raised :class:`~ngff_zarr.structural_validation.ValidationError`
carries the expected :class:`~ngff_zarr.structural_validation.SpecRule` and a
populated ``location``. The companion :func:`validate_plate`/:func:`validate_well`
orchestrators are also checked for their ``STRICT``/``SCHEMA_ONLY`` behavior.
"""

import pytest
from ngff_zarr.structural_validation import (
    SpecRule,
    ValidateOptions,
    ValidationError,
    ValidationLevel,
    validate_plate,
    validate_plate_well_index_consistency,
    validate_well,
    validate_well_acquisition,
)
from ngff_zarr.v04.zarr_metadata import (
    Plate,
    PlateAcquisition,
    PlateColumn,
    PlateRow,
    PlateWell,
    Well,
    WellImage,
)


def build_valid_plate() -> Plate:
    """Return a small, fully consistent 2x2 plate.

    Rows ``A`` (index 0) and ``B`` (index 1); columns ``1`` (index 0) and ``2``
    (index 1); two wells whose ``path`` and recorded indices agree. This baseline
    passes the index-consistency rule, so each test mutates exactly one field of a
    fresh copy to trip it.
    """
    return Plate(
        columns=[PlateColumn(name="1"), PlateColumn(name="2")],
        rows=[PlateRow(name="A"), PlateRow(name="B")],
        wells=[
            PlateWell(path="A/1", rowIndex=0, columnIndex=0),
            PlateWell(path="B/2", rowIndex=1, columnIndex=1),
        ],
    )


def build_multi_acquisition_plate() -> Plate:
    """Return :func:`build_valid_plate` extended with two acquisitions.

    With more than one acquisition declared, every well image must reference one.
    """
    plate = build_valid_plate()
    plate.acquisitions = [PlateAcquisition(id=0), PlateAcquisition(id=1)]
    return plate


# ---------------------------------------------------------------------------
# plate-row-index-consistency
# ---------------------------------------------------------------------------


def test_plate_well_index_consistency_valid():
    validate_plate_well_index_consistency(build_valid_plate())


@pytest.mark.parametrize(
    "case",
    [
        "row-index-mismatch",
        "column-index-mismatch",
        "unknown-row",
        "unknown-column",
        "malformed-path",
    ],
)
def test_plate_well_index_consistency_invalid(case):
    plate = build_valid_plate()
    if case == "row-index-mismatch":
        # path "A/1" => row "A" is index 0, but rowIndex claims 1.
        plate.wells[0].rowIndex = 1
    elif case == "column-index-mismatch":
        # path "A/1" => column "1" is index 0, but columnIndex claims 1.
        plate.wells[0].columnIndex = 1
    elif case == "unknown-row":
        # Row "Z" is not declared in plate.rows.
        plate.wells[0] = PlateWell(path="Z/1", rowIndex=0, columnIndex=0)
    elif case == "unknown-column":
        # Column "9" is not declared in plate.columns.
        plate.wells[0] = PlateWell(path="A/9", rowIndex=0, columnIndex=0)
    else:  # malformed-path
        # A path that is not "<row>/<column>".
        plate.wells[0] = PlateWell(path="A-1", rowIndex=0, columnIndex=0)
    with pytest.raises(ValidationError) as exc_info:
        validate_plate_well_index_consistency(plate)
    assert exc_info.value.rule == SpecRule.PLATE_ROW_INDEX_CONSISTENCY
    assert exc_info.value.location == "plate.wells[0]"


# ---------------------------------------------------------------------------
# well-acquisition-missing
# ---------------------------------------------------------------------------


def test_well_acquisition_valid_no_or_single_acquisition():
    # Zero or one acquisition imposes no requirement, so a missing reference is OK.
    well = Well(images=[WellImage(path="0")])
    validate_well_acquisition(build_valid_plate(), well)  # acquisitions is None
    single = build_valid_plate()
    single.acquisitions = [PlateAcquisition(id=0)]
    validate_well_acquisition(single, well)


def test_well_acquisition_valid_multi_with_references():
    plate = build_multi_acquisition_plate()
    well = Well(
        images=[
            WellImage(path="0", acquisition=0),
            WellImage(path="1", acquisition=1),
        ]
    )
    validate_well_acquisition(plate, well)


def test_well_acquisition_invalid_missing_reference():
    plate = build_multi_acquisition_plate()
    # The second image omits its acquisition reference; the first is fine.
    well = Well(
        images=[
            WellImage(path="0", acquisition=0),
            WellImage(path="1"),
        ]
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_well_acquisition(plate, well)
    assert exc_info.value.rule == SpecRule.WELL_ACQUISITION_MISSING
    assert exc_info.value.location == "well.images[1].acquisition"


# ---------------------------------------------------------------------------
# Orchestrators: validate_plate / validate_well
# ---------------------------------------------------------------------------


def test_validate_plate_strict_and_schema_only():
    plate = build_valid_plate()
    plate.wells[0].rowIndex = 1  # trip plate-row-index-consistency
    with pytest.raises(ValidationError) as exc_info:
        validate_plate(plate)  # STRICT by default
    assert exc_info.value.rule == SpecRule.PLATE_ROW_INDEX_CONSISTENCY
    # SCHEMA_ONLY runs no structural rule, so the same plate passes.
    validate_plate(plate, ValidateOptions(level=ValidationLevel.SCHEMA_ONLY))


def test_validate_well_strict_and_schema_only():
    plate = build_multi_acquisition_plate()
    well = Well(images=[WellImage(path="0")])  # missing acquisition reference
    with pytest.raises(ValidationError) as exc_info:
        validate_well(plate, well)  # STRICT by default
    assert exc_info.value.rule == SpecRule.WELL_ACQUISITION_MISSING
    # SCHEMA_ONLY runs no structural rule, so the same well passes.
    validate_well(plate, well, ValidateOptions(level=ValidationLevel.SCHEMA_ONLY))
