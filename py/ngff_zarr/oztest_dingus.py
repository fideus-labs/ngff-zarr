# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The oztest dingus (https://github.com/German-BioImaging/oztest).

``ngff-zarr-oztest <path>`` reads the OME-Zarr hierarchy at ``path`` with
validation on and prints ``{"validity": "valid" | "invalid", ...}``. A crash
exits with a nonzero status, which oztest reports as an error.
"""

import json
import sys
import warnings
from pathlib import Path


def validate_zarr(path) -> dict:
    """The oztest answer for the hierarchy at ``path``."""
    from jsonschema import ValidationError as SchemaError

    from . import from_hcs_zarr, from_ome_zarr
    from .structural_validation import ValidationError as StructuralError

    store = Path(path)
    if (store / "zarr.json").exists():
        attributes = json.loads((store / "zarr.json").read_text()).get("attributes", {})
    else:
        attributes = json.loads((store / ".zattrs").read_text())
    ome = attributes.get("ome", attributes)
    if "well" in ome:
        message = "ngff-zarr reads a well through its plate"
        return {"validity": "invalid", "xfail": True, "message": message}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if "plate" in ome:
                plate = from_hcs_zarr(store, validate=True)
                for well in plate.wells:
                    found = plate.get_well_by_indices(well.rowIndex, well.columnIndex)
                    for index in range(len(found.images) if found else 0):
                        found.get_image(index)
            else:
                kind = "scene" if "scene" in ome else "multiscales"
                from_ome_zarr(store, validate=True, kind=kind)
    except (ValueError, SchemaError, StructuralError) as error:
        return {"validity": "invalid", "message": str(error)}
    return {"validity": "valid"}


def main() -> None:
    print(json.dumps(validate_zarr(sys.argv[-1])))
