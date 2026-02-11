#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Export each scale of a local OME-Zarr (.zarr or .ozx) as a NIfTI file.

Usage
-----
    pixi run --as-is -e test python py/test/ome_zarr_to_nifti.py INPUT [--output-dir DIR] [--compress]

Examples
--------
    # Write scale_0.nii, scale_1.nii, ... next to the input
    pixi run --as-is -e test python py/test/ome_zarr_to_nifti.py data/brain.zarr

    # Write compressed NIfTI files into a specific directory
    pixi run --as-is -e test python py/test/ome_zarr_to_nifti.py data/brain.ozx --output-dir /tmp/niftis --compress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from ngff_zarr import from_ngff_zarr
from ngff_zarr.ngff_image import NgffImage
from ngff_zarr.rfc4 import AnatomicalOrientationValues


def _build_affine(image: NgffImage) -> np.ndarray:
    """Build a NIfTI-style 4x4 affine from an NgffImage's spatial metadata."""
    affine = np.eye(4)

    for i, dim in enumerate(("x", "y", "z")):
        if dim in image.scale:
            affine[i, i] = image.scale[dim]
        if dim in image.translation:
            affine[i, 3] = image.translation[dim]

    # Flip sign for axes whose orientation is opposite to NIfTI RAS+ convention.
    if image.axes_orientations:
        negative_orientations = {
            "x": AnatomicalOrientationValues.right_to_left,
            "y": AnatomicalOrientationValues.anterior_to_posterior,
            "z": AnatomicalOrientationValues.superior_to_inferior,
        }
        for i, dim in enumerate(("x", "y", "z")):
            ornt = image.axes_orientations.get(dim)
            if ornt is not None:
                ornt_val = ornt.value if hasattr(ornt, "value") else ornt
                if ornt_val == negative_orientations.get(dim):
                    affine[i, i] = -abs(affine[i, i])

    return affine


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export each scale of a local OME-Zarr as a NIfTI file.",
    )
    parser.add_argument(
        "input",
        help="Path to a local .zarr directory or .ozx file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help=(
            "Directory to write NIfTI files into. "
            "Defaults to the parent directory of the input."
        ),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Write compressed .nii.gz files instead of .nii.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    suffix = ".nii.gz" if args.compress else ".nii"

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a base name from the input, stripping .zarr / .ome.zarr / .ozx
    stem = input_path.name
    for ext in (".ome.zarr", ".zarr", ".ozx"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    print(f"Reading multiscales from {input_path} ...")
    multiscales = from_ngff_zarr(str(input_path))

    n_scales = len(multiscales.images)
    print(f"Found {n_scales} scale(s).")

    for idx, image in enumerate(multiscales.images):
        out_name = f"{stem}_scale_{idx}{suffix}"
        out_path = output_dir / out_name

        print(
            f"  [{idx + 1}/{n_scales}] shape={image.data.shape}  "
            f"dims={list(image.dims)}  "
            f"scale={image.scale}"
        )

        data = np.asarray(image.data)
        affine = _build_affine(image)

        nii = nib.Nifti1Image(data, affine)
        nib.save(nii, str(out_path))
        print(f"    -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
