# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Write RFC-5 test stores with their expected answers.

    python rfc5_test_stores.py OUT_DIR

Each store is an OME-Zarr 0.6 scene: a fixed and a moving image, and one
transformation from the fixed to the moving image's coordinate system.
``OUT_DIR/manifest.json`` lists, per store, points of the fixed image and where
the transformation must send them. The answers are exact by construction, not
computed by ngff-zarr: the affine is applied as a matrix, and every field is
linear in position, which linear interpolation reproduces exactly between its
samples. Any reader can be checked against them, for instance with
``ngff-zarr transform-points STORE points.json``.

The geometry is anisotropic (z spacing 2.0, y/x 0.5 micrometer) with a
non-zero origin, so that an axis swap or an index/physical mix-up shows.
"""

import json
import sys
from pathlib import Path

import dask.array as da
import ngff_zarr as nz
import numpy as np
from ngff_zarr import CoordinateSystemIdentifier as CSI
from ngff_zarr.v06.zarr_metadata import Affine, Displacements, TransformSequence

DIMS = ("z", "y", "x")
SHAPE = (9, 17, 21)
SPACING = np.array([2.0, 0.5, 0.5])
ORIGIN = np.array([-4.0, 1.0, 3.0])
UNITS = dict.fromkeys(DIMS, "micrometer")
# The affine (rows z, y, x; the last column the translation) and the linear
# field u(p) = M p + b, both small enough to keep the points on the field.
AFFINE = np.array(
    [[1.0, 0.02, 0.0, 0.5], [0.0, 0.98, 0.03, -0.25], [0.01, 0.0, 1.01, 0.125]]
)
M = np.array([[0.01, 0.0, 0.02], [0.0, -0.015, 0.01], [0.005, 0.01, 0.0]])
B = np.array([0.3, -0.2, 0.1])
FIELD_PATH = "coordinateTransformations/field"


def _image(name):
    data = np.random.default_rng(len(name)).random(SHAPE).astype(np.float32)
    return nz.NgffImage(
        data=da.from_array(data),
        dims=DIMS,
        scale=dict(zip(DIMS, SPACING.tolist())),
        translation=dict(zip(DIMS, ORIGIN.tolist())),
        axes_units=UNITS,
        name=name,
    )


def _linear_field(step):
    """u(p) = M p + b sampled every ``step`` voxels over the image's extent."""
    shape = tuple((n - 1) // step + 1 for n in SHAPE)
    spacing = SPACING * step
    grid = np.stack(
        np.meshgrid(
            *[ORIGIN[i] + spacing[i] * np.arange(shape[i]) for i in range(3)],
            indexing="ij",
        )
    )
    values = np.einsum("ij,j...->i...", M, grid) + B[:, None, None, None]
    dims = ("c", *DIMS)
    return nz.NgffImage(
        data=da.from_array(values.astype(np.float64)),
        dims=dims,
        scale={"c": 1.0, **dict(zip(DIMS, spacing.tolist()))},
        translation={"c": 0.0, **dict(zip(DIMS, ORIGIN.tolist()))},
        axes_units=UNITS,
        axes_types={"c": "displacement"},
        name="field",
    )


def _write(store, transform, field=None):
    fixed = nz.to_multiscales(_image("fixed"), scale_factors=[])
    moving = nz.to_multiscales(_image("moving"), scale_factors=[])
    transform.input = CSI(
        path="fixed", name=fixed.metadata.intrinsic_coordinate_system.name
    )
    transform.output = CSI(
        path="moving", name=moving.metadata.intrinsic_coordinate_system.name
    )
    if field is not None:
        nz.to_ome_zarr(
            f"{store}/{FIELD_PATH}",
            nz.to_multiscales(field, scale_factors=[]),
            version="0.6",
        )
    nz.NgffScene(
        images={"fixed": fixed, "moving": moving}, coordinateTransformations=[transform]
    ).to_ome_zarr(str(store), overwrite=False)


def _u(p):
    return p @ M.T + B


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261028)
    # Points well inside the image, so that the affine keeps them on the field.
    extent = SPACING * (np.array(SHAPE) - 1)
    points = ORIGIN + extent * (0.2 + 0.6 * rng.random((16, 3)))
    affine = lambda p: p @ AFFINE[:, :3].T + AFFINE[:, 3]  # noqa: E731
    cases = []

    _write(out / "affine.ome.zarr", Affine(affine=AFFINE.tolist()))
    cases.append(("affine.ome.zarr", "affine", affine(points)))

    for name, step in (("displacements_dense", 1), ("displacements_coarse", 4)):
        field = _linear_field(step)
        _write(
            out / f"{name}.ome.zarr",
            Displacements(path=FIELD_PATH, interpolation="linear"),
            field,
        )
        cases.append(
            (
                f"{name}.ome.zarr",
                f"displacements, field every {step} voxel(s)",
                points + _u(points),
            )
        )

    field = _linear_field(2)
    sequence = TransformSequence(
        transformations=[
            Affine(affine=AFFINE.tolist()),
            Displacements(path=FIELD_PATH, interpolation="linear"),
        ]
    )
    _write(out / "sequence_affine_displacements.ome.zarr", sequence, field)
    moved = affine(points)
    cases.append(
        (
            "sequence_affine_displacements.ome.zarr",
            "affine, then displacements",
            moved + _u(moved),
        )
    )

    (out / "points.json").write_text(json.dumps(points.tolist()))
    manifest = {
        "dims": list(DIMS),
        "points": "points.json",
        "tolerance": 1e-6,
        "cases": [
            {
                "store": store,
                "transform": 0,
                "description": text,
                "expected": expected.tolist(),
            }
            for store, text, expected in cases
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main(sys.argv[1])
