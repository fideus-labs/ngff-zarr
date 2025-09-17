#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT

"""
# Multiscale transformation checks.

This is a test to make sure that the transforms are propagated correctly to reduced scales in a multiscale representation.

The coordinates represent the center of a pixel.
The transform will transform a coordinate to "real space".

  r = s*p + t

p, pixel location. t is the translate portion of the transform. s is the scale.

Each scale will have it's own set of pixels, transform, and scale. pi, ti, si respectively.
s0 is the original scale factor, it is often used to describe the size of a voxel.

The downsampling factor is the ratio of original pixels to scaled pixels.

    f = ( D0/D1, H0/H1, W0/W1 )

The volume will cover the same real volume, so the ratio of the scale factors will be
the downsampling used.

    s1 = f * s0

To find the translation volume begins in the same pixel location for any resolution:

  pi = (-0.5, -0.5, -0.5)

Then we can solve.

 -0.5s0 + t0 = -0.5*s1 + t1

  t1 = t0 + 0.5*(s1 - s0)
  t1 = t0 + 0.5*(f - 1)s0

"""

import ngff_zarr
import numpy


def check_transformation(meta_transfomations, expected):
    # Must contain scale as first transform v04.
    scale = meta_transfomations[0]
    assert scale.type == "scale"
    for a, b in zip(scale.scale, expected[0]):
        assert a == b
    # Optional scale
    if len(expected) == 1:
        return

    translation = meta_transfomations[1]

    assert translation.type == "translation"

    for a, b in zip(translation.translation, expected[1]):
        assert a == b


def test_multiscale_translate():
    # Input
    dims = ["t", "c", "z", "y", "x"]
    scale = {"t": 60, "c": 1, "z": 2, "y": 0.35, "x": 0.35}
    translation = {"t": 0, "c": 0, "z": 10, "y": 20, "x": 30}
    scale_factors = {"x": 2, "y": 2, "z": 1}
    image_shape = (3, 2, 32, 64, 64)

    # Generation
    arr = numpy.zeros(image_shape, dtype="uint8")
    image = ngff_zarr.ngff_image.NgffImage(
        arr, dims, scale=scale, translation=translation
    )
    multiscales = ngff_zarr.to_multiscales(image, scale_factors=[scale_factors])

    # Checking the values of the ngff image.
    original_scale = multiscales.images[0].scale
    original_translation = multiscales.images[0].translation

    next_scale = multiscales.images[1].scale
    next_translation = multiscales.images[1].translation

    for key in scale_factors:
        t1 = (
            original_translation[key]
            + 0.5 * (scale_factors[key] - 1) * original_scale[key]
        )
        s1 = original_scale[key] * scale_factors[key]
        assert next_translation[key] == t1
        assert next_scale[key] == s1

    # Check values found in the metadata
    datasets = multiscales.metadata.datasets
    set0 = datasets[0]
    set1 = datasets[1]

    original = [[scale[k] for k in dims], [translation[k] for k in dims]]

    # Newly generated scale values.
    scale1 = {}
    translation1 = {}

    for k in dims:
        if k in scale_factors:
            scale1[k] = scale[k] * scale_factors[k]
            translation1[k] = translation[k] + 0.5 * (scale_factors[k] - 1) * scale[k]
        else:
            scale1[k] = scale[k]
            translation1[k] = translation[k]

    scaled = [[scale1[k] for k in dims], [translation1[k] for k in dims]]

    check_transformation(set0.coordinateTransformations, original)
    check_transformation(set1.coordinateTransformations, scaled)
