# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Declare a field image as an RFC-5 ``displacements`` or ``coordinates`` transform.

A field store that only types its component axis is labelled, not applicable:
a reader sees what the channels are, but no transformation says between which
coordinate systems the field maps, so nothing can apply it. The declaration is
the ``displacements`` (or ``coordinates``) entry on the multiscales'
``coordinateTransformations``; building it by hand means reaching into the
version metadata model and validating axis order, component count and system
references oneself. This function is that composition, validated, in one call.

Two layouts use it. A standalone field store -- the artifact a registration
emits -- declares the transform on its own multiscales, mapping a spatial
coordinate system onto itself through its own level-0 array; the default
arguments produce exactly that. A field written beside the image it displaces
declares the transform on the image's multiscales instead, with ``path``
naming the field's array and ``input_system``/``output_system`` referencing
the image's own coordinate systems.

Declared and written at 0.6, the store is applicable by this package's own
reader: :func:`~ngff_zarr.ngff_displacement_field_to_itk_transform` rebuilds an
ITK transform from it with no other information.
"""

from dataclasses import replace
from typing import Literal

from .multiscales import NgffMultiscales
from .ngff_image import _DEFAULT_AXIS_TYPES
from .v06.zarr_metadata import (
    Axis,
    Coordinates,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Displacements,
    validate_transform,
)

_COMPONENT_TYPES = {"displacements": "displacement", "coordinates": "coordinate"}


def declare_field_transform(
    multiscales: NgffMultiscales,
    *,
    transform_type: Literal["displacements", "coordinates"] = "displacements",
    path: str | None = None,
    input_system: str | None = None,
    output_system: str | None = None,
    coordinate_system: str = "physical",
    interpolation: str = "linear",
) -> NgffMultiscales:
    """Return ``multiscales`` with the field transform declared on it.

    Functional and additive: the argument is not mutated, and the entry is
    appended to any ``coordinateTransformations`` already declared. The result
    must be written at OME-Zarr 0.6 or later; earlier versions cannot carry
    multiscale-level transformations and :func:`~ngff_zarr.to_ome_zarr`
    refuses them.

    :param multiscales: The multiscales to declare the transform on: the field
        itself (standalone, the default), or the image the field displaces
        (with ``path`` naming the field's array).
    :type  multiscales: NgffMultiscales
    :param transform_type: ``displacements`` (the field holds offsets) or
        ``coordinates`` (the field holds absolute output positions).
    :type  transform_type: str, optional
    :param path: The zarr path of the field's array. Default: the multiscales'
        own finest dataset -- the standalone store. When given, the field is
        elsewhere and its shape cannot be checked here; the references still
        are.
    :type  path: str, optional
    :param input_system: Name of the declared coordinate system the transform
        maps from. Pass both ``input_system`` and ``output_system``, or
        neither: with neither, a spatial system named ``coordinate_system`` is
        declared from the field's own axes and the transform maps it onto
        itself, which is the standalone total field.
    :type  input_system: str, optional
    :param output_system: Name of the declared coordinate system the transform
        maps onto. See ``input_system``.
    :type  output_system: str, optional
    :param coordinate_system: Name of the spatial coordinate system to declare
        when ``input_system``/``output_system`` are not given.
    :type  coordinate_system: str, optional
    :param interpolation: How a reader interpolates the field between grid
        points.
    :type  interpolation: str, optional
    :return: A new multiscales carrying the declaration.
    :rtype: NgffMultiscales
    :raises ValueError: If the field's axes are not one component axis of the
        type the transform calls for followed by the spatial axes, if it holds
        a number of components other than the number of spatial axes, if a
        named system is not declared, or if the entry does not validate
        against the systems it references.
    """
    if transform_type not in _COMPONENT_TYPES:
        msg = (
            f"transform_type must be one of {sorted(_COMPONENT_TYPES)}, "
            f"got '{transform_type}'"
        )
        raise ValueError(msg)
    metadata = multiscales.metadata
    if (input_system is None) != (output_system is None):
        msg = "pass both input_system and output_system, or neither"
        raise ValueError(msg)
    if path is None:
        spatial = _check_field_image(multiscales, _COMPONENT_TYPES[transform_type])
        path = metadata.datasets[0].path
    elif input_system is None:
        msg = (
            "with path, pass input_system and output_system: the field is "
            "elsewhere, so there are no axes here to derive a system from"
        )
        raise ValueError(msg)

    systems = list(getattr(metadata, "coordinateSystems", None) or [])
    declared = {system.name: len(system.axes) for system in systems}
    if input_system is None:
        units = multiscales.images[0].axes_units or {}
        systems = _with_spatial_system(systems, coordinate_system, spatial, units)
        input_system = output_system = coordinate_system
    else:
        for name in (input_system, output_system):
            if name not in declared:
                msg = (
                    f"'{name}' names no declared coordinate system "
                    f"(declared: {sorted(declared)})"
                )
                raise ValueError(msg)
        if declared[input_system] != declared[output_system]:
            msg = (
                f"'{input_system}' names {declared[input_system]} axes and "
                f"'{output_system}' names {declared[output_system]}; a field "
                "transform maps a point onto one of the same dimension, which "
                "is what the field holds per point"
            )
            raise ValueError(msg)

    entry_class = Displacements if transform_type == "displacements" else Coordinates
    entry = entry_class(
        input=CoordinateSystemIdentifier(name=input_system),
        output=CoordinateSystemIdentifier(name=output_system),
        path=path,
        interpolation=interpolation,
    )
    validate_transform(entry, systems)
    metadata = replace(
        metadata,
        coordinateSystems=systems,
        coordinateTransformations=[
            *(metadata.coordinateTransformations or []),
            entry,
        ],
    )
    return replace(multiscales, metadata=metadata)


def _check_field_image(multiscales: NgffMultiscales, component_type: str) -> list:
    """The standalone checks -- the multiscales is the field, so its shape is
    here to check: one component axis of the right type first, then the
    spatial axes, one component per spatial axis. Returns the spatial dims."""
    image = multiscales.images[0]
    component_dims = [
        dim
        for dim, axis_type in (image.axes_types or {}).items()
        if axis_type == component_type
    ]
    if len(component_dims) != 1:
        msg = (
            f"the field image must have exactly one axis of type "
            f"'{component_type}' (axes_types on the NgffImage); got "
            f"{component_dims or 'none'} on dims {tuple(image.dims)}"
        )
        raise ValueError(msg)
    types = image.axes_types or {}
    other = [
        dim
        for dim in image.dims
        if dim != component_dims[0]
        and types.get(dim, _DEFAULT_AXIS_TYPES.get(dim, "space")) != "space"
    ]
    if other:
        msg = (
            f"a field transform is defined on spatial axes only; dims "
            f"{tuple(image.dims)} name {other}, which a coordinate system "
            "cannot carry as space"
        )
        raise ValueError(msg)
    if image.dims[0] != component_dims[0]:
        msg = (
            f"the field's dims are {tuple(image.dims)}; the component axis "
            f"'{component_dims[0]}' must come first, then the spatial axes in "
            "order"
        )
        raise ValueError(msg)
    spatial = [dim for dim in image.dims if dim != component_dims[0]]
    if image.data.shape[0] != len(spatial):
        msg = (
            f"the field holds {image.data.shape[0]} components per point over "
            f"spatial axes {tuple(spatial)}, which name {len(spatial)} axes"
        )
        raise ValueError(msg)
    return spatial


def _with_spatial_system(systems: list, name: str, spatial: list, units) -> list:
    """The systems with a spatial one named ``name``, declared from the field's
    own axes if absent; an existing one must already carry those axes in order,
    or the declaration would reference a system the field does not match."""
    existing = next((system for system in systems if system.name == name), None)
    if existing is not None:
        if [axis.name for axis in existing.axes] != spatial:
            msg = (
                f"coordinate system '{name}' is already declared with axes "
                f"{[axis.name for axis in existing.axes]}, but the field's "
                f"spatial axes are {spatial}"
            )
            raise ValueError(msg)
        other = [
            f"{axis.name} ({axis.type})"
            for axis in existing.axes
            if axis.type != "space"
        ]
        if other:
            msg = (
                f"coordinate system '{name}' carries {other} where a field "
                "transform needs space; it is defined on spatial axes only"
            )
            raise ValueError(msg)
        return systems
    axes = [Axis(name=dim, type="space", unit=units.get(dim)) for dim in spatial]
    return [*systems, CoordinateSystem(name=name, axes=axes)]
