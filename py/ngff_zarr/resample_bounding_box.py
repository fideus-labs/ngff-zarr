# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Find the region of a moving image needed to resample a fixed image grid."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .ngff_image import NgffImage
from .ngff_transform_to_itk_transform import ngff_transform_to_itk_transform
from .rfc4 import anatomical_orientation_to_itk_direction
from .v06.zarr_metadata import BaseTransform

_SPATIAL_DIMS = ("x", "y", "z")


@dataclass
class ResampleBoundingBox:
    """The region of a moving image needed to resample a fixed image grid.

    Every mapping is keyed by dimension name, so callers never have to track
    whether a given array is in Zarr or ITK axis order.
    """

    #: Spatial dimension names, in the moving image's (Zarr) order.
    dims: tuple[str, ...]
    #: Start of the region in the moving image's index space. May be negative
    #: when the transformed grid extends past the moving image origin.
    start_index: dict[str, int]
    #: Size of the region in moving-image pixels, before clamping.
    size: dict[str, int]
    #: Tight extent of the transformed fixed grid, in moving-image physical
    #: coordinates. Independent of ``padding``.
    corners_min: dict[str, float]
    corners_max: dict[str, float]
    #: Physical coordinates of the padded region's first and last pixel.
    padded_corners_min: dict[str, float]
    padded_corners_max: dict[str, float]
    #: Shape of the moving image, used to clamp the region into bounds.
    moving_shape: dict[str, int] = field(default_factory=dict)

    def clamped(self) -> dict[str, tuple[int, int]]:
        """Return the region intersected with the moving image bounds.

        :return: ``{dim: (start, stop)}`` with ``0 <= start <= stop <= shape``.
        :rtype: dict[str, tuple[int, int]]
        """
        bounds = {}
        for dim in self.dims:
            extent = self.moving_shape[dim]
            start = max(0, min(self.start_index[dim], extent))
            stop = max(start, min(self.start_index[dim] + self.size[dim], extent))
            bounds[dim] = (start, stop)
        return bounds

    @property
    def is_empty(self) -> bool:
        """Whether the region does not overlap the moving image at all."""
        return any(stop <= start for start, stop in self.clamped().values())

    def slices(self, dims: Sequence[str] | None = None) -> tuple[slice, ...]:
        """Slices selecting the region, clamped to the moving image bounds.

        Negative start indices are clamped rather than passed through, since a
        negative bound would silently wrap around instead of raising.

        :param dims: Dimension order of the array to be sliced. Defaults to the
            spatial dims. Dimensions outside the region (``t``, ``c``) get a
            full slice.
        :type  dims: Sequence[str] | None

        :return: One slice per entry in ``dims``.
        :rtype: tuple[slice, ...]
        """
        order = tuple(dims) if dims is not None else self.dims
        bounds = self.clamped()
        return tuple(
            slice(*bounds[dim]) if dim in bounds else slice(None) for dim in order
        )

    def crop(self, moving: NgffImage) -> NgffImage | None:
        """Lazily crop ``moving`` to this region, correcting its translation.

        The returned image indexes the same Dask graph -- nothing is computed --
        so only this region's chunks are read when it is finally materialized.

        :param moving: The moving image the region was computed against.
        :type  moving: NgffImage

        :return: The cropped image, or ``None`` when the region does not
            overlap the moving image.
        :rtype: NgffImage | None
        """
        if self.is_empty:
            return None
        bounds = self.clamped()
        translation = _shifted_translation(
            moving, {dim: start for dim, (start, _stop) in bounds.items()}
        )
        return NgffImage(
            data=moving.data[self.slices(moving.dims)],
            dims=moving.dims,
            scale=dict(moving.scale),
            translation=translation,
            name=moving.name,
            axes_units=moving.axes_units,
            axes_orientations=moving.axes_orientations,
            axes_types=moving.axes_types,
            channel_names=moving.channel_names,
            channel_colors=moving.channel_colors,
        )


def _spatial_dims(ngff_image: NgffImage) -> list[str]:
    return [dim for dim in ngff_image.dims if dim in _SPATIAL_DIMS]


def _check_geometry(label: str, image: NgffImage, spatial: Sequence[str]) -> None:
    """Require a finite scale and translation for every spatial axis.

    A missing or non-finite entry would otherwise reach the pipeline and come
    back as a plausible-looking region computed from the wrong geometry.
    """
    for name, values in (("scale", image.scale), ("translation", image.translation)):
        for dim in spatial:
            if dim not in values:
                msg = f"{label} image {name} has no entry for dimension '{dim}'"
                raise ValueError(msg)
            if not np.isfinite(values[dim]):
                msg = (
                    f"{label} image {name} for dimension '{dim}' is "
                    f"{values[dim]}; it must be finite"
                )
                raise ValueError(msg)
        if name == "scale":
            for dim in spatial:
                if values[dim] == 0:
                    msg = f"{label} image scale for dimension '{dim}' is zero"
                    raise ValueError(msg)


def _check_index_arrays(result: dict, itk_dims: Sequence[str]) -> None:
    """Reject a pipeline result that cannot fill one bound per dimension.

    The returned mappings are built from these two arrays alone, so a short or
    non-finite result would otherwise surface as an opaque failure further
    down, or as an undefined bound a caller slices with.
    """
    for name in ("paddedStartIndex", "paddedSize"):
        values = result[name]
        if len(values) != len(itk_dims):
            msg = (
                f"the pipeline reported {len(values)} {name} values for "
                f"{len(itk_dims)} dimensions"
            )
            raise ValueError(msg)
        for dim, value in zip(itk_dims, values):
            if not math.isfinite(value):
                msg = (
                    f"the {name} for dimension '{dim}' is not finite; check "
                    "the transform and the image scale and translation"
                )
                raise ValueError(msg)


def _check_region_contains_corners(result: dict, itk_dims: Sequence[str]) -> None:
    """Reject a region that does not contain the points it was derived from.

    The pipeline computes the integer region in 32-bit index space while
    reporting the corners as doubles, so a grid whose extent exceeds that range
    comes back as a wrapped -- and typically empty -- region. Padding is
    non-negative, so a correct region always spans its own tight corners;
    checking that catches the wrap without assuming an axis direction.
    """
    corners_min = result["corners"]["min"]
    corners_max = result["corners"]["max"]
    padded_min = result["paddedCorners"]["min"]
    padded_max = result["paddedCorners"]["max"]

    for axis, dim in enumerate(itk_dims):
        values = (
            corners_min[axis],
            corners_max[axis],
            padded_min[axis],
            padded_max[axis],
        )
        if any(value is None or not np.isfinite(value) for value in values):
            msg = (
                f"the resample region for dimension '{dim}' is not finite; "
                "check the transform and the image scale and translation"
            )
            raise ValueError(msg)
        # paddedCorners hold the start and end index positions, which invert
        # when an axis direction is negative, so compare against the span.
        low = min(padded_min[axis], padded_max[axis])
        high = max(padded_min[axis], padded_max[axis])
        tight_low = min(corners_min[axis], corners_max[axis])
        tight_high = max(corners_min[axis], corners_max[axis])
        if tight_low < low or tight_high > high:
            msg = (
                f"the resample region for dimension '{dim}' does not contain "
                f"the transformed grid ({tight_low} to {tight_high} lies "
                f"outside {low} to {high}). The region spans more than the "
                "index range the pipeline can represent; check for a scale "
                "mismatch between the fixed and moving images."
            )
            raise ValueError(msg)


def _shifted_translation(ngff_image: NgffImage, starts: dict) -> dict:
    """Translation of the sub-grid of ``ngff_image`` that starts at ``starts``.

    ITK places index ``i`` at ``origin + direction @ (i * spacing)`` and
    :func:`ngff_image_to_itk_image` uses ``translation`` as the origin, so the
    origin of a sub-grid is the translation moved by the index offset rotated
    through the RFC-4 direction matrix. Dimensions with no orientation, and
    non-spatial dimensions, move along their own axis.
    """
    translation = dict(ngff_image.translation)
    spatial = _spatial_dims(ngff_image)
    itk_dims = [dim for dim in _SPATIAL_DIMS if dim in spatial]
    direction = _itk_direction(ngff_image, itk_dims)
    offset = direction @ np.array(
        [starts.get(dim, 0) * ngff_image.scale[dim] for dim in itk_dims], dtype=float
    )
    for index, dim in enumerate(itk_dims):
        translation[dim] = ngff_image.translation[dim] + float(offset[index])
    for dim in ngff_image.dims:
        if dim not in spatial and dim in starts:
            translation[dim] = (
                ngff_image.translation[dim] + starts[dim] * ngff_image.scale[dim]
            )
    return translation


def _itk_direction(ngff_image: NgffImage, itk_dims: Sequence[str]) -> np.ndarray:
    """Direction matrix from RFC-4 orientation, matching ngff_image_to_itk_image.

    All-or-nothing: unless every spatial axis carries an orientation that maps
    onto an LPS axis, the direction falls back to identity.
    """
    direction = np.eye(len(itk_dims))
    orientations = ngff_image.axes_orientations
    if not orientations:
        return direction

    columns = []
    seen_axes = set()
    for dim in itk_dims:
        orientation = orientations.get(dim)
        if orientation is None:
            return direction
        column = anatomical_orientation_to_itk_direction(orientation.value)
        if column is None:
            return direction
        # A column pointing outside the matrix's dimension (e.g. a
        # superior/inferior orientation on a 2D image), or two dims mapping
        # onto the same LPS axis, would truncate to a singular matrix; keep
        # the identity fallback instead.
        if any(component != 0 for component in column[len(itk_dims) :]):
            return direction
        axis = max(range(len(column)), key=lambda i: abs(column[i]))
        if axis in seen_axes:
            return direction
        seen_axes.add(axis)
        columns.append(column)

    for col_index, column in enumerate(columns):
        for row in range(len(itk_dims)):
            direction[row, col_index] = column[row]
    return direction


def _metadata_only_itk_image(
    ngff_image: NgffImage,
    itk_dims: Sequence[str],
    direction: np.ndarray,
):
    """Build an ITK-Wasm image carrying geometry only, with an empty buffer.

    ``ngff_image.data`` is never converted to an array -- only its ``shape`` and
    ``dtype`` are read, which are Dask metadata. This is what lets the region be
    computed against images whose pixels are remote or enormous.
    """
    from itkwasm import Image, ImageType, IntTypes, PixelTypes

    dims = tuple(ngff_image.dims)
    image_type = ImageType(
        dimension=len(itk_dims),
        componentType=IntTypes.UInt8,
        pixelType=PixelTypes.Scalar,
        components=1,
    )
    return Image(
        imageType=image_type,
        name=ngff_image.name,
        origin=[float(ngff_image.translation[dim]) for dim in itk_dims],
        spacing=[float(ngff_image.scale[dim]) for dim in itk_dims],
        direction=direction,
        size=[int(ngff_image.data.shape[dims.index(dim)]) for dim in itk_dims],
        metadata={},
        data=np.empty((0,), dtype=np.uint8),
    )


#: RFC-5 transformation types, plus the v0.4 spellings of the three that
#: predate it. ``ngff_zarr.Scale`` and friends are the v0.4 dataclasses, which
#: do not share the v0.6 base class.
_NGFF_TRANSFORM_TYPES = frozenset(
    {
        "identity",
        "scale",
        "translation",
        "rotation",
        "affine",
        "mapAxis",
        "byDimension",
        "bijection",
        "sequence",
        "coordinates",
        "displacements",
    }
)


def _is_ngff_transform(transform) -> bool:
    if isinstance(transform, BaseTransform):
        return True
    return getattr(transform, "type", None) in _NGFF_TRANSFORM_TYPES


def _as_itk_transform_list(transform) -> list:
    """Normalize a supported transform input into an ITK-Wasm transform list.

    ITK applies the *last* entry of a transform list first, and so does an
    ``itk.CompositeTransform``, so the order carries over unchanged.
    """
    from itkwasm import Transform as ItkTransform

    if isinstance(transform, ItkTransform):
        return [transform]

    if isinstance(transform, dict):
        return [_transform_from_dict(transform)]

    if isinstance(transform, (list, tuple)) and transform:
        entries = list(transform)
        if all(isinstance(entry, ItkTransform) for entry in entries):
            return entries
        if all(isinstance(entry, dict) for entry in entries):
            return [_transform_from_dict(entry) for entry in entries]

    # An itk.Transform, including the CompositeTransform Elastix returns.
    if hasattr(transform, "GetTransformTypeAsString"):
        import itk

        as_dict = itk.dict_from_transform(transform)
        if isinstance(as_dict, dict):
            as_dict = [as_dict]
        return [_transform_from_dict(entry) for entry in as_dict]

    msg = (
        f"unsupported transform type {type(transform).__name__}. Expected an "
        "RFC-5 coordinate transformation, an itk.Transform, or an ITK-Wasm "
        "Transform / TransformList."
    )
    raise TypeError(msg)


def _transform_from_dict(entry: dict):
    from itkwasm import Transform as ItkTransform
    from itkwasm import TransformType

    entry = dict(entry)
    transform_type = entry.get("transformType")
    if isinstance(transform_type, dict):
        # itk.dict_from_transform always materializes the parameters as
        # float64 but keeps the transform's declared value type, so a
        # float-parameterized transform (e.g. DisplacementFieldTransform
        # [itk.F, 2]) arrives declared float32 with a float64 buffer. The
        # pipeline then reads the buffer as float32 and computes a region
        # from garbage. Declare the type the buffer actually has.
        parameters = entry.get("parameters")
        if parameters is not None:
            actual = str(np.asarray(parameters).dtype)
            if actual in ("float32", "float64"):
                transform_type = {**transform_type, "parametersValueType": actual}
        entry["transformType"] = TransformType(**transform_type)
    return ItkTransform(**entry)


def resample_bounding_box(
    transform,
    fixed: NgffImage,
    moving: NgffImage,
    padding: int = 1,
    *,
    fields: Mapping[str, object] | None = None,
) -> ResampleBoundingBox:
    """Compute the moving-image region needed to resample a fixed image grid.

    The region is derived from image geometry alone -- the pixel buffers of
    ``fixed`` and ``moving`` are never read, and their Dask graphs are never
    computed. That is the point: describe two images and a transform with a few
    numbers, learn exactly which block of the moving image a resample will
    touch, and only then move pixels.

    Two kinds of transform are accepted, and they are interpreted in different
    coordinate spaces:

    * An **RFC-5 coordinate transformation** acts on the intrinsic coordinate
      system, where a point is ``translation + scale * index``. Its parameters
      are in Zarr axis order and no direction matrix applies. Its ``input``
      and ``output`` identifiers are **not resolved**: the transformation is
      applied from the fixed image's intrinsic system to the moving image's,
      whatever the identifiers name.
    * An **ITK transform** (for example a ``CompositeTransform`` returned by
      Elastix) acts on ITK physical space, so the geometry is built the way
      :func:`ngff_zarr.ngff_image_to_itk_image` builds it, including the
      direction matrix derived from RFC-4 anatomical orientation.

    An ITK transform need not be linear. An RFC-5 transformation is converted
    first, so it must be one this package can convert: a linear mapping --
    ``identity``, ``scale``, ``translation``, ``rotation``, ``affine``,
    ``mapAxis``, ``byDimension``, ``bijection``, or a ``sequence`` of them --
    or a ``displacements`` or ``coordinates`` transformation whose field is
    passed in ``fields``.

    In both cases the transform maps *fixed* points into *moving* space.

    :param transform: An RFC-5 coordinate transformation, an ``itk.Transform``,
        or an ITK-Wasm ``Transform`` / ``TransformList``.

    :param fixed: The image whose grid is resampled. Geometry only.
    :type  fixed: NgffImage

    :param moving: The image to be sampled. Geometry only.
    :type  moving: NgffImage

    :param padding: Pixels of padding added per side. The default of 1 covers
        linear interpolation, which reads one neighbor beyond the continuous
        index bound. Use 0 for the tight region, or more for wider kernels.
    :type  padding: int

    :param fields: The field images an RFC-5 ``displacements`` or
        ``coordinates`` transformation points at, keyed by its ``path``, as
        :func:`ngff_zarr.ngff_transform_to_itk_transform` takes them. Required
        for those two, ignored otherwise. A field
        carrying an anatomical orientation is refused here, since this branch
        works on the intrinsic systems where none applies: convert it with
        :func:`ngff_zarr.ngff_transform_to_itk_transform`, passing ``fixed``
        and ``moving``, and pass the ITK transform it returns.
    :type  fields: Mapping[str, NgffImage | NgffMultiscales], optional

    :return: The region, keyed by dimension name in Zarr order.
    :rtype: ResampleBoundingBox

    :raises ValueError: If ``fixed`` and ``moving`` do not share the same
        spatial dimensions, if there are not 2, 3 or 4 of them, if a scale or
        translation entry is missing, zero or non-finite, if ``padding`` is
        negative, if the transform couples spatial and non-spatial axes, or if
        the region spans more than the index range the pipeline can represent.
    :raises NotImplementedError: If the RFC-5 transformation is not linear.
    """
    from itkwasm_downsample import resample_bounding_box as itkwasm_bounding_box

    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        msg = f"padding must be a non-negative integer, got {padding!r}"
        raise ValueError(msg)

    fixed_spatial = _spatial_dims(fixed)
    moving_spatial = _spatial_dims(moving)
    if fixed_spatial != moving_spatial:
        msg = (
            f"fixed image has spatial dims {tuple(fixed_spatial)} but moving "
            f"image has {tuple(moving_spatial)}; they must match"
        )
        raise ValueError(msg)
    if len(fixed_spatial) < 2:
        msg = (
            f"images have {len(fixed_spatial)} spatial dims "
            f"{tuple(fixed_spatial)}; only 2 and 3 are supported"
        )
        raise ValueError(msg)

    for label, image in (("fixed", fixed), ("moving", moving)):
        _check_geometry(label, image, fixed_spatial)

    # ITK orders points fastest-axis-first: x, then y, then z, whatever order
    # the image spells its spatial dims in. Reversing the dims is only right
    # for the canonical (z, y, x); binding by name is right for every order.
    itk_dims = [dim for dim in _SPATIAL_DIMS if dim in fixed_spatial]

    fixed_dims = tuple(fixed.dims)
    fixed_extent = {
        dim: int(fixed.data.shape[fixed_dims.index(dim)]) for dim in fixed_spatial
    }
    moving_dims = tuple(moving.dims)
    moving_shape = {
        dim: int(moving.data.shape[moving_dims.index(dim)]) for dim in moving_spatial
    }
    if any(extent == 0 for extent in fixed_extent.values()):
        # A grid with a zero-length axis has no samples to resample, so there
        # is nothing to fetch. The pipeline reports an all-zero region here;
        # returning it directly keeps the corners meaningful rather than the
        # sentinels a degenerate boundary walk produces.
        zeros = dict.fromkeys(fixed_spatial, 0)
        return ResampleBoundingBox(
            dims=tuple(fixed_spatial),
            start_index=dict(zeros),
            size=dict(zeros),
            corners_min=dict.fromkeys(fixed_spatial, 0.0),
            corners_max=dict.fromkeys(fixed_spatial, 0.0),
            padded_corners_min=dict.fromkeys(fixed_spatial, 0.0),
            padded_corners_max=dict.fromkeys(fixed_spatial, 0.0),
            moving_shape=moving_shape,
        )

    if _is_ngff_transform(transform):
        transform_list = ngff_transform_to_itk_transform(
            transform, fixed.dims, fields=fields
        )
        # An RFC-5 transformation is defined on the intrinsic coordinate
        # system, which carries no direction matrix.
        fixed_direction = np.eye(len(itk_dims))
        moving_direction = np.eye(len(itk_dims))
    else:
        transform_list = _as_itk_transform_list(transform)
        fixed_direction = _itk_direction(fixed, itk_dims)
        moving_direction = _itk_direction(moving, itk_dims)

    result = itkwasm_bounding_box(
        transform_list,
        _metadata_only_itk_image(fixed, itk_dims, fixed_direction),
        _metadata_only_itk_image(moving, itk_dims, moving_direction),
        padding=padding,
    )

    _check_index_arrays(result, itk_dims)
    _check_region_contains_corners(result, itk_dims)

    # The pipeline reports arrays fastest-axis-first; key them by dimension
    # name so no caller has to remember that, and order the mappings like
    # ``dims`` so a printed result reads in Zarr order.
    def by_dim(values) -> dict:
        indexed = dict(zip(itk_dims, values))
        return {dim: indexed[dim] for dim in fixed_spatial}

    return ResampleBoundingBox(
        dims=tuple(fixed_spatial),
        start_index={k: int(v) for k, v in by_dim(result["paddedStartIndex"]).items()},
        size={k: int(v) for k, v in by_dim(result["paddedSize"]).items()},
        corners_min={k: float(v) for k, v in by_dim(result["corners"]["min"]).items()},
        corners_max={k: float(v) for k, v in by_dim(result["corners"]["max"]).items()},
        padded_corners_min={
            k: float(v) for k, v in by_dim(result["paddedCorners"]["min"]).items()
        },
        padded_corners_max={
            k: float(v) for k, v in by_dim(result["paddedCorners"]["max"]).items()
        },
        moving_shape=moving_shape,
    )
