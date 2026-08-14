# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

from .._supported_versions import NgffVersion
from .._zarr_types import StoreLike
from ..rfc4 import AnatomicalOrientation
from ..v04.zarr_metadata import (
    AxesType as AxesTypeV04,
)
from ..v04.zarr_metadata import (
    MethodMetadata,
    Omero,
    SupportedDims,
    Units,
)

if TYPE_CHECKING:
    from ..ngff_image import NgffImage
    from ..v04.zarr_metadata import Metadata as Metadata_v04
    from ..v05.zarr_metadata import Metadata as Metadata_v05


# OME-Zarr v0.6 (RFC-5) extends the axis types with the discrete vector-field
# axes used by displacement and coordinate fields, and adds an optional
# ``discrete`` flag on the axis.
AxesType = Union[AxesTypeV04, Literal["displacement"], Literal["coordinate"]]


@dataclass
class Axis:
    name: SupportedDims
    type: AxesType
    unit: Units | None = None
    orientation: AnatomicalOrientation | None = None
    discrete: bool | None = None


@dataclass
class CoordinateSystem:
    name: str
    axes: list[Axis]


@dataclass
class CoordinateSystemIdentifier:
    """
    CoordinateSystemIdentifier field used in transformations metadata.

    There, the input/output fields of transformations must be an object with
    'path' and 'name' fields.
    """

    path: str | None = None
    name: str | None = None


@dataclass(kw_only=True)
class BaseTransform(ABC):  # noqa: B024
    input: CoordinateSystemIdentifier | None = None
    output: CoordinateSystemIdentifier | None = None
    name: str | None = None
    type: str | None = None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BaseTransform":
        return cls(**data)


@dataclass(kw_only=True)
class Identity(BaseTransform):
    type: str = "identity"


@dataclass(kw_only=True)
class Scale(BaseTransform):
    scale: list[float]
    type: str = "scale"


@dataclass(kw_only=True)
class Translation(BaseTransform):
    translation: list[float]
    type: str = "translation"


@dataclass(kw_only=True)
class Rotation(BaseTransform):
    rotation: list[list[float]]
    path: str | None = None
    type: str = "rotation"


@dataclass(kw_only=True)
class Affine(BaseTransform):
    affine: list[list[float]]
    path: str | None = None
    type: str = "affine"


@dataclass(kw_only=True)
class Coordinates(BaseTransform):
    path: str
    interpolation: str | None = None
    type: str = "coordinates"


@dataclass(kw_only=True)
class Displacements(BaseTransform):
    path: str
    interpolation: str | None = None
    type: str = "displacements"


@dataclass(kw_only=True)
class MapAxis(BaseTransform):
    """An axis permutation stored as a transpose vector of integer indices.

    The value at position ``i`` names which input axis becomes the ``i``-th
    output axis. Every zero-based input axis index appears exactly once.
    """

    mapAxis: list[int]
    type: str = "mapAxis"


Transform = Union[
    Identity,
    Scale,
    Translation,
    Rotation,
    Affine,
    Coordinates,
    Displacements,
    MapAxis,
    "ByDimension",
    "Bijection",
    "TransformSequence",
]


@dataclass
class ByDimensionItem:
    """One lower-dimensional transformation of a byDimension transform.

    ``input_axes`` and ``output_axes`` hold zero-based axis indices into the
    parent byDimension's input and output coordinate systems.
    """

    transformation: Transform
    input_axes: list[int]
    output_axes: list[int]


@dataclass(kw_only=True)
class ByDimension(BaseTransform):
    """A high dimensional transform built from lower dimensional ones.

    Every axis index of the output coordinate system appears in exactly one
    item's ``output_axes``.
    """

    transformations: list[ByDimensionItem]
    type: str = "byDimension"


@dataclass(kw_only=True)
class Bijection(BaseTransform):
    """An invertible transform with explicit forward and inverse directions."""

    forward: Transform
    inverse: Transform
    type: str = "bijection"


@dataclass(kw_only=True)
class TransformSequence(BaseTransform):
    transformations: list[Transform]
    name: str | None = "transformSequence"
    type: str = "sequence"


def _axis_count(
    identifier: CoordinateSystemIdentifier | None,
    coordinateSystems: list[CoordinateSystem],
) -> int | None:
    """Number of axes of the referenced coordinate system, when resolvable."""
    if identifier is None or identifier.name is None:
        return None
    for coordinate_system in coordinateSystems:
        if coordinate_system.name == identifier.name:
            return len(coordinate_system.axes)
    return None


def _item_dimensions(transformation: Transform) -> int | None:
    """Dimensionality a byDimension item's axis lists must have, if knowable."""
    if isinstance(transformation, Scale):
        return len(transformation.scale)
    if isinstance(transformation, Translation):
        return len(transformation.translation)
    if isinstance(transformation, MapAxis):
        return len(transformation.mapAxis)
    return None


def _require_integer_axes(axes: list, context: str) -> None:
    """Axis indices are zero-based integer positions; reject anything else."""
    for axis in axes:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError(f"{context} axis indices must be integers; got {axis!r}")


def validate_transform(
    transformation: Transform,
    coordinateSystems: list[CoordinateSystem] | None = None,
) -> None:
    """Check the RFC-5 constraints the JSON schema cannot express.

    Raises ``ValueError`` when the transform's parameters contradict the
    spec: a ``mapAxis`` that is not a permutation, a ``byDimension`` whose
    items skip or duplicate an output axis, or a ``bijection`` between
    coordinate systems of different dimensionality. Dimension checks against
    the ``input`` and ``output`` coordinate systems run when those resolve
    to a known system and are skipped otherwise, as for the wrapped
    transforms that omit them.
    """
    coordinateSystems = coordinateSystems or []

    if isinstance(transformation, MapAxis):
        indices = transformation.mapAxis
        _require_integer_axes(indices, "mapAxis")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(
                "mapAxis must be a permutation holding every zero-based "
                f"input axis index exactly once; got {indices}"
            )
        for identifier in (transformation.input, transformation.output):
            count = _axis_count(identifier, coordinateSystems)
            if count is not None and count != len(indices):
                raise ValueError(
                    f"mapAxis length {len(indices)} does not match the "
                    f"{count} axes of coordinate system '{identifier.name}'"
                )
    elif isinstance(transformation, ByDimension):
        input_count = _axis_count(transformation.input, coordinateSystems)
        seen_output_axes: set[int] = set()
        for item in transformation.transformations:
            axes = list(item.input_axes) + list(item.output_axes)
            _require_integer_axes(axes, "byDimension")
            if any(axis < 0 for axis in axes):
                raise ValueError(
                    f"byDimension axis indices must be non-negative; got {axes}"
                )
            if input_count is not None and any(
                axis >= input_count for axis in item.input_axes
            ):
                raise ValueError(
                    f"byDimension input axes {item.input_axes} exceed the "
                    f"{input_count} axes of coordinate system "
                    f"'{transformation.input.name}'"
                )
            duplicated = seen_output_axes.intersection(item.output_axes)
            if duplicated or len(set(item.output_axes)) != len(item.output_axes):
                raise ValueError(
                    "byDimension output axes must each be produced by exactly "
                    f"one transformation; axis {sorted(duplicated) or item.output_axes} "
                    "appears more than once"
                )
            seen_output_axes.update(item.output_axes)
            dimensions = _item_dimensions(item.transformation)
            if dimensions is not None and (
                len(item.input_axes) != dimensions
                or len(item.output_axes) != dimensions
            ):
                raise ValueError(
                    f"byDimension item of type '{item.transformation.type}' is "
                    f"{dimensions}-dimensional but maps {len(item.input_axes)} "
                    f"input axes to {len(item.output_axes)} output axes"
                )
        count = _axis_count(transformation.output, coordinateSystems)
        if count is not None and seen_output_axes != set(range(count)):
            raise ValueError(
                "byDimension items must cover every output axis exactly once; "
                f"coordinate system '{transformation.output.name}' has {count} "
                f"axes but the items produce {sorted(seen_output_axes)}"
            )
    elif isinstance(transformation, Bijection):
        input_count = _axis_count(transformation.input, coordinateSystems)
        output_count = _axis_count(transformation.output, coordinateSystems)
        if (
            input_count is not None
            and output_count is not None
            and input_count != output_count
        ):
            raise ValueError(
                "bijection input and output coordinate systems must have the "
                f"same dimensionality; got {input_count} and {output_count}"
            )


@dataclass
class Dataset:
    """
    Dataset in the multiscales metadata.

    path: Path to the dataset within the Zarr store.
    coordinateTransformations:
        List of transformations to map from dataset.
        Must be one of
        - Single identity
        - single scale
        - sequence of scale and translation
    """

    path: str
    coordinateTransformations: list[Transform]


@dataclass
class Metadata:
    coordinateSystems: list[CoordinateSystem]
    datasets: list[Dataset]
    coordinateTransformations: list[Transform] | None = None
    omero: Omero | None = None
    name: str = "image"
    type: str | None = None
    metadata: MethodMetadata | None = None
    #: Unrecognized keys captured on read (see
    #: :attr:`ngff_zarr.v04.zarr_metadata.Metadata.extra`). Carried across
    #: version conversion so a value read back through ``to_version`` retains
    #: the field; a read-side validation aid that is never serialized.
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        We use the post-init to do some on-the-fly validation
        that is not covered by the json schemas well.
        """

        _ = self.intrinsic_coordinate_system

    @property
    def intrinsic_coordinate_system(self) -> CoordinateSystem:
        output_cs = [ds.coordinateTransformations[0].output for ds in self.datasets]

        if output_cs[0] is None:
            raise ValueError(
                "No output coordinate system found in dataset coordinate transformations. "
            )

        # raise an error if these aren't all the same
        if not all(output == output_cs[0] for output in output_cs):
            raise ValueError(
                "Multiple different outputs coordinate systems found in"
                f" coordinate transformations for multiscales: {output_cs}. "
                "This is out of spec for this ome-zarr 0.6."
            )

        # find the cs in the list of coordinate systems with the same name as the output
        return next(cs for cs in self.coordinateSystems if cs.name == output_cs[0].name)

    def to_version(
        self, version: Union[str, NgffVersion]
    ) -> Union["Metadata", "Metadata_v05", "Metadata_v04"]:
        if isinstance(version, str):
            # raise error for invalid version string
            version = NgffVersion(version)

        if version == NgffVersion.V04:
            return self._to_v05()._to_v04()
        if version == NgffVersion.V05:
            return self._to_v05()
        if version == NgffVersion.V06:
            return self
        raise ValueError(f"Unsupported version conversion: 0.6 -> {version}")

    @classmethod
    def from_version(
        cls, metadata: Union["Metadata", "Metadata_v05", "Metadata_v04"]
    ) -> "Metadata":
        from ..v04.zarr_metadata import Metadata as Metadata_v04
        from ..v05.zarr_metadata import Metadata as Metadata_v05

        if isinstance(metadata, Metadata_v05):
            return cls._from_v05(metadata)
        if isinstance(metadata, Metadata_v04):
            metadata_v05 = Metadata_v05._from_v04(metadata)
            return cls._from_v05(metadata_v05)
        raise ValueError(
            f"Unsupported metadata type ({type(metadata)}) for conversion to v0.6"
        )

    def _to_v05(self) -> "Metadata_v05":
        from ..v04.zarr_metadata import Scale as Scale_v05
        from ..v04.zarr_metadata import Translation as Translation_v05
        from ..v05.zarr_metadata import Dataset as Dataset_v05
        from ..v05.zarr_metadata import Metadata as Metadata_v05

        datasets = []
        for idx, ds in enumerate(self.datasets):
            path = ds.path

            coordinateTransformations = []
            transforms = ds.coordinateTransformations

            # set reasonable defaults
            spatial_dims = ("x", "y", "z")
            scale = Scale(
                scale=[
                    2.0**idx if d in spatial_dims else 1.0 for d in self.dimension_names
                ]
            )
            translation = Translation(translation=[0.0 for d in self.dimension_names])
            outputs = []

            for transform in transforms:
                if isinstance(transform, TransformSequence):
                    for t in transform.transformations:
                        if isinstance(t, Scale):
                            scale = t
                        elif isinstance(t, Identity):
                            scale = Scale(scale=[1.0 for d in self.dimension_names])
                        elif isinstance(t, Translation):
                            translation = t
                elif isinstance(transform, Scale):
                    scale = transform
                elif isinstance(transform, Translation):
                    translation = transform
                elif isinstance(transform, Identity):
                    scale = Scale(scale=[1.0 for d in self.dimension_names])
                    translation = Translation(
                        translation=[0.0 for d in self.dimension_names]
                    )

                outputs.append(transform.output)

            # make sure all outputs are the same
            if not all(output == outputs[0] for output in outputs):
                raise ValueError(
                    "Multiple different outputs coordinate systems found in"
                    f" coordinate transformations for multiscales: {transforms}. "
                )
            output = outputs[0]

            scale = Scale_v05(scale=scale.scale)
            translation = Translation_v05(translation=translation.translation)
            coordinateTransformations = [scale, translation]

            cs = next(cs for cs in self.coordinateSystems if cs.name == output.name)

            datasets.append(
                Dataset_v05(
                    path=path,
                    coordinateTransformations=coordinateTransformations,
                )
            )

        metadata = Metadata_v05(
            axes=cs.axes,
            datasets=datasets,
            coordinateTransformations=None,
            name=self.name,
            metadata=self.metadata,
            type=self.type,
            omero=self.omero,
        )

        return metadata

    @classmethod
    def _from_v05(cls, metadata_v05: "Metadata_v05") -> "Metadata":
        from ..v04.zarr_metadata import Scale as Scale_v05
        from ..v04.zarr_metadata import Translation as Translation_v05

        coordinate_systems = [
            CoordinateSystem(name="intrinsic", axes=metadata_v05.axes)
        ]

        datasets = []
        for index, ds in enumerate(metadata_v05.datasets):
            scale = [1.0 for d in metadata_v05.dimension_names]
            translation = [0.0 for d in metadata_v05.dimension_names]

            for transform in ds.coordinateTransformations:
                if isinstance(transform, Scale_v05):
                    scale = transform.scale
                elif isinstance(transform, Translation_v05):
                    translation = transform.translation

            sequence = TransformSequence(
                transformations=[
                    Scale(scale=scale),
                    Translation(translation=translation),
                ],
                input=CoordinateSystemIdentifier(path=ds.path),
                name=f"scale{index}_to_intrinsic",
                output=CoordinateSystemIdentifier(name=coordinate_systems[0].name),
            )

            datasets.append(
                Dataset(
                    path=ds.path,
                    coordinateTransformations=[sequence],
                )
            )

        metadata = cls(
            coordinateSystems=coordinate_systems,
            datasets=datasets,
            name=metadata_v05.name,
            omero=metadata_v05.omero,
            coordinateTransformations=None,
        )

        return metadata

    @classmethod
    def _from_zarr_attrs(
        cls,
        root_attrs: dict,
        store: StoreLike,
        validate: bool = False,
        subpath: str | None = None,
    ) -> tuple["Metadata", list["NgffImage"]]:
        """Create Metadata instance from ome-zarr metadata dictionary.

        When the multiscales group is nested inside the store (e.g. a
        displacement field written next to its image), ``subpath`` locates the
        group so dataset arrays resolve against the right prefix. Dataset
        paths in the returned metadata stay relative to the group.
        """
        import posixpath
        import sys

        import dask.array

        from ..ngff_image import NgffImage
        from ..parse_metadata import _parse_omero
        from ..rfc4_validation import (
            has_rfc4_orientation_metadata,
            validate_rfc4_orientation,
        )
        from ..validate import validate as validate_ngff

        # make sure root_attrs['ome]['multiscales'] exists
        if "ome" not in root_attrs or "multiscales" not in root_attrs["ome"]:
            raise ValueError(
                "Invalid OME-Zarr metadata: missing 'ome' or 'multiscales' field."
            )

        if validate:
            validate_ngff(
                root_attrs,
                version=root_attrs["ome"]["multiscales"][0].get("version", "0.6"),
            )

            # RFC 4 validation for anatomical orientation
            if "axes" in root_attrs["ome"]["multiscales"][0] and isinstance(
                root_attrs["ome"]["multiscales"][0]["axes"], list
            ):
                # Type cast each axis item to dict for validation
                axes_dicts = []
                for axis in root_attrs["ome"]["multiscales"][0]["axes"]:
                    if isinstance(axis, dict):
                        axes_dicts.append(axis)
                if axes_dicts and has_rfc4_orientation_metadata(axes_dicts):
                    validate_rfc4_orientation(axes_dicts)

        omero = _parse_omero(root_attrs.get("ome", {}).get("omero"))
        root_attrs = root_attrs["ome"]["multiscales"][0]

        coordinate_systems = []
        for cs in root_attrs.get("coordinateSystems", []):
            axes = [Axis(**axis) for axis in cs["axes"]]

            coordinate_systems.append(CoordinateSystem(name=cs["name"], axes=axes))

        if not coordinate_systems:
            raise ValueError(
                "Invalid OME-Zarr v0.6 metadata: "
                " missing 'coordinateSystems' in multiscales metadata."
            )

        images = []
        datasets = []
        for index, dataset in enumerate(root_attrs["datasets"]):
            dataset_path = dataset["path"]
            if subpath:
                dataset_path = posixpath.join(subpath, dataset_path)
            data = dask.array.from_zarr(store, component=dataset_path)
            # Convert endianness to native if needed
            if (sys.byteorder == "little" and data.dtype.byteorder == ">") or (
                sys.byteorder == "big" and data.dtype.byteorder == "<"
            ):
                data = data.astype(data.dtype.newbyteorder())

            # parse dataset coordinate transformations if present
            dims = [ax.name for ax in coordinate_systems[0].axes]
            scale = Scale(scale=[1.0 for d in dims])
            translation = Translation(translation=[0.0 for d in dims])
            coordinateTransformations: list[Transform] = [
                TransformSequence(
                    transformations=[scale, translation],
                    input=CoordinateSystemIdentifier(path=dataset["path"]),
                    output=CoordinateSystemIdentifier(name=coordinate_systems[0].name),
                    name=f"scale_{index}_to_{coordinate_systems[0].name}",
                )
            ]
            if "coordinateTransformations" in dataset:
                coordinateTransformations = cls._parse_transforms(
                    dataset["coordinateTransformations"], coordinate_systems
                )

                # extract scale and translation for ngff_image convenience
                for transform in coordinateTransformations:
                    if isinstance(transform, TransformSequence):
                        for t in transform.transformations:
                            if isinstance(t, Scale):
                                scale = t
                            elif isinstance(t, Identity):
                                scale = Scale(scale=[1.0 for d in dims])
                            elif isinstance(t, Translation):
                                translation = t
                    elif isinstance(transform, Scale):
                        scale = transform
                    elif isinstance(transform, Translation):
                        translation = transform
                    elif isinstance(transform, Identity):
                        scale = Scale(scale=[1.0 for d in dims])
                        translation = Translation(translation=[0.0 for d in dims])
                    else:
                        raise ValueError(
                            "Unsupported transform type: "
                            f"{transform.type} in dataset {dataset['path']}"
                        )

            cs_intrinsic = next(
                cs
                for cs in coordinate_systems
                if cs.name == coordinateTransformations[0].output.name
            )

            datasets.append(
                Dataset(
                    path=dataset["path"],
                    coordinateTransformations=coordinateTransformations,
                )
            )

            ngff_image = NgffImage(
                data=data,
                dims=dims,
                scale=dict(zip(dims, scale.scale)),
                translation=dict(zip(dims, translation.translation)),
                name=root_attrs.get("name", "image"),
                axes_units=dict(zip(dims, [ax.unit for ax in cs_intrinsic.axes])),
            )
            images.append(ngff_image)

        additionalTransformations = root_attrs.get("coordinateTransformations", None)
        if additionalTransformations is not None:
            additionalTransformations = cls._parse_transforms(
                additionalTransformations, coordinate_systems
            )

        metadata = cls(
            coordinateSystems=coordinate_systems,
            datasets=datasets,
            name=root_attrs.get("name", "image"),
            omero=omero,
            coordinateTransformations=additionalTransformations,
        )

        return metadata, images

    @classmethod
    def _parse_transforms(
        cls, transforms: list[dict], coordinateSystems: list[CoordinateSystem]
    ) -> list[Transform]:
        """
        Parse a list of possibly nested transformation dictionaries into Transform instances.
        """
        parsed_transforms = []
        for transform in transforms:
            if transform["type"] == "identity":
                transformation = Identity()
            elif transform["type"] == "scale":
                transformation = Scale.from_dict(transform)
            elif transform["type"] == "translation":
                transformation = Translation.from_dict(transform)
            elif transform["type"] == "rotation":
                transformation = Rotation.from_dict(transform)
            elif transform["type"] == "affine":
                transformation = Affine.from_dict(transform)
            elif transform["type"] == "coordinates":
                transformation = Coordinates.from_dict(transform)
            elif transform["type"] == "displacements":
                transformation = Displacements.from_dict(transform)
            elif transform["type"] == "mapAxis":
                transformation = MapAxis(mapAxis=list(transform["mapAxis"]))
            elif transform["type"] == "byDimension":
                items = []
                for item in transform["transformations"]:
                    (sub_transform,) = cls._parse_transforms(
                        [item["transformation"]], coordinateSystems
                    )
                    items.append(
                        ByDimensionItem(
                            transformation=sub_transform,
                            input_axes=list(item["input_axes"]),
                            output_axes=list(item["output_axes"]),
                        )
                    )
                transformation = ByDimension(transformations=items)
            elif transform["type"] == "bijection":
                (forward,) = cls._parse_transforms(
                    [transform["forward"]], coordinateSystems
                )
                (inverse,) = cls._parse_transforms(
                    [transform["inverse"]], coordinateSystems
                )
                transformation = Bijection(forward=forward, inverse=inverse)
            elif transform["type"] == "sequence":
                # TODO: Undo nested sequences on import?
                sub_transforms = cls._parse_transforms(
                    transform["transformations"], coordinateSystems
                )
                transformation = TransformSequence(transformations=sub_transforms)
            else:
                raise ValueError(f"Unsupported transform type: {transform['type']}")

            # resolve input/output to CoordinateSystem instances if matching name found
            # because coordinate system may not exist for multiscale transforms
            # where input is a dataset path
            coordinate_system_names = [cs.name for cs in coordinateSystems]
            tf_input = transform.get("input", None)
            tf_output = transform.get("output", None)

            input = None
            output = None
            if isinstance(tf_input, dict):
                input = CoordinateSystemIdentifier()
                if "name" in tf_input and tf_input["name"] in coordinate_system_names:
                    input.name = tf_input["name"]
                if "path" in tf_input:
                    input.path = tf_input["path"]

            if isinstance(tf_output, dict):
                output = CoordinateSystemIdentifier()
                if "name" in tf_output and tf_output["name"] in coordinate_system_names:
                    output.name = tf_output["name"]
                if "path" in tf_output:
                    output.path = tf_output["path"]

            transformation.input = input
            transformation.output = output
            # The wrapper branches above construct their transform from the
            # payload fields alone, so the optional name is restored here for
            # every type.
            if transform.get("name") is not None:
                transformation.name = transform["name"]
            validate_transform(transformation, coordinateSystems)
            parsed_transforms.append(transformation)

        return parsed_transforms

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.coordinateSystems[0].axes])
