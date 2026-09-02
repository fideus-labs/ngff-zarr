# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import copy
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

from .._store_types import StoreLike
from .._supported_versions import (
    V06_ONDISK_VERSION,
    V06_SUPERSEDED_TAGS,
    NgffVersion,
)
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
    from ..v09.zarr_metadata import Metadata as Metadata_v09


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

    def axis_count(
        self, coordinateSystems: list[CoordinateSystem] | None
    ) -> int | None:
        """Number of axes of the referenced coordinate system, when resolvable.

        Returns ``None`` when this identifier names no system in
        ``coordinateSystems``, as for the wrapped transforms that omit
        ``input`` and ``output``.
        """
        if self.name is None or not coordinateSystems:
            return None
        for coordinate_system in coordinateSystems:
            if coordinate_system.name == self.name:
                return len(coordinate_system.axes)
        return None


def _resolved_axis_count(
    identifier: CoordinateSystemIdentifier | None,
    coordinateSystems: list[CoordinateSystem] | None,
) -> int | None:
    return None if identifier is None else identifier.axis_count(coordinateSystems)


def _require_keys(data: dict, keys: tuple[str, ...], context: str) -> None:
    """Reject a transformation payload that omits a required field."""
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(
            f"{context} transformation is missing required field(s) "
            f"{', '.join(missing)}"
        )


def _require_integer_axes(axes: list, context: str) -> None:
    """Axis indices are zero-based integer positions; reject anything else."""
    for axis in axes:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise ValueError(f"{context} axis indices must be integers; got {axis!r}")


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

    def validate(  # noqa: B027
        self,
        coordinateSystems: list[CoordinateSystem] | None = None,
        version: object | None = None,
    ) -> None:
        """Check this transform's RFC-5 constraints.

        Constraints that follow from the parameters alone are enforced by
        ``__post_init__``, so an invalid instance cannot be constructed;
        this method checks them again, which covers instances mutated after
        construction. Constraints against the ``input`` and ``output``
        coordinate systems run when those resolve in ``coordinateSystems``
        and are skipped otherwise. Raises ``ValueError`` on the first
        violation.

        ``version`` selects the bounds a version-dependent rule applies; a
        rule that holds at every version ignores it.

        Transform types without constraints beyond their field types, such
        as ``Scale`` or ``Identity``, inherit this no-op deliberately.
        """


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

    def __post_init__(self) -> None:
        self._check_intrinsic()

    def _check_intrinsic(self) -> None:
        """The permutation rules, which every version states identically."""
        indices = self.mapAxis
        _require_integer_axes(indices, "mapAxis")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(
                "mapAxis must be a permutation holding every zero-based "
                f"input axis index exactly once; got {indices}"
            )

    def validate(
        self,
        coordinateSystems: list[CoordinateSystem] | None = None,
        version: object | None = None,
    ) -> None:
        from ..structural_validation import is_rfc3_axis_model_allowed

        self._check_intrinsic()
        # The 2-to-5 arity mirrors the minItems and maxItems the 0.4 through
        # 0.6 schemas set, and follows from their five-axis cap. RFC-3 lifts
        # that cap at 0.9.dev1, whose mapAxis definition sets neither bound, so
        # a permutation over six axes is valid there and the check stands down.
        # It cannot live in __post_init__, which has no version to consult.
        if not is_rfc3_axis_model_allowed(version) and not (
            2 <= len(self.mapAxis) <= 5
        ):
            raise ValueError(
                "mapAxis must hold between 2 and 5 indices, one per axis "
                f"of the coordinate systems it permutes; got {self.mapAxis}"
            )
        for identifier in (self.input, self.output):
            count = _resolved_axis_count(identifier, coordinateSystems)
            if count is not None and count != len(self.mapAxis):
                raise ValueError(
                    f"mapAxis length {len(self.mapAxis)} does not match the "
                    f"{count} axes of coordinate system '{identifier.name}'"
                )


#: The ``maxItems`` the schema sets on ``droppedInputs`` and ``createdOutputs``.
#: Its companion ``maximum: 4`` is not mirrored: an index is bounded by the
#: coordinate system it points into, which ``validate`` checks.
_PROJECT_AXIS_MAX_OPERATIONS = 3


@dataclass(kw_only=True)
class ProjectAxis(BaseTransform):
    """A projection that drops input axes and inserts zero-valued output axes.

    ``droppedInputs`` are indices of the input vector to remove,
    ``createdOutputs`` indices of the output vector to zero-fill; at least one
    is given. Output dimensionality is the input less the dropped plus the
    created. Dropping loses information, so it is not invertible in general.
    """

    droppedInputs: list[int] | None = None
    createdOutputs: list[int] | None = None
    type: str = "projectAxis"

    def __post_init__(self) -> None:
        self._check_intrinsic()

    def _check_intrinsic(self) -> None:
        if self.droppedInputs is None and self.createdOutputs is None:
            raise ValueError(
                "projectAxis must declare droppedInputs, createdOutputs, or both"
            )
        for field_name in ("droppedInputs", "createdOutputs"):
            indices = getattr(self, field_name)
            if indices is None:
                continue
            _require_integer_axes(indices, field_name)
            if not indices:
                raise ValueError(f"{field_name} must hold at least one index")
            if any(index < 0 for index in indices):
                raise ValueError(
                    f"{field_name} indices are zero-based positions and must "
                    f"not be negative; got {indices}"
                )
            if len(set(indices)) != len(indices):
                raise ValueError(f"{field_name} indices must be unique; got {indices}")
            if len(indices) > _PROJECT_AXIS_MAX_OPERATIONS:
                raise ValueError(
                    f"{field_name} may name at most "
                    f"{_PROJECT_AXIS_MAX_OPERATIONS} axes; got {indices}"
                )

    def validate(self, coordinateSystems: list[CoordinateSystem] | None = None) -> None:
        self._check_intrinsic()
        dropped = self.droppedInputs or []
        created = self.createdOutputs or []
        input_count = _resolved_axis_count(self.input, coordinateSystems)
        output_count = _resolved_axis_count(self.output, coordinateSystems)

        if input_count is not None:
            if len(dropped) > input_count:
                raise ValueError(
                    f"projectAxis drops {len(dropped)} axes, more than the "
                    f"{input_count} axes of coordinate system "
                    f"'{self.input.name}'"
                )
            beyond = [index for index in dropped if index >= input_count]
            if beyond:
                raise ValueError(
                    f"droppedInputs indices {beyond} are beyond the "
                    f"{input_count} axes of coordinate system "
                    f"'{self.input.name}'"
                )

        if output_count is not None:
            if len(created) > output_count:
                raise ValueError(
                    f"projectAxis creates {len(created)} axes, more than the "
                    f"{output_count} axes of coordinate system "
                    f"'{self.output.name}'"
                )
            beyond = [index for index in created if index >= output_count]
            if beyond:
                raise ValueError(
                    f"createdOutputs indices {beyond} are beyond the "
                    f"{output_count} axes of coordinate system "
                    f"'{self.output.name}'"
                )

        if input_count is not None and output_count is not None:
            expected = input_count - len(dropped) + len(created)
            if expected != output_count:
                raise ValueError(
                    f"projectAxis maps {input_count} input axes to {expected} "
                    f"output axes, but coordinate system '{self.output.name}' "
                    f"declares {output_count}"
                )


Transform = Union[
    Identity,
    Scale,
    Translation,
    Rotation,
    Affine,
    Coordinates,
    Displacements,
    MapAxis,
    ProjectAxis,
    "ByDimension",
    "Bijection",
    "TransformSequence",
]


def _item_dimensions(transformation: "Transform") -> int | None:
    """Dimensionality a byDimension item's axis lists must have, if knowable."""
    if isinstance(transformation, Scale):
        return len(transformation.scale)
    if isinstance(transformation, Translation):
        return len(transformation.translation)
    if isinstance(transformation, MapAxis):
        return len(transformation.mapAxis)
    return None


_BY_DIMENSION_LEGACY_KEYS = {"input_axes": "inputAxes", "output_axes": "outputAxes"}


@dataclass
class ByDimensionItem:
    """One lower-dimensional transformation of a byDimension transform.

    ``inputAxes`` and ``outputAxes`` hold zero-based axis indices into the
    parent byDimension's input and output coordinate systems.
    """

    transformation: Transform
    inputAxes: list[int]
    outputAxes: list[int]

    def __post_init__(self) -> None:
        self._check_intrinsic()

    def _check_intrinsic(self) -> None:
        axes = list(self.inputAxes) + list(self.outputAxes)
        _require_integer_axes(axes, "byDimension")
        if any(axis < 0 for axis in axes):
            raise ValueError(
                f"byDimension axis indices must be non-negative; got {axes}"
            )
        if len(set(self.outputAxes)) != len(self.outputAxes):
            raise ValueError(
                "byDimension output axes must each be produced by exactly "
                f"one transformation; {self.outputAxes} repeats an axis"
            )
        dimensions = _item_dimensions(self.transformation)
        if dimensions is not None and (
            len(self.inputAxes) != dimensions or len(self.outputAxes) != dimensions
        ):
            raise ValueError(
                f"byDimension item of type '{self.transformation.type}' is "
                f"{dimensions}-dimensional but maps {len(self.inputAxes)} "
                f"input axes to {len(self.outputAxes)} output axes"
            )

    @classmethod
    def from_dict(
        cls, data: dict, coordinateSystems: list[CoordinateSystem] | None = None
    ) -> "ByDimensionItem":
        # ngff-zarr 0.43.0 wrote these two keys in snake_case; the spec and the
        # 0.6rc0 schema spell them inputAxes and outputAxes, which is what is
        # written now. Both spellings are read, the spec one taking precedence
        # when a document carries both.
        for legacy, canonical in _BY_DIMENSION_LEGACY_KEYS.items():
            if legacy in data and canonical not in data:
                data = {**data, canonical: data[legacy]}
        _require_keys(
            data, ("transformation", "inputAxes", "outputAxes"), "byDimension item"
        )
        (transformation,) = Metadata._parse_transforms(
            [data["transformation"]], coordinateSystems or []
        )
        return cls(
            transformation=transformation,
            inputAxes=list(data["inputAxes"]),
            outputAxes=list(data["outputAxes"]),
        )


@dataclass(kw_only=True)
class ByDimension(BaseTransform):
    """A high dimensional transform built from lower dimensional ones.

    Every axis index of the output coordinate system appears in exactly one
    item's ``outputAxes``.
    """

    transformations: list[ByDimensionItem]
    type: str = "byDimension"

    def __post_init__(self) -> None:
        self._check_intrinsic()

    def _check_intrinsic(self) -> None:
        seen_outputAxes: set[int] = set()
        for item in self.transformations:
            duplicated = seen_outputAxes.intersection(item.outputAxes)
            if duplicated:
                raise ValueError(
                    "byDimension output axes must each be produced by exactly "
                    f"one transformation; axis {sorted(duplicated)} appears "
                    "more than once"
                )
            seen_outputAxes.update(item.outputAxes)

    @property
    def produced_outputAxes(self) -> set[int]:
        """The output axis indices the items produce, taken together."""
        axes: set[int] = set()
        for item in self.transformations:
            axes.update(item.outputAxes)
        return axes

    def validate(
        self,
        coordinateSystems: list[CoordinateSystem] | None = None,
        version: object | None = None,
    ) -> None:
        for item in self.transformations:
            item._check_intrinsic()
            # A wrapper passes the version down: a mapAxis nested here obeys
            # the bounds of the document's version, not of v0.6.
            item.transformation.validate(coordinateSystems, version)
        self._check_intrinsic()
        input_count = _resolved_axis_count(self.input, coordinateSystems)
        if input_count is not None:
            for item in self.transformations:
                if any(axis >= input_count for axis in item.inputAxes):
                    raise ValueError(
                        f"byDimension input axes {item.inputAxes} exceed the "
                        f"{input_count} axes of coordinate system "
                        f"'{self.input.name}'"
                    )
        output_count = _resolved_axis_count(self.output, coordinateSystems)
        if output_count is not None:
            produced = self.produced_outputAxes
            if produced != set(range(output_count)):
                raise ValueError(
                    "byDimension items must cover every output axis exactly "
                    f"once; coordinate system '{self.output.name}' has "
                    f"{output_count} axes but the items produce {sorted(produced)}"
                )

    @classmethod
    def from_dict(
        cls, data: dict, coordinateSystems: list[CoordinateSystem] | None = None
    ) -> "ByDimension":
        _require_keys(data, ("transformations",), "byDimension")
        return cls(
            transformations=[
                ByDimensionItem.from_dict(item, coordinateSystems)
                for item in data["transformations"]
            ]
        )


@dataclass(kw_only=True)
class Bijection(BaseTransform):
    """An invertible transform with explicit forward and inverse directions."""

    forward: Transform
    inverse: Transform
    type: str = "bijection"

    def validate(
        self,
        coordinateSystems: list[CoordinateSystem] | None = None,
        version: object | None = None,
    ) -> None:
        # A wrapper passes the version down, as byDimension does.
        self.forward.validate(coordinateSystems, version)
        self.inverse.validate(coordinateSystems, version)
        input_count = _resolved_axis_count(self.input, coordinateSystems)
        output_count = _resolved_axis_count(self.output, coordinateSystems)
        if (
            input_count is not None
            and output_count is not None
            and input_count != output_count
        ):
            raise ValueError(
                "bijection input and output coordinate systems must have the "
                f"same dimensionality; got {input_count} and {output_count}"
            )

    @classmethod
    def from_dict(
        cls, data: dict, coordinateSystems: list[CoordinateSystem] | None = None
    ) -> "Bijection":
        _require_keys(data, ("forward", "inverse"), "bijection")
        systems = coordinateSystems or []
        (forward,) = Metadata._parse_transforms([data["forward"]], systems)
        (inverse,) = Metadata._parse_transforms([data["inverse"]], systems)
        return cls(forward=forward, inverse=inverse)


@dataclass(kw_only=True)
class TransformSequence(BaseTransform):
    transformations: list[Transform]
    name: str | None = "transformSequence"
    type: str = "sequence"

    def validate(
        self,
        coordinateSystems: list[CoordinateSystem] | None = None,
        version: object | None = None,
    ) -> None:
        for transformation in self.transformations:
            transformation.validate(coordinateSystems, version)


def validate_transform(
    transformation: Transform,
    coordinateSystems: list[CoordinateSystem] | None = None,
    version: object | None = None,
) -> None:
    """Check a transform's RFC-5 constraints; see ``BaseTransform.validate``.

    Constraints that follow from the parameters alone hold by construction.
    This also checks the ones that need the ``input`` and ``output``
    coordinate systems, when those resolve in ``coordinateSystems``.
    """
    transformation.validate(coordinateSystems)


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

    @property
    def axes(self) -> list[Axis]:
        """The intrinsic coordinate system's axes.

        A property, not a field, so ``dataclasses.asdict`` and
        ``dataclasses.fields`` ignore it and the serialized entry is unchanged.
        The axis rules in :mod:`ngff_zarr.structural_validation` read
        ``metadata.axes``.
        """
        return self.intrinsic_coordinate_system.axes

    def to_version(
        self, version: Union[str, NgffVersion]
    ) -> Union["Metadata", "Metadata_v05", "Metadata_v04", "Metadata_v09"]:
        if isinstance(version, str):
            # raise error for invalid version string
            version = NgffVersion(version)

        if version == NgffVersion.V04:
            return self._to_v05()._to_v04()
        if version == NgffVersion.V05:
            return self._to_v05()
        if version == NgffVersion.V06:
            return self
        if version == NgffVersion.V09dev1:
            from ..v09.zarr_metadata import Metadata as Metadata_v09

            return Metadata_v09.from_version(self)
        raise ValueError(f"Unsupported version conversion: 0.6 -> {version}")

    @classmethod
    def from_version(
        cls,
        metadata: Union["Metadata", "Metadata_v05", "Metadata_v04", "Metadata_v09"],
    ) -> "Metadata":
        from ..v04.zarr_metadata import Metadata as Metadata_v04
        from ..v05.zarr_metadata import Metadata as Metadata_v05
        from ..v09.zarr_metadata import Metadata as Metadata_v09

        if isinstance(metadata, Metadata_v09):
            return metadata._to_v06()
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

        from .._zarrista_utils import open_lazy_array
        from ..ngff_image import NgffImage, non_default_axes_types
        from ..parse_metadata import _parse_omero
        from ..validate import validate as validate_ngff

        # make sure root_attrs['ome]['multiscales'] exists
        if "ome" not in root_attrs or "multiscales" not in root_attrs["ome"]:
            raise ValueError(
                "Invalid OME-Zarr metadata: missing 'ome' or 'multiscales' field."
            )

        # From 0.6 the version is recorded on the ``ome`` namespace rather than
        # on each multiscales entry, as the pre-release string the store was
        # written with. The per-entry value is the fallback, as on the v0.4 read
        # path. Read unconditionally: the transform rules whose bounds RFC-3
        # lifts need it whether or not the schema pass runs.
        declared_version = str(
            root_attrs["ome"].get("version")
            or root_attrs["ome"]["multiscales"][0].get("version")
            or "0.6"
        )

        if validate:
            schema_version = declared_version
            schema_attrs = root_attrs
            if schema_version in V06_SUPERSEDED_TAGS:
                # The bundled 0.6 schemas accept one tag, the pre-release they
                # were published with. A store tagged with one an earlier
                # release wrote differs from a valid store in that string
                # alone, so the rest of the document is validated with the tag
                # substituted, and the substitution is reported:
                # ``upgrade_ome_zarr`` rewrites the tag in place.
                warnings.warn(
                    f"OME-Zarr store carries the superseded 0.6 pre-release tag "
                    f"{schema_version!r}; the bundled schemas are tagged "
                    f"{V06_ONDISK_VERSION.value!r}. Validating the rest of the "
                    "document against them. upgrade_ome_zarr(store, "
                    "version='0.6') rewrites the tag.",
                    stacklevel=2,
                )
                schema_attrs = copy.deepcopy(root_attrs)
                schema_attrs["ome"]["version"] = V06_ONDISK_VERSION.value
                schema_version = V06_ONDISK_VERSION.value
            validate_ngff(schema_attrs, version=schema_version)

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
            data = open_lazy_array(store, dataset_path)
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
                    dataset["coordinateTransformations"],
                    coordinate_systems,
                    declared_version,
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
                axes_types=non_default_axes_types(cs_intrinsic.axes),
            )
            images.append(ngff_image)

        additionalTransformations = root_attrs.get("coordinateTransformations", None)
        if additionalTransformations is not None:
            additionalTransformations = cls._parse_transforms(
                additionalTransformations, coordinate_systems, declared_version
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
        cls,
        transforms: list[dict],
        coordinateSystems: list[CoordinateSystem],
        version: object | None = None,
    ) -> list[Transform]:
        """Parse transformation dictionaries, nested ones included.

        ``version`` selects the bounds a version-dependent rule applies, such
        as the ``mapAxis`` arity RFC-3 lifts at 0.9.dev1.
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
                _require_keys(transform, ("mapAxis",), "mapAxis")
                transformation = MapAxis(mapAxis=list(transform["mapAxis"]))
            elif transform["type"] == "projectAxis":
                transformation = ProjectAxis(
                    droppedInputs=(
                        list(transform["droppedInputs"])
                        if "droppedInputs" in transform
                        else None
                    ),
                    createdOutputs=(
                        list(transform["createdOutputs"])
                        if "createdOutputs" in transform
                        else None
                    ),
                )
            elif transform["type"] == "byDimension":
                transformation = ByDimension.from_dict(transform, coordinateSystems)
            elif transform["type"] == "bijection":
                transformation = Bijection.from_dict(transform, coordinateSystems)
            elif transform["type"] == "sequence":
                # TODO: Undo nested sequences on import?
                sub_transforms = cls._parse_transforms(
                    transform["transformations"], coordinateSystems, version
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
            validate_transform(transformation, coordinateSystems, version)
            parsed_transforms.append(transformation)

        return parsed_transforms

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.coordinateSystems[0].axes])


#: The public, version-scoped API of this module. CoordinateSystem here is the 0.6 model; ngff_zarr.v09.zarr_metadata carries a different one.
__all__ = [
    "Axis",
    "CoordinateSystem",
    "CoordinateSystemIdentifier",
    "BaseTransform",
    "Identity",
    "Scale",
    "Translation",
    "Rotation",
    "Affine",
    "Coordinates",
    "Displacements",
    "MapAxis",
    "ProjectAxis",
    "ByDimensionItem",
    "ByDimension",
    "Bijection",
    "TransformSequence",
    "Dataset",
    "Metadata",
]
