# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
import functools
import logging
import re
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Literal, Union

from .._store_types import StoreLike
from .._supported_versions import NgffVersion

# Import RFC 4 support
from ..rfc4 import AnatomicalOrientation

if TYPE_CHECKING:
    from ..ngff_image import NgffImage
    from ..v05.zarr_metadata import Metadata as Metadata_v05
    from ..v06.zarr_metadata import Metadata as Metadata_v06
    from ..v09.zarr_metadata import Metadata as Metadata_v09

logger = logging.getLogger(__name__)

SupportedDims = Union[
    Literal["c"], Literal["x"], Literal["y"], Literal["z"], Literal["t"]
]

SpatialDims = Union[Literal["x"], Literal["y"], Literal["z"]]
AxesType = Union[Literal["time"], Literal["space"], Literal["channel"]]
SpaceUnits = Union[
    Literal["angstrom"],
    Literal["attometer"],
    Literal["centimeter"],
    Literal["decimeter"],
    Literal["exameter"],
    Literal["femtometer"],
    Literal["foot"],
    Literal["gigameter"],
    Literal["hectometer"],
    Literal["inch"],
    Literal["kilometer"],
    Literal["megameter"],
    Literal["meter"],
    Literal["micrometer"],
    Literal["mile"],
    Literal["millimeter"],
    Literal["nanometer"],
    Literal["parsec"],
    Literal["petameter"],
    Literal["picometer"],
    Literal["terameter"],
    Literal["yard"],
    Literal["yoctometer"],
    Literal["yottameter"],
    Literal["zeptometer"],
    Literal["zettameter"],
]
TimeUnits = Union[
    Literal["attosecond"],
    Literal["centisecond"],
    Literal["day"],
    Literal["decisecond"],
    Literal["exasecond"],
    Literal["femtosecond"],
    Literal["gigasecond"],
    Literal["hectosecond"],
    Literal["hour"],
    Literal["kilosecond"],
    Literal["megasecond"],
    Literal["microsecond"],
    Literal["millisecond"],
    Literal["minute"],
    Literal["nanosecond"],
    Literal["petasecond"],
    Literal["picosecond"],
    Literal["second"],
    Literal["terasecond"],
    Literal["yoctosecond"],
    Literal["yottasecond"],
    Literal["zeptosecond"],
    Literal["zettasecond"],
]
Units = Union[SpaceUnits, TimeUnits]

#: Axis unit as RFC-3 allows it: the controlled vocabulary, or any other
#: string. Every published axes schema declares ``"unit": {"type": "string"}``
#: with no ``enum``, and an axis of an arbitrary type has no unit in the closed
#: space/time vocabulary. The union keeps the vocabulary so editors still
#: complete it. Mirrors the TypeScript port's ``AxisUnit``.
AxisUnit = Union[Units, str]

supported_dims = ["x", "y", "z", "c", "t"]

space_units = [
    "angstrom",
    "attometer",
    "centimeter",
    "decimeter",
    "exameter",
    "femtometer",
    "foot",
    "gigameter",
    "hectometer",
    "inch",
    "kilometer",
    "megameter",
    "meter",
    "micrometer",
    "mile",
    "millimeter",
    "nanometer",
    "parsec",
    "petameter",
    "picometer",
    "terameter",
    "yard",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
]

time_units = [
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
]


def is_dimension_supported(dim: str) -> bool:
    """Helper for string validation"""
    return dim in supported_dims


def is_unit_supported(unit: str) -> bool:
    """Helper for string validation"""
    return (unit in time_units) or (unit in space_units)


@functools.lru_cache(maxsize=1)
def _get_axis_fields() -> set[str]:
    """Get the set of valid field names for the Axis dataclass.

    Cached to avoid repeated introspection.
    """
    return {f.name for f in fields(Axis)}


def _filter_axis_dict(axis_dict: dict) -> dict:
    """Filter an axis dictionary to only include valid Axis fields.

    Logs a warning if unknown fields are encountered.

    Raises:
        ValueError: If the required field 'name' is missing from the axis
            dictionary.
    """
    # `name` is the only required axis field: the v0.4 schema accepts an axis
    # that declares no `type`, so a missing one is read as undefined.
    if "name" not in axis_dict:
        raise ValueError(
            f"Axis dictionary is missing required field 'name': {axis_dict}"
        )

    axis_fields = _get_axis_fields()
    unknown_fields = set(axis_dict.keys()) - axis_fields
    if unknown_fields:
        axis_name = axis_dict.get("name", "unknown")
        logger.warning(
            f"Ignoring unknown fields {unknown_fields} in axis '{axis_name}'. "
            f"These fields are not part of the OME-NGFF v0.4 specification."
        )
    filtered = {k: v for k, v in axis_dict.items() if k in axis_fields}
    filtered.setdefault("type", None)
    return filtered


@dataclass
class Axis:
    name: SupportedDims
    type: AxesType | None
    unit: Units | None = None
    orientation: AnatomicalOrientation | None = None


@dataclass
class Identity:
    type: str = "identity"


@dataclass
class Scale:
    scale: list[float]
    type: str = "scale"


@dataclass
class Translation:
    translation: list[float]
    type: str = "translation"


Transform = Union[Scale, Translation]


@dataclass
class Dataset:
    path: str
    coordinateTransformations: list[Transform]


@dataclass
class OmeroWindow:
    #: Every bound is optional, as in the OME-Zarr schema, which requires none
    #: of them. A window that carries only some of its bounds is read with the
    #: rest left unset rather than dropped.
    min: float | None = None
    max: float | None = None
    start: float | None = None
    end: float | None = None


@dataclass
class OmeroChannel:
    #: ``color`` and ``window`` are optional so that a channel is read as
    #: written. The list is positional, one entry per index of the ``c`` axis,
    #: so a channel that cannot be fully built must still occupy its place.
    color: str | None = None
    window: OmeroWindow | None = None
    label: str | None = None
    active: bool | None = None

    def validate_color(self):
        """Raise when ``color`` is present and is not six hexadecimal digits.

        A channel with no ``color`` passes: whether the field may be absent is
        the schema's decision, not this predicate's.
        """
        if self.color is None:
            return
        if not re.fullmatch(r"[0-9A-Fa-f]{6}", self.color):
            raise ValueError(f"Invalid color '{self.color}'. Must be 6 hex digits.")


@dataclass
class Omero:
    channels: list[OmeroChannel]
    version: str | None = None


@dataclass
class MethodMetadata:
    description: str
    method: str
    version: str


@dataclass
class PlateAcquisition:
    id: int
    name: str | None = None
    maximumfieldcount: int | None = None
    description: str | None = None
    starttime: int | None = None
    endtime: int | None = None


@dataclass
class PlateColumn:
    name: str


@dataclass
class PlateRow:
    name: str


@dataclass
class PlateWell:
    path: str
    rowIndex: int
    columnIndex: int


@dataclass
class Plate:
    columns: list[PlateColumn]
    rows: list[PlateRow]
    wells: list[PlateWell]
    version: str = "0.4"
    acquisitions: list[PlateAcquisition] | None = None
    field_count: int | None = None
    name: str | None = None


@dataclass
class WellImage:
    path: str
    acquisition: int | None = None


@dataclass
class Well:
    images: list[WellImage]
    version: str = "0.4"


# Recognized keys of a single ``multiscales[]`` entry. Any other key present in
# an entry is a leak -- a group-level ``ome`` / ``multiscales`` wrapper, a
# ``zarr_format`` byte that belongs on the enclosing Zarr v3 group, and so on --
# and is routed to :attr:`Metadata.extra` for the v0.5 namespacing rules to
# inspect. ``omero`` is absent on purpose: it is a sibling of ``multiscales``,
# parsed separately, never a field of the entry. ``@type`` *is* recognized: the
# writer (:func:`ngff_zarr.to_ngff_zarr`) stamps the JSON-LD ``@type``
# (``"ngff:Image"``) onto every entry, so it is a legitimate library-written key,
# not a leak.
_MULTISCALE_ENTRY_FIELDS = frozenset(
    {
        "@type",
        "version",
        "name",
        "axes",
        "datasets",
        "coordinateTransformations",
        "type",
        "metadata",
    }
)


@dataclass
class Metadata:
    axes: list[Axis]
    datasets: list[Dataset]
    coordinateTransformations: list[Transform] | None
    omero: Omero | None = None
    name: str = "image"
    version: str = "0.4"
    type: str | None = None
    metadata: MethodMetadata | None = None
    #: Unrecognized keys that leaked into the ``multiscales[]`` entry (a
    #: malformed or double-namespaced v0.5 document). Captured verbatim so the
    #: v0.5 namespacing rules (``zarr-format``, ``ome-namespace``) can flag them;
    #: mirrors the Rust port's ``#[serde(flatten)]`` ``extra`` passthrough. It is
    #: a validation aid only and is never serialized back to the store.
    extra: dict = field(default_factory=dict)

    def to_version(
        self, version: Union[str, NgffVersion]
    ) -> Union["Metadata", "Metadata_v05", "Metadata_v06", "Metadata_v09"]:
        if isinstance(version, str):
            # raise error for invalid version string
            version = NgffVersion(version)

        if version == NgffVersion.V04:
            return self
        if version == NgffVersion.V05:
            return self._to_v05()
        if version == NgffVersion.V06:
            return self._to_v05()._to_v06()
        if version == NgffVersion.V09dev1:
            from ..v09.zarr_metadata import Metadata as Metadata_v09

            return Metadata_v09.from_version(self)
        raise ValueError(f"Unsupported version conversion: 0.4 -> {version}")

    @classmethod
    def from_version(
        cls,
        metadata: Union["Metadata", "Metadata_v05", "Metadata_v06", "Metadata_v09"],
    ) -> "Metadata":
        from ..v05.zarr_metadata import Metadata as Metadata_v05
        from ..v06.zarr_metadata import Metadata as Metadata_v06
        from ..v09.zarr_metadata import Metadata as Metadata_v09

        if isinstance(metadata, Metadata_v05):
            return cls._from_v05(metadata)
        if isinstance(metadata, Metadata_v09):
            return cls._from_v05(Metadata_v05._from_v06(metadata._to_v06()))
        if isinstance(metadata, Metadata_v06):
            return cls._from_v05(Metadata_v05._from_v06(metadata))
        raise ValueError(f"Unsupported metadata type: {type(metadata)}")

    def _to_v05(self) -> "Metadata_v05":
        from ..v05.zarr_metadata import Metadata as Metadata_v05

        metadata = Metadata_v05(
            axes=self.axes,
            datasets=self.datasets,
            coordinateTransformations=self.coordinateTransformations,
            name=self.name,
            metadata=self.metadata,
            type=self.type,
            omero=self.omero,
        )
        return metadata

    @classmethod
    def _from_v05(cls, metadata_v05: "Metadata_v05") -> "Metadata":

        metadata = cls(
            axes=metadata_v05.axes,
            datasets=metadata_v05.datasets,
            coordinateTransformations=metadata_v05.coordinateTransformations,
            name=metadata_v05.name,
            metadata=metadata_v05.metadata,
            type=metadata_v05.type,
            omero=metadata_v05.omero,
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

        Parameters
        ----------
        root_attrs : dict
            The root attributes dictionary
        store : StoreLike
            The zarr store
        validate : bool
            Whether to validate the metadata
        subpath : str, optional
            Sub-path within the store for HCS well/image access (e.g., 'A/1/0')
        """
        import posixpath
        import sys

        import packaging.version

        from .._zarrista_utils import open_lazy_array
        from ..ngff_image import NgffImage
        from ..parse_metadata import _parse_omero, _raw_axes
        from ..rfc4_validation import (
            has_any_rfc4_orientation,
            validate_rfc4_orientation,
        )
        from ..validate import validate as validate_ngff

        # Validate structure before any processing to avoid cryptic KeyError
        if "multiscales" not in root_attrs or not root_attrs["multiscales"]:
            raise ValueError(
                "Invalid OME-Zarr v0.4 format: 'multiscales' key is missing or empty in root attributes. "
                "This may be a v0.5 file (which has 'ome' key instead of 'multiscales'). "
                f"Available keys: {list(root_attrs.keys())}"
            )

        if validate:
            # OME-Zarr v0.5 hoists the spec ``version`` to the group-level
            # ``ome`` namespace and validates the ``ome``-wrapped instance
            # against the v0.5 image schema; v0.4 keeps ``version`` on each
            # multiscale entry and validates the bare root. Prefer the
            # group-level version (the v0.5 read path floors it to ``"0.5"``) so
            # a v0.5 store selects the v0.5 schema, falling back to the per-entry
            # version for v0.4 and the legacy v0.1-0.3 layouts.
            schema_version = str(
                root_attrs.get("version")
                or root_attrs["multiscales"][0].get("version", "0.4")
            )
            if packaging.version.parse(schema_version) >= packaging.version.parse(
                "0.5"
            ):
                # ``root_attrs`` is the unwrapped ``ome`` content (multiscales +
                # hoisted version + omero); re-wrap it so the instance matches
                # the v0.5 image schema's required ``ome`` namespace.
                validate_ngff({"ome": root_attrs}, version=schema_version)
            else:
                validate_ngff(root_attrs, version=schema_version)

            # RFC 4 validation for anatomical orientation. The axes are read
            # through the shared helper, which knows where each version keeps
            # them, and a non-dict axis entry is left to the schema check.
            axes_dicts = [
                axis
                for axis in _raw_axes(root_attrs["multiscales"][0])
                if isinstance(axis, dict)
            ]
            if axes_dicts and has_any_rfc4_orientation(axes_dicts):
                validate_rfc4_orientation(axes_dicts)

        omero = _parse_omero(root_attrs.get("omero"))
        # OME-Zarr v0.5 hoists the spec ``version`` to the group-level ``ome``
        # namespace; v0.4 carries it on each multiscale entry. Capture the
        # group-level value (if any) before descending into the entry so the
        # parsed Metadata records the true spec version -- which the v0.5
        # structural rules (zarr-format, ome-namespace) gate on. v0.4 has no
        # group-level version, so this falls back to the entry's below.
        group_version = root_attrs.get("version")
        root_attrs = root_attrs["multiscales"][0]

        # This handles backwards compatibility for version<=0.3
        if "axes" not in root_attrs:
            dims = tuple(reversed(supported_dims))
            axes = [
                Axis(name="t", type="time"),
                Axis(name="c", type="channel"),
                Axis(name="z", type="space"),
                Axis(name="y", type="space"),
                Axis(name="x", type="space"),
            ]
            units = dict.fromkeys(dims)
        else:
            axes_list = root_attrs["axes"]
            if not axes_list:
                raise ValueError(
                    "Multiscale metadata contains empty axes list. "
                    "At least one axis must be defined."
                )

            # Determine if we have v0.4+ (dict-based axes) or v0.3 (string-based axes)
            # by checking if the first axis is a dict
            first_axis_is_dict = isinstance(axes_list[0], dict)

            if first_axis_is_dict:
                # v0.4+ format with dict-based axes
                dims = tuple(a["name"] if "name" in a else a for a in axes_list)
                axes = [Axis(**_filter_axis_dict(axis)) for axis in axes_list]
            else:
                # v0.3 format with string-based axes
                dims = tuple(axes_list)
                type_dict = {
                    "t": "time",
                    "c": "channel",
                    "z": "space",
                    "y": "space",
                    "x": "space",
                }
                axes = [Axis(name=axis, type=type_dict[axis]) for axis in axes_list]

            units = dict.fromkeys(dims)
            for axis in axes_list:
                # Only process unit information for dict-style axes that have both
                # a name and a unit (v0.4+). For v0.3 string axes, this loop is a no-op.
                if isinstance(axis, dict):
                    name = axis.get("name")
                    unit = axis.get("unit")
                    if name is not None and unit is not None:
                        units[name] = unit

        images = []
        datasets = []
        for dataset in root_attrs["datasets"]:
            # If we have a subpath, prepend it to the dataset path using posixpath.join
            # to guarantee a single '/' separator (handles leading/trailing slashes safely)
            dataset_path = dataset["path"]
            if subpath:
                dataset_path = posixpath.join(subpath, dataset_path)

            data = open_lazy_array(store, dataset_path)
            # Convert endianness to native if needed
            if (sys.byteorder == "little" and data.dtype.byteorder == ">") or (
                sys.byteorder == "big" and data.dtype.byteorder == "<"
            ):
                data = data.astype(data.dtype.newbyteorder())

            scale = dict.fromkeys(dims, 1.0)
            translation = dict.fromkeys(dims, 0.0)
            coordinateTransformations = []
            if "coordinateTransformations" in dataset:
                for transformation in dataset["coordinateTransformations"]:
                    if "scale" in transformation:
                        scale = transformation["scale"]
                        scale = dict(zip(dims, scale))
                        coordinateTransformations.append(Scale(transformation["scale"]))
                    elif "translation" in transformation:
                        translation = transformation["translation"]
                        translation = dict(zip(dims, translation))
                        coordinateTransformations.append(
                            Translation(transformation["translation"])
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
                scale=scale,
                translation=translation,
                name=root_attrs.get("name", "image"),
                axes_units=units,
            )
            images.append(ngff_image)

        # Keys that a malformed or double-namespaced document leaked into the
        # multiscale entry -- e.g. a group-level ``ome`` / ``multiscales``
        # wrapper, or a ``zarr_format`` byte that belongs on the enclosing
        # Zarr v3 group, never on the entry. Captured verbatim so the v0.5
        # namespacing rules can flag them.
        extra = {
            key: value
            for key, value in root_attrs.items()
            if key not in _MULTISCALE_ENTRY_FIELDS
        }

        metadata = cls(
            axes=axes,
            datasets=datasets,
            name=root_attrs.get("name", "image"),
            version=group_version or root_attrs.get("version", "0.4"),
            omero=omero,
            coordinateTransformations=root_attrs.get("coordinateTransformations", None),
            extra=extra,
        )

        if validate:
            # Strict structural validation of the parsed Metadata, layered after
            # the schema pass and RFC 4 dict check above. The structural rules
            # encode the v0.4+ metadata model (typed axes and per-dataset
            # coordinate transformations); the legacy v0.1-0.3 layouts predate
            # it, so the rules are scoped to v0.4 and newer. Imported lazily so
            # the default validate=False read path incurs no extra import cost.
            from ..structural_validation import (
                ValidateOptions,
                ValidationLevel,
                validate_structural,
            )

            spec_version = packaging.version.parse(str(metadata.version))
            if spec_version >= packaging.version.parse("0.4"):
                validate_structural(
                    metadata, ValidateOptions(level=ValidationLevel.STRICT)
                )

        return metadata, images

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.axes])
