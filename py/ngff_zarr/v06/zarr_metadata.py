# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import List, Literal, Union, TYPE_CHECKING
from dataclasses import dataclass, field

from ..v04.zarr_metadata import (
    AxesType as AxesTypeV04,
    MethodMetadata,
    Omero,
    SupportedDims,
    Units,
)
from ..rfc4 import AnatomicalOrientation
from .._supported_versions import NgffVersion
from .._zarr_types import StoreLike
from abc import ABC

if TYPE_CHECKING:
    from ..ngff_image import NgffImage
    from ..v05.zarr_metadata import Metadata as Metadata_v05
    from ..v04.zarr_metadata import Metadata as Metadata_v04


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
    axes: List[Axis]

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
class BaseTransform(ABC):
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
    scale: List[float]
    type: str = "scale"

@dataclass(kw_only=True)
class Translation(BaseTransform):
    translation: List[float]
    type: str = "translation"

@dataclass(kw_only=True)
class Rotation(BaseTransform):
    rotation: List[List[float]]
    path: str | None = None
    type: str = "rotation"

@dataclass(kw_only=True)
class Affine(BaseTransform):
    affine: List[List[float]]
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


Transform = Union[
    Identity,
    Scale,
    Translation,
    Rotation,
    Affine,
    Coordinates,
    Displacements,
    "TransformSequence",
]

@dataclass(kw_only=True)
class TransformSequence(BaseTransform):
    transformations: List[Transform]
    name: str | None = "transformSequence"
    type: str = "sequence"

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
    coordinateTransformations: List[Transform]

@dataclass
class Metadata:
    coordinateSystems: List[CoordinateSystem]
    datasets: List[Dataset]
    coordinateTransformations: List[Transform] | None = None
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

    def to_version(self, version: Union[str, NgffVersion]) -> Union["Metadata", "Metadata_v05", "Metadata_v04"]:
        if isinstance(version, str):
            # raise error for invalid version string
            version = NgffVersion(version)

        if version == NgffVersion.V04:
            return self._to_v05()._to_v04()
        elif version == NgffVersion.V05:
            return self._to_v05()
        elif version == NgffVersion.V06:
            return self
        else:
            raise ValueError(f"Unsupported version conversion: 0.6 -> {version}")
        
    @classmethod
    def from_version(cls, metadata: Union["Metadata", "Metadata_v05", "Metadata_v04"]) -> "Metadata":
        from ..v05.zarr_metadata import Metadata as Metadata_v05
        from ..v04.zarr_metadata import Metadata as Metadata_v04

        if isinstance(metadata, Metadata_v05):
            return cls._from_v05(metadata)
        elif isinstance(metadata, Metadata_v04):
            metadata_v05 = Metadata_v05._from_v04(metadata)
            return cls._from_v05(metadata_v05)
        else:
            raise ValueError(f"Unsupported metadata type ({type(metadata)}) for conversion to v0.6")
        
    def _to_v05(self) -> "Metadata_v05":
        from ..v05.zarr_metadata import Metadata as Metadata_v05
        from ..v05.zarr_metadata import Dataset as Dataset_v05
        from ..v04.zarr_metadata import Scale as Scale_v05
        from ..v04.zarr_metadata import Translation as Translation_v05

        datasets = []
        for idx, ds in enumerate(self.datasets):
            path = ds.path

            coordinateTransformations = []
            transforms = ds.coordinateTransformations

            # set reasonable defaults
            spatial_dims = ("x", "y", "z")
            scale = Scale(scale=[2.0**idx if d in spatial_dims else 1.0 for d in self.dimension_names])
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
                    translation = Translation(translation=[0.0 for d in self.dimension_names])

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

            cs = [cs for cs in self.coordinateSystems if cs.name == output.name][0]

            datasets.append(Dataset_v05(
                path=path,
                coordinateTransformations=coordinateTransformations,
            ))

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
            CoordinateSystem(
                name="intrinsic",
                axes=metadata_v05.axes
            )
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
                    Translation(translation=translation)],
                input=CoordinateSystemIdentifier(path=ds.path),
                name=f"scale{index}_to_intrinsic",
                output=CoordinateSystemIdentifier(name=coordinate_systems[0].name)
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
        from ..validate import validate as validate_ngff
        from ..parse_metadata import _parse_omero
        from ..rfc4_validation import validate_rfc4_orientation, has_rfc4_orientation_metadata
        from ..ngff_image import NgffImage

        # make sure root_attrs['ome]['multiscales'] exists
        if "ome" not in root_attrs or "multiscales" not in root_attrs["ome"]:
            raise ValueError("Invalid OME-Zarr metadata: missing 'ome' or 'multiscales' field.")

        if validate:
            validate_ngff(root_attrs, version=root_attrs['ome']['multiscales'][0].get("version", "0.6"))

            # RFC 4 validation for anatomical orientation
            if "axes" in root_attrs['ome']['multiscales'][0] and isinstance(root_attrs['ome']['multiscales'][0]["axes"], list):
                # Type cast each axis item to dict for validation
                axes_dicts = []
                for axis in root_attrs['ome']['multiscales'][0]["axes"]:
                    if isinstance(axis, dict):
                        axes_dicts.append(axis)
                if axes_dicts and has_rfc4_orientation_metadata(axes_dicts):
                    validate_rfc4_orientation(axes_dicts)

        omero = _parse_omero(root_attrs.get("ome", {}).get("omero"))
        root_attrs = root_attrs['ome']['multiscales'][0]
        
        coordinate_systems = []
        for cs in root_attrs.get("coordinateSystems", []):
            axes = [Axis(**axis) for axis in cs["axes"]]        
            
            coordinate_systems.append(
                CoordinateSystem(
                    name=cs["name"],
                    axes=axes
                )
            )

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
            coordinateTransformations: List[Transform] = [
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
                    coordinate_systems
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
                            "Unsupported transform type: " \
                            f"{transform.type} in dataset {dataset['path']}"
                            )
                
            cs_intrinsic = [
                cs for cs in coordinate_systems if cs.name == coordinateTransformations[0].output.name
                ][0]

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
                axes_units=dict(zip(dims, [ax.unit for ax in cs_intrinsic.axes]))
                )
            images.append(ngff_image)

        additionalTransformations = root_attrs.get("coordinateTransformations", None)
        if additionalTransformations is not None:
            additionalTransformations = cls._parse_transforms(additionalTransformations, coordinate_systems)

        metadata = cls(
            coordinateSystems=coordinate_systems,
            datasets=datasets,
            name=root_attrs.get("name", "image"),
            omero=omero,
            coordinateTransformations=additionalTransformations,
        )

        return metadata, images

    @classmethod
    def _parse_transforms(cls, transforms: List[dict], coordinateSystems: List[CoordinateSystem]) -> List[Transform]:
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
            elif transform["type"] == "sequence":
                # TODO: Undo nested sequences on import?
                sub_transforms = cls._parse_transforms(transform["transformations"], coordinateSystems)
                transformation = TransformSequence(
                    transformations=sub_transforms
                )
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
            parsed_transforms.append(transformation)
        
        return parsed_transforms

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.coordinateSystems[0].axes])