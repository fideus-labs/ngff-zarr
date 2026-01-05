# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

from ..v04.zarr_metadata import Axis, Omero, MethodMetadata
from .._supported_versions import NgffVersion
from .._zarr_types import StoreLike
from abc import ABC

if TYPE_CHECKING:
    from ..v05.zarr_metadata import Metadata as Metadata_v05
    from ..v04.zarr_metadata import Metadata as Metadata_v04

@dataclass
class CoordinateSystem:
    name: str
    axes: List[Axis]

@dataclass
class InputOutput:
    """
    InputOutput field used in Scene metadata.
     
    There, the input/output fields of transformations must be an object with
    'path' and 'coordinateSystem' fields.
    """
    path: str
    coordinateSystem: str

@dataclass(kw_only=True)
class BaseTransform(ABC):
    input: Optional[Union[CoordinateSystem, InputOutput, str]] = None
    output: Optional[Union[CoordinateSystem, InputOutput, str]] = None
    name: Optional[str] = None
    type: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "BaseTransform":
        return cls(**data)

@dataclass(kw_only=True)
class Identity(BaseTransform):
    type: str = "identity"
    name: Optional[str] = "identity"

@dataclass(kw_only=True)
class Scale(BaseTransform):
    scale: List[float]
    name: Optional[str] = "scale"
    type: str = "scale"

@dataclass(kw_only=True)
class Translation(BaseTransform):
    translation: List[float]
    name: Optional[str] = "translation"
    type: str = "translation"

@dataclass(kw_only=True)
class Rotation(BaseTransform):
    rotation: List[List[float]]
    path: Optional[str] = None
    name: Optional[str] = "rotation"
    type: str = "rotation"

@dataclass(kw_only=True)
class Affine(BaseTransform):
    affine: List[List[float]]
    path: Optional[str] = None
    name: Optional[str] = "affine"
    type: str = "affine"

Transform = Union[Identity, Scale, Translation, Rotation, Affine, "TransformSequence"]

@dataclass(kw_only=True)
class TransformSequence(BaseTransform):
    transformations: List[Transform]
    name: Optional[str] = "transformSequence"
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
    coordinateTransformations: Optional[List[Transform]] = None
    omero: Optional[Omero] = None
    name: str = "image"
    type: Optional[str] = None
    metadata: Optional[MethodMetadata] = None

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
        outputs = []
        for idx, ds in enumerate(self.datasets):
            path = ds.path

            coordinateTransformations = []
            transforms = ds.coordinateTransformations

            # set reasonable defaults
            spatial_dims = ("x", "y", "z")
            scale=[2.0**idx if d in spatial_dims else 1.0 for d in self.dimension_names]
            translation=[0.0 for d in self.dimension_names]


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
            assert len(set(outputs)) == 1
            output = outputs[0]

            scale = Scale_v05(scale=scale.scale)
            translation = Translation_v05(translation=translation.translation)
            coordinateTransformations = [scale, translation]

            cs = [cs for cs in self.coordinateSystems if cs.name == output][0]

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
                input=ds.path,
                name=f"scale{index}_to_intrinsic",
                output=coordinate_systems[0].name
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
        ) -> tuple["Metadata", list["NgffImage"]]:
        """Create Metadata instance from ome-zarr metadata dictionary."""
        import sys
        import dask.array
        from ..validate import validate as validate_ngff
        from ..parse_metadata import _parse_omero
        from ..rfc4_validation import validate_rfc4_orientation, has_rfc4_orientation_metadata
        from ..ngff_image import NgffImage

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

        omero = _parse_omero(root_attrs.get("omero", None))
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

        images = []
        datasets = []
        for index, dataset in enumerate(root_attrs["datasets"]):
            data = dask.array.from_zarr(store, component=dataset["path"])
            # Convert endianness to native if needed
            if (sys.byteorder == "little" and data.dtype.byteorder == ">") or (
                sys.byteorder == "big" and data.dtype.byteorder == "<"
            ):
                data = data.astype(data.dtype.newbyteorder())

            scale = {d: 1.0 for d in dims}
            translation = {d: 0.0 for d in dims}
            if "coordinateTransformations" in dataset:
                for transformation in dataset["coordinateTransformations"]:
                    if transformation["type"] == "sequence":
                        for seq_transform in transformation["transforms"]:
                            if seq_transform["type"] == "scale":
                                scale = seq_transform["scale"]
                            elif seq_transform["type"] == "translation":
                                translation = seq_transform["translation"]
                    
                    if "scale" in transformation:
                        scale = transformation["scale"]
                    elif "translation" in transformation:
                        translation = transformation["translation"]
                    
                    output_cs = [c for c in coordinate_systems if c.name == transformation.get("output")][0]
                    sequence = TransformSequence(
                        transformations=[Scale(scale), Translation(translation)],
                        input=transformation.get("input", dataset["path"]),
                        name=transformation.get("name", f"scale{index}_to_intrinsic"),
                        output=output_cs
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
                        raise ValueError(f"Unsupported transform type: {transform['type']} in dataset {dataset['path']}")
                
            cs_intrinsic = [cs for cs in coordinate_systems if cs.name == coordinateTransformations[0].output][0]

            datasets.append(
                Dataset(
                    path=dataset["path"],
                    coordinateTransformations=[sequence],
                )
            )

            ngff_image = NgffImage(
                data=data,
                dims=dims,
                scale=dict(zip(dims, scale)),
                translation=dict(zip(dims, translation)),
                name=root_attrs.get("name", "image"),
                axes_units=dict(zip(dims, [ax.unit for ax in cs_intrinsic.axes]))
                )
            images.append(ngff_image)

        coordinateTransformations = root_attrs.get("coordinateTransformations", None)
        if coordinateTransformations is not None:
            coordinateTransformations = cls._parse_transforms(coordinateTransformations, coordinate_systems)
        metadata = cls(
            coordinateSystems=coordinate_systems,
            datasets=datasets,
            name=root_attrs.get("name", "image"),
            omero=omero,
            coordinateTransformations=coordinateTransformations,
        )

        return metadata, images

    def _parse_transforms(self, transforms: List[dict], coordinateSystems: List[CoordinateSystem]) -> List[Transform]:
        """
        Parse a list of possibly nested transformation dictionaries into Transform instances.
        """
        parsed_transforms = []
        for transform in transforms:
            if transform["type"] == "identity":
                transformation = Identity()
            elif transform["type"] == "scale":
                transformation = Scale.from_dict(transform["scale"])
            elif transform["type"] == "translation":
                transformation = Translation.from_dict(transform["translation"])
            elif transform["type"] == "rotation":
                transformation = Rotation.from_dict(transform["rotation"])
            elif transform["type"] == "affine":
                transformation = Affine.from_dict(transform["affine"])
            elif transform["type"] == "sequence":
                sub_transforms = self._parse_transforms(transform["transforms"], coordinateSystems)
                transformation = TransformSequence(
                    transformations=sub_transforms
                )
            else:
                raise ValueError(f"Unsupported transform type: {transform['type']}")
            
            # resolve input/output to CoordinateSystem instances if matching name found
            # because coordinate system may not exist for multiscale transforms
            # where input is a dataset path
            coordinate_system_names = [cs.name for cs in coordinateSystems]
            input = transform.get("input", None)
            if input in coordinate_system_names:
                input = [cs.name for cs in coordinateSystems if cs.name == input][0]

            output = transform.get("output", None)
            if output in coordinate_system_names:
                output = [cs.name for cs in coordinateSystems if cs.name == output][0]
            transformation.input = input
            transformation.output = output
            parsed_transforms.append(transformation)
        
        return parsed_transforms

    @property
    def dimension_names(self) -> tuple:
        return tuple([ax.name for ax in self.coordinateSystems[0].axes])