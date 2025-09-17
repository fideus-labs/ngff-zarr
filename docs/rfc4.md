<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# RFC-4: Anatomical Orientation Support

[RFC-4] adds support for anatomical orientation metadata to OME-NGFF axes,
enabling precise description of spatial axis directions in biological and
medical imaging data.

## Overview

Anatomical orientation refers to the specific arrangement and directional
alignment of anatomical structures within an imaging dataset. This is crucial
for ensuring accurate alignment and comparison of images to anatomical atlases,
facilitating consistent analysis and interpretation of biological data.

## Usage

### Programmatic Usage

#### Enabling RFC-4

To enable [RFC-4] support when writing OME-NGFF Zarr data, include `4` in the
`enabled_rfcs` parameter:

```python
import ngff_zarr

# Enable RFC-4 when converting to NGFF Zarr
ngff_zarr.to_ngff_zarr(
    store="output.ome.zarr",
    multiscales=multiscales,
    enabled_rfcs=[4]  # Enable RFC-4
)
```

### CLI Usage

RFC-4 can be enabled via the command line using the `--enable-rfc` flag:

```bash
# Convert a medical image with RFC-4 anatomical orientation enabled
ngff-zarr -i image.nii.gz -o output.ome.zarr --enable-rfc 4

# Enable multiple RFCs (when available)
ngff-zarr -i image.nii.gz -o output.ome.zarr --enable-rfc 4 --enable-rfc 5

# Without RFC-4 (default behavior - orientation metadata filtered out)
ngff-zarr -i image.nii.gz -o output.ome.zarr
```

### ITK Integration

When converting ITK images (including from NRRD, NIfTI, or DICOM formats),
anatomical orientation is automatically added based on ITK's LPS coordinate
system (see the sections `Anatomical Orientation Values` and
`ITK LPS Coordinate System` below for an explanation):

```python
import ngff_zarr
from itk import imread

# Read a medical image (NRRD, NIfTI, DICOM, etc.)
image = imread("image.nrrd")

# Convert to NGFF image with anatomical orientation
ngff_image = ngff_zarr.itk_image_to_ngff_image(
    image,
    add_anatomical_orientation=True
)

# Convert to multiscales and write to Zarr with RFC-4 enabled
multiscales = ngff_zarr.to_multiscales(ngff_image)
ngff_zarr.to_ngff_zarr(
    store="output.ome.zarr",
    multiscales=multiscales,
    enabled_rfcs=[4]
)
```

### ITK-Wasm Integration

Similar support is available for ITK-Wasm:

```python
import ngff_zarr
from itkwasm_image_io import imread

# Read with ITK-Wasm
image = imread("image.nii.gz")

# Convert with anatomical orientation
ngff_image = ngff_zarr.itk_image_to_ngff_image(
    image,
    add_anatomical_orientation=True
)
```

## Anatomical Orientation Values

[RFC-4] supports the following anatomical orientation values:

- `left-to-right`: Describes the directional orientation from the left side to
  the right lateral side of an anatomical structure or body.
- `right-to-left`: Describes the directional orientation from the right side to
  the left lateral side of an anatomical structure or body.
- `anterior-to-posterior`: Describes the directional orientation from the front
  (anterior) to the back (posterior) of an anatomical structure or body.
- `posterior-to-anterior`: Describes the directional orientation from the back
  (posterior) to the front (anterior) of an anatomical structure or body.
- `inferior-to-superior`: Describes the directional orientation from below
  (inferior) to above (superior) in an anatomical structure or body.
- `superior-to-inferior`: Describes the directional orientation from above
  (superior) to below (inferior) in an anatomical structure or body.
- `dorsal-to-ventral`: Describes the directional orientation from the top/upper
  (dorsal) to the belly/lower (ventral) in an anatomical structure or body.
- `ventral-to-dorsal`: Describes the directional orientation from the
  belly/lower (ventral) to the top/upper (dorsal) in an anatomical structure or
  body.
- `dorsal-to-palmar`: Describes the directional orientation from the top/upper
  (dorsal) to the palm of the hand (palmar) in a body.
- `palmar-to-dorsal`: Describes the directional orientation from the palm of the
  hand (palmar) to the top/upper (dorsal) in a body.
- `dorsal-to-plantar`: Describes the directional orientation from the top/upper
  (dorsal) to the sole of the foot (plantar) in a body.
- `plantar-to-dorsal`: Describes the directional orientation from the sole of
  the foot (plantar) to the top/upper (dorsal) in a body.
- `rostral-to-caudal`: Describes the directional orientation from the nasal
  (rostral) to the tail (caudal) end of an anatomical structure, typically used
  in reference to the central nervous system.
- `caudal-to-rostral`: Describes the directional orientation from the tail
  (caudal) to the nasal (rostral) end of an anatomical structure, typically used
  in reference to the central nervous system.
- `cranial-to-caudal`: Describes the directional orientation from the head
  (cranial) to the tail (caudal) end of an anatomical structure or body.
- `caudal-to-cranial`: Describes the directional orientation from the tail
  (caudal) to the head (cranial) end of an anatomical structure or body.
- `proximal-to-distal`: Describes the directional orientation from the center of
  the body to the periphery of an anatomical structure or limb.
- `distal-to-proximal`: Describes the directional orientation from the periphery
  of an anatomical structure or limb to the center of the body.

## ITK LPS Coordinate System

ITK uses the LPS (Left-Posterior-Superior) coordinate system by default. In LPS,
the axis directions indicate the direction of increasing coordinate values:

- **L (Left)**: The X-axis increases from right-to-left
- **P (Posterior)**: The Y-axis increases from anterior-to-posterior
- **S (Superior)**: The Z-axis increases from inferior-to-superior

When converting ITK images, the following mappings are applied:

- **X axis**: `right-to-left` (L = Left direction)
- **Y axis**: `anterior-to-posterior` (P = Posterior direction)
- **Z axis**: `inferior-to-superior` (S = Superior direction)

## Convenience Coordinate Systems

For common use cases, we provide pre-defined orientation dictionaries:

### LPS Coordinate System

The `LPS` constant provides orientations for the DICOM and ITK standard
coordinate system. LPS stands for Left-Posterior-Superior, where:

- **L (Left)**: X-axis increases from right-to-left
- **P (Posterior)**: Y-axis increases from anterior-to-posterior
- **S (Superior)**: Z-axis increases from inferior-to-superior

```python
import ngff_zarr as nz
import dask.array as da

# Create image data
data = da.zeros((100, 100, 100))

# Using the LPS convenience constant
ngff_image = nz.NgffImage(
    data=data,
    dims=("z", "y", "x"),
    scale={"x": 1.0, "y": 1.0, "z": 1.0},
    translation={"x": 0.0, "y": 0.0, "z": 0.0},
    axes_orientations=nz.LPS
)

# Convert to multiscales and write with RFC-4 enabled
multiscales = nz.to_multiscales(ngff_image)
nz.to_ngff_zarr(
    store="output.ome.zarr",
    multiscales=multiscales,
    enabled_rfcs=[4]
)
```

The `LPS` constant is equivalent to:

```python
{
    "x": AnatomicalOrientation(type="anatomical", value="right-to-left"),
    "y": AnatomicalOrientation(type="anatomical", value="anterior-to-posterior"),
    "z": AnatomicalOrientation(type="anatomical", value="inferior-to-superior")
}
```

### RAS Coordinate System

The `RAS` constant provides orientations for the neuroimaging coordinate system
used in NIfTI. RAS stands for Right-Anterior-Superior, where:

- **R (Right)**: X-axis increases from left-to-right
- **A (Anterior)**: Y-axis increases from posterior-to-anterior
- **S (Superior)**: Z-axis increases from inferior-to-superior

```python
import ngff_zarr as nz
import dask.array as da

# Create neuroimaging data
data = da.zeros((256, 256, 256))

# Using the RAS convenience constant
ngff_image = nz.NgffImage(
    data=data,
    dims=("z", "y", "x"),
    scale={"x": 1.0, "y": 1.0, "z": 1.0},
    translation={"x": 0.0, "y": 0.0, "z": 0.0},
    axes_orientations=ngff_zarr.RAS
)

# Convert to multiscales and write with RFC-4 enabled
multiscales = nz.to_multiscales(ngff_image)
nz.to_ngff_zarr(
    store="brain_data.ome.zarr",
    multiscales=multiscales,
    enabled_rfcs=[4]
)
```

The `RAS` constant is equivalent to:

```python
{
    "x": AnatomicalOrientation(type="anatomical", value="left-to-right"),
    "y": AnatomicalOrientation(type="anatomical", value="posterior-to-anterior"),
    "z": AnatomicalOrientation(type="anatomical", value="inferior-to-superior")
}
```

## Manual Orientation Specification

You can also manually specify orientations for custom coordinate systems:

```python
from ngff_zarr import (
    AnatomicalOrientation,
    AnatomicalOrientationValues,
    NgffImage
)
import dask.array as da

# Create orientation objects
x_orientation = AnatomicalOrientation(
    value=AnatomicalOrientationValues.right_to_left
)
y_orientation = AnatomicalOrientation(
    value=AnatomicalOrientationValues.anterior_to_posterior
)
z_orientation = AnatomicalOrientation(
    value=AnatomicalOrientationValues.inferior_to_superior
)

# Create NGFF image with orientations
data = da.zeros((100, 100, 100))
ngff_image = NgffImage(
    data=data,
    dims=("z", "y", "x"),
    scale={"x": 1.0, "y": 1.0, "z": 1.0},
    translation={"x": 0.0, "y": 0.0, "z": 0.0},
    axes_orientations={
        "x": x_orientation,
        "y": y_orientation,
        "z": z_orientation
    }
)
```

## Metadata Format

When RFC-4 is enabled, spatial axes in the OME-NGFF metadata include an
`orientation` field:

```json
{
  "axes": [
    {
      "name": "z",
      "type": "space",
      "unit": "micrometer",
      "orientation": {
        "type": "anatomical",
        "value": "inferior-to-superior"
      }
    },
    {
      "name": "y",
      "type": "space",
      "unit": "micrometer",
      "orientation": {
        "type": "anatomical",
        "value": "anterior-to-posterior"
      }
    },
    {
      "name": "x",
      "type": "space",
      "unit": "micrometer",
      "orientation": {
        "type": "anatomical",
        "value": "right-to-left"
      }
    }
  ]
}
```

## RFC-4 Functions

### Core Functions

- `itk_lps_to_anatomical_orientation(axis_name)`: Map ITK LPS axes to
  orientations
- `add_anatomical_orientation_to_axis(axis_dict, orientation)`: Add orientation
  to axis
- `remove_anatomical_orientation_from_axis(axis_dict)`: Remove orientation from
  axis

### Data Classes

- `AnatomicalOrientation`: Orientation specification with type and value
- `AnatomicalOrientationValues`: Enum of supported orientation values

### Convenience Constants

- `LPS`: Pre-defined orientations for ITK coordinate system:
  - L (Left): X-axis from right-to-left
  - P (Posterior): Y-axis from anterior-to-posterior
  - S (Superior): Z-axis from inferior-to-superior
- `RAS`: Pre-defined orientations for neuroimaging coordinate system:
  - R (Right): X-axis from left-to-right
  - A (Anterior): Y-axis from posterior-to-anterior
  - S (Superior): Z-axis from inferior-to-superior

## Compatibility

When RFC-4 is not enabled (the default), orientation metadata is automatically
filtered out during Zarr writing to maintain compatibility with standard
OME-NGFF consumers.

## Best Practices

1. **Enable RFC-4** when working with medical/biological images where
   orientation matters
2. **Use convenience constants** (`LPS`, `RAS`) for standard coordinate systems
3. **Use ITK integration** for automatic LPS-based orientation when converting
   from medical image formats
4. **Document orientation assumptions** in your analysis pipelines when working
   with oriented data
5. **Validate orientations** match your expectations, especially when combining
   data from different sources

[RFC-4]: https://ngff.openmicroscopy.org/rfc/4/index.html
