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

#### Writing Anatomical Orientation

Anatomical orientation is written to the output automatically whenever it is
present -- there is no separate opt-in. Provide orientation either by setting
`axes_orientations` on the `NgffImage`, or by passing the `orientation` keyword
to `to_multiscales()` (a preset name such as `"LPS"` or `"RAS"`, or a per-axis
mapping of `AnatomicalOrientation`):

```python
import ngff_zarr

# Provide orientation via the `orientation` keyword (a preset name here)
multiscales = ngff_zarr.to_multiscales(ngff_image, orientation="LPS")

# Orientation is written automatically -- no enable flag required
ngff_zarr.to_ngff_zarr(
    store="output.ome.zarr",
    multiscales=multiscales,
)
```

Images that carry no anatomical orientation produce standard OME-NGFF metadata
with no `orientation` field.

### CLI Usage

Use the `--orientation` flag to attach anatomical orientation to the spatial
axes. It accepts a preset coordinate system (`LPS` or `RAS`) or ordered
dim/value pairs. Orientation is written to the output automatically when
present:

```bash
# Convert a medical image using the LPS coordinate system
ngff-zarr -i image.nii.gz -o output.ome.zarr --orientation LPS

# Or specify orientation per axis as ordered dim/value pairs
ngff-zarr -i image.nii.gz -o output.ome.zarr \
    --orientation x right-to-left y anterior-to-posterior z inferior-to-superior

# Inputs that already carry orientation (e.g. NIfTI, NRRD, DICOM via ITK)
# write it automatically with no extra flags
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

# Convert to multiscales and write to Zarr -- because the image carries
# orientation, it is written automatically
multiscales = ngff_zarr.to_multiscales(ngff_image)
ngff_zarr.to_ngff_zarr(
    store="output.ome.zarr",
    multiscales=multiscales,
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

The following values describe subject-local orientation within layered and
polarized tissues (e.g. biopsies, histology slides), independent of the
specimen's position relative to the whole organism:

- `superficial-to-deep`: Describes the directional orientation from the outer
  surface (superficial) to the inner depth (deep) of a layered tissue such as
  skin, gut, or cortex.
- `deep-to-superficial`: Describes the directional orientation from the inner
  depth (deep) to the outer surface (superficial) of a layered tissue such as
  skin, gut, or cortex.
- `apical-to-basal`: Describes the directional orientation from the apical
  surface (apical) to the basal surface (basal) of an epithelial layer or
  polarized cell structure.
- `basal-to-apical`: Describes the directional orientation from the basal
  surface (basal) to the apical surface (apical) of an epithelial layer or
  polarized cell structure.
- `apex-to-base`: Describes the directional orientation from the tip (apex) to
  the broad base (base) of an organ such as the heart or lung.
- `base-to-apex`: Describes the directional orientation from the broad base
  (base) to the tip (apex) of an organ such as the heart or lung.

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

# Convert to multiscales and write -- orientation is written automatically
multiscales = nz.to_multiscales(ngff_image)
nz.to_ngff_zarr(
    store="output.ome.zarr",
    multiscales=multiscales,
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
    axes_orientations=nz.RAS
)

# Convert to multiscales and write -- orientation is written automatically
multiscales = nz.to_multiscales(ngff_image)
nz.to_ngff_zarr(
    store="brain_data.ome.zarr",
    multiscales=multiscales,
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

When anatomical orientation is present, the spatial axes in the OME-NGFF
metadata include an `orientation` field:

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

## Validation and Conformance

### What RFC-4 requires

Beyond the vocabulary above, [RFC-4] constrains where and how orientation is
declared:

- `orientation` is used **only** on spatial axes (`type: "space"`), never on a
  time or channel axis.
- An orientation object carries a `type` -- `"anatomical"` is the only defined
  value -- and a `value` drawn from the 24-term vocabulary.
- A set of axes carries **one direction of each antonym pair**: an image cannot
  declare both `left-to-right` and `right-to-left`.
- Orientation is optional per axis. An absent field and an explicit `null` mean
  the same thing, but writers SHOULD omit the field rather than serialize
  `null`.

Three of these are enforced by `validate_structural()` as the rules
`axis-orientation-anatomical-type`, `axis-orientation-on-non-space` and
`axis-orientation-unique-axis`, raising a `ValidationError` that names the
offending rule and axis:

```python
from ngff_zarr import from_ngff_zarr, validate_structural, ValidationError

multiscales = from_ngff_zarr("image.ome.zarr")
try:
    # The structural rules read the flat v0.4 axis list, so convert first when
    # the store was read into a newer data model.
    validate_structural(multiscales.metadata.to_version("0.4"))
except ValidationError as exc:
    print(exc.rule.value, exc.message)
    # axis-orientation-unique-axis Axes 'z' and 'y' describe the same
    # anatomical axis; RFC 4 allows only one direction per axis.
```

### The `conformance` subcommand

`ngff-zarr conformance` prints a single JSON verdict for one store, in the
canonical format a tool-agnostic conformance driver diffs against its manifest:

```bash
ngff-zarr conformance brain_icbm_ras.ome.zarr
```

```json
{
  "input": "brain_icbm_ras.ome.zarr",
  "format": "ome-zarr",
  "rfc4_valid": true,
  "axes": {
    "z": "inferior-to-superior",
    "y": "anterior-to-posterior",
    "x": "right-to-left"
  },
  "violations": [],
  "warnings": []
}
```

`axes` maps each axis that declares an orientation value to that value;
unannotated axes are omitted. `rfc4_valid` is false whenever `violations` is
non-empty. The axes are read raw, so malformed metadata is still classified
rather than rejected before it can be reported, and a code repeats once per
offending axis:

| Code | Condition |
|---|---|
| `orientation-on-non-space` | orientation declared on a non-spatial axis |
| `bad-type` | `orientation.type` is not `"anatomical"` |
| `missing-value` | orientation object without a `value` |
| `bad-value` | `value` outside the 24-term vocabulary |
| `duplicate-anatomical-axis` | two axes describing the same anatomical axis |
| `null-orientation` *(warning)* | `orientation: null` -- writers SHOULD omit it |
| `input-unreadable` | the input could not be read as JSON metadata |
| `input-not-ome-zarr` | no `multiscales[0].axes` was found |

An unreadable or non-OME-Zarr input reports the read-failure code as a
violation, never `"rfc4_valid": true` with empty axes.

## RFC-4 Conformance and Validation Data

A reference validator, tool-agnostic conformance suite, sample datasets,
and viewer screenshots are available at:

<https://github.com/fideus-labs/ome-zarr-rfc4-validation>

The corpus spans OME-Zarr alongside source formats that carry orientation in
their own headers -- DICOM, NIfTI, NRRD, MINC, Analyze, Bruker ParaVision, FDF
and whole-slide imaging -- exercises the full 24-term vocabulary, and
includes must-reject cases for each rule above. Its driver runs any tool that
speaks the canonical contract, so the subcommand above can be checked against
the whole suite:

```bash
python conformance/run_conformance.py \
    --data-dir hf-data \
    --tool "ngff-zarr conformance {input}"
```

The sample data lives in a separate
[Hugging Face repo](https://huggingface.co/fideus-labs/ome-zarr-rfc4-data),
mirrored to a public S3 bucket that needs no credentials. The screenshots show
the same files opened in 3D Slicer, ITK-SNAP, NiiVue and napari -- useful for
seeing which viewers act on the `orientation` field and which only display axis
names.

## RFC-4 Functions

### Core Functions

- `itk_lps_to_anatomical_orientation(axis_name)`: Map ITK LPS axes to
  orientations
- `orientation_from_name(name)`: Resolve a preset name (`"LPS"`/`"RAS"`) to a
  per-axis orientation mapping
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

Anatomical orientation is written only when it is present on the image's axes.
Images without orientation produce standard OME-NGFF metadata with no
`orientation` field, so consumers that do not understand [RFC-4] are
unaffected.

## Best Practices

1. **Provide orientation** (via `axes_orientations` or the `orientation`
   keyword of `to_multiscales`) when working with medical/biological images
   where orientation matters
2. **Use convenience constants** (`LPS`, `RAS`) for standard coordinate systems
3. **Use ITK integration** for automatic LPS-based orientation when converting
   from medical image formats
4. **Document orientation assumptions** in your analysis pipelines when working
   with oriented data
5. **Validate orientations** match your expectations, especially when combining
   data from different sources -- `ngff-zarr conformance <store>` reports the
   per-axis verdict, and `validate_structural()` enforces the orientation rules
   as part of structural validation

[RFC-4]: https://ngff.openmicroscopy.org/rfc/4/index.html
