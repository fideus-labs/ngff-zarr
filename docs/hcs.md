<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🧬 High Content Screening (HCS) Support

NGFF-Zarr provides comprehensive support for High Content Screening (HCS) data,
implementing the plate and well metadata structures defined in the OME-Zarr
specification. This enables working with multi-well plate data commonly used in
drug discovery, phenotypic screening, and high-throughput imaging experiments.

## Background

High Content Screening is a method used in biological research and drug
discovery that involves the automated analysis of complex biological systems
using fluorescence microscopy. In HCS experiments, biological samples are
typically arranged in multi-well plates (such as 96-well or 384-well plates),
with each well containing different experimental conditions, treatments, or
samples.

The OME-Zarr specification defines a hierarchical structure for HCS data:

- **Plate**: The top-level container representing a physical screening plate
- **Wells**: Individual wells within the plate, identified by row and column
  coordinates
- **Fields**: Multiple imaging fields (fields of view) within each well
- **Acquisitions**: Different time points or experimental conditions

## Implementation

NGFF-Zarr's HCS implementation follows the OME-Zarr v0.4 specification and
provides:

### Core Data Classes

- `Plate`: Plate-level metadata including rows, columns, wells, and acquisitions
- `PlateWell`: Well location and path information
- `PlateRow` / `PlateColumn`: Row and column definitions
- `PlateAcquisition`: Acquisition metadata for time series or multi-condition
  experiments
- `Well`: Well-level metadata with image references
- `WellImage`: Individual field/image metadata within a well

### High-Level Interface

- `HCSPlate`: Convenient wrapper for plate-level operations
- `HCSWell`: Convenient wrapper for well-level operations and image access

## Reading HCS Data

### Loading a Plate

```python
import ngff_zarr as nz

# Load HCS plate data
plate = nz.from_hcs_zarr("path/to/hcs.ome.zarr")

print(f"Plate: {plate.metadata.name}")
print(f"Rows: {len(plate.metadata.rows)}")
print(f"Columns: {len(plate.metadata.columns)}")
print(f"Wells: {len(plate.metadata.wells)}")
```

### Accessing Wells

```python
# Get a specific well by row and column
well = plate.get_well("A", "1")  # Row A, Column 1

if well:
    print(f"Well A/1 has {len(well.images)} field(s)")

    # Access images/fields within the well
    image = well.get_image(0)  # First field
    if image:
        print(f"Image shape: {image.images[0].data.shape}")
        print(f"Image axes: {image.images[0].dims}")
```

### Working with Acquisitions

```python
# If the plate has multiple acquisitions (time points)
if plate.metadata.acquisitions:
    for acq in plate.metadata.acquisitions:
        print(f"Acquisition {acq.id}: {acq.name}")

    # Get image from specific acquisition
    well = plate.get_well("A", "1")
    image = well.get_image_by_acquisition(acquisition_id=0, field_index=0)
```

### Iterating Through Plate Data

```python
# Iterate through all wells
for well_meta in plate.metadata.wells:
    row_name = plate.metadata.rows[well_meta.rowIndex].name
    col_name = plate.metadata.columns[well_meta.columnIndex].name

    well = plate.get_well(row_name, col_name)
    if well:
        print(f"Well {row_name}/{col_name}: {len(well.images)} field(s)")

        # Process each field in the well
        for field_idx in range(len(well.images)):
            image = well.get_image(field_idx)
            if image:
                # Process the multiscale image
                ngff_image = image.images[0]  # First scale level
                print(f"  Field {field_idx}: {ngff_image.data.shape}")
```

## CLI Support for HCS Data

The `ngff-zarr` CLI tool supports converting individual well images from HCS plates using sub-path syntax.

### Converting a Specific Well Image

To convert a specific well and field from an HCS plate, provide the full path including the well coordinates and field index:

```bash
# Convert well A/1, field 0 from an HCS plate
ngff-zarr -i plate.ome.zarr/A/1/0 -o well_A1_field0.ome.zarr

# Convert well B/2, field 1
ngff-zarr -i plate.ome.zarr/B/2/1 -o well_B2_field1.ome.zarr
```

The CLI will automatically detect that you're accessing a well within an HCS plate and load the appropriate image data.

### Error Handling

If you attempt to convert the entire plate (without specifying a well/field), you'll receive a helpful error message:

```bash
$ ngff-zarr -i plate.ome.zarr -o output.ome.zarr
Error: The input appears to be an HCS (High Content Screening) plate structure
with 96 wells. To convert a specific well/image, provide the full path including
well and field:
  Examples: 'plate.ome.zarr/A/1/0', 'plate.ome.zarr/A/2/0', 'plate.ome.zarr/A/3/0'
For programmatic access to the full plate, use from_hcs_zarr() instead of from_ngff_zarr().
```

### Programmatic Access

For programmatic access to a specific well/field image within a plate, use the `from_ngff_zarr()` function with sub-path syntax. To load the full plate structure, use `from_hcs_zarr()`.

```python
from ngff_zarr import from_ngff_zarr

# Load a specific well image using sub-path
multiscales = from_ngff_zarr("plate.ome.zarr/A/1/0")

# Access the image data
image = multiscales.images[0]  # Full resolution
print(f"Shape: {image.data.shape}")
print(f"Axes: {image.dims}")
```

This is equivalent to using the HCS API:

```python
from ngff_zarr import from_hcs_zarr

# Load the plate
plate = from_hcs_zarr("plate.ome.zarr")

# Navigate to the well and image
well = plate.get_well("A", "1")
multiscales = well.get_image(0)
```

## Writing HCS Data

NGFF-Zarr provides support for writing HCS datasets through the `write_hcs_well_image` function. This function is designed for the typical HCS workflow where individual well images (fields of view) are written as they are acquired during the experiment.

### Basic Writing Workflow

The expected workflow for writing HCS data is to:

1. Create plate metadata describing the plate layout
2. Create the plate structure using `to_hcs_zarr`
3. Write individual field images as they are acquired using `write_hcs_well_image`

### Creating Plate Metadata

The plate metadata structure is the same for both v0.4 and v0.5. The version is specified when creating the `Plate` object:

```python
from ngff_zarr import (
    Plate, PlateColumn, PlateRow, PlateWell, PlateAcquisition
)

# Define plate structure
columns = [PlateColumn(name="1"), PlateColumn(name="2"), PlateColumn(name="3")]
rows = [PlateRow(name="A"), PlateRow(name="B")]

# Define wells
wells = [
    PlateWell(path="A/1", rowIndex=0, columnIndex=0),
    PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    PlateWell(path="A/3", rowIndex=0, columnIndex=2),
    PlateWell(path="B/1", rowIndex=1, columnIndex=0),
    PlateWell(path="B/2", rowIndex=1, columnIndex=1),
    PlateWell(path="B/3", rowIndex=1, columnIndex=2),
]

# Optional: Define acquisitions for time series
acquisitions = [
    PlateAcquisition(id=0, name="Baseline", maximumfieldcount=2),
    PlateAcquisition(id=1, name="Post-treatment", maximumfieldcount=2),
]

# Create plate metadata for v0.4 (default)
plate_metadata_v04 = Plate(
    name="Drug Screening Plate",
    columns=columns,
    rows=rows,
    wells=wells,
    acquisitions=acquisitions,
    field_count=2,  # Number of fields per well
    version="0.4",  # Use v0.4 format (Zarr v2)
)

# Create plate metadata for v0.5
plate_metadata_v05 = Plate(
    name="Drug Screening Plate",
    columns=columns,
    rows=rows,
    wells=wells,
    acquisitions=acquisitions,
    field_count=2,  # Number of fields per well
    version="0.5",  # Use v0.5 format (Zarr v3)
)
```

**Key Differences Between v0.4 and v0.5:**

- **v0.4**: Uses Zarr format 2, version field included in metadata dicts
- **v0.5**: Uses Zarr format 3, version at top-level OME wrapper only
- **Metadata Structure**: v0.5 stores version in `.zattrs["ome"]["version"]` instead of in individual metadata objects

### Setting Up the Plate Structure

First, create the plate structure using `to_hcs_zarr`:

```python
import ngff_zarr as nz
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr

# Create the HCS plate structure
hcs_plate = HCSPlate(store="my_screen.ome.zarr", plate_metadata=plate_metadata)
to_hcs_zarr(hcs_plate, "my_screen.ome.zarr")
```

### Writing Individual Well Images

After creating the plate structure, use `write_hcs_well_image` to write individual field images as they are acquired. **Important:** Pass the `version` parameter to match your plate version.

**Critical:** All wells in a plate must use the same OME-Zarr version (0.4 or 0.5). Mixing versions within a single plate is not supported.

### RFC-9 Zipped OME-Zarr (.ozx) Support

HCS plates can be written to RFC-9 compliant zipped OME-Zarr (`.ozx`) format by simply using an `.ozx` file extension. This format is ideal for sharing and archiving HCS datasets as a single file.

**Requirements:**
- OME-Zarr version 0.5 (required for RFC-9)
- Zarr-python >= 3.0.0

#### Using HCSPlateWriter for Efficient .ozx Writing

For writing multiple wells to `.ozx` format (or for parallel writing to any format), use the `HCSPlateWriter` context manager. This defers `.ozx` file creation until all wells are written, making it much more efficient than writing wells individually.

```python
import ngff_zarr as nz
from ngff_zarr.hcs import HCSPlateWriter

# Create plate metadata with version 0.5 (required for .ozx)
plate_metadata = nz.Plate(
    name="Drug Screening Plate",
    columns=columns,
    rows=rows,
    wells=wells,
    field_count=2,
    version="0.5",  # Required for .ozx
)

# Use context manager to write multiple wells efficiently
with HCSPlateWriter("my_screen.ozx", plate_metadata) as writer:
    for well_data in acquisition_data:
        writer.write_well_image(
            multiscales=well_data.image,
            row_name=well_data.row,
            column_name=well_data.col,
            field_index=well_data.field,
        )
# .ozx file is created automatically when exiting the context
```

**Benefits of HCSPlateWriter:**
- **Efficient .ozx creation**: Single ZIP operation instead of re-zipping after each well
- **Parallel writing support**: Thread-safe for concurrent well writes
- **Memory efficient**: Handles large plates without loading everything into memory
- **Automatic cleanup**: Temporary files are cleaned up automatically

#### Parallel Writing Example

```python
from concurrent.futures import ThreadPoolExecutor

def write_well(writer, well_data):
    writer.write_well_image(
        multiscales=well_data.image,
        row_name=well_data.row,
        column_name=well_data.col,
        field_index=well_data.field,
    )

with HCSPlateWriter("plate.ozx", plate_metadata) as writer:
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda wd: write_well(writer, wd), well_data_list)
```

#### Single-Well .ozx Writing (Simple API)

For writing a single well or a few wells, you can still use `write_hcs_well_image` directly, but note that each call will recreate the entire `.ozx` file:

```python
# Less efficient for multiple wells - creates .ozx after each write
nz.write_hcs_well_image(
    store="my_screen.ozx",
    multiscales=field_image,
    plate_metadata=plate_metadata,
    row_name="A",
    column_name="1",
    field_index=0,
    version="0.5",  # Must be 0.5 for .ozx
)
```

#### Reading .ozx Files

The resulting `.ozx` file is a standard ZIP archive containing the entire HCS plate structure and can be read using `from_hcs_zarr`:

```python
# Read .ozx file - works the same as regular .ome.zarr
plate = nz.from_hcs_zarr("my_screen.ozx")
well = plate.get_well("A", "1")
image = well.get_image(0)
```

**Note:** `.ozx` files automatically use sharding (chunks_per_shard=2 by default) for efficient compression and access.

#### Converting Existing Plates to .ozx

If you've already written an HCS plate to a regular `.ome.zarr` directory, you can convert it to `.ozx` format using `write_store_to_zip`:

```python
from ngff_zarr import write_store_to_zip

# Convert an existing plate to .ozx
write_store_to_zip(
    "existing_plate.ome.zarr",  # Source plate directory
    "plate.ozx",                 # Output .ozx file
    version="0.5"                # Required for .ozx
)
```

This is useful for:
- **Archiving**: Convert large plates to single-file archives
- **Sharing**: Package complete plates for easy distribution

The conversion preserves all plate structure, well data, and metadata. The resulting `.ozx` file can be read using `from_hcs_zarr` just like any other HCS plate.

### Standard Writing Examples

```python
# IMPORTANT: The plate and all wells must use the same version
# Define the version once to ensure consistency
ome_zarr_version = "0.4"  # or "0.5"

# Write first field to well A/1 (v0.4 format)
nz.write_hcs_well_image(
    store="my_screen.ome.zarr",
    multiscales=field_image,  # Your Multiscales image data
    plate_metadata=plate_metadata_v04,
    row_name="A",
    column_name="1",
    field_index=0,  # First field of view
    acquisition_id=0,
    version=ome_zarr_version,  # Must match plate version
)

# Write second field to the same well
nz.write_hcs_well_image(
    store="my_screen.ome.zarr",
    multiscales=field_image_2,
    plate_metadata=plate_metadata_v04,
    row_name="A",
    column_name="1",
    field_index=1,  # Second field of view
    acquisition_id=0,
    version=ome_zarr_version,  # Use same version
)

# Write to different well
nz.write_hcs_well_image(
    store="my_screen.ome.zarr",
    multiscales=field_image_3,
    plate_metadata=plate_metadata_v04,
    row_name="A",
    column_name="2",
    field_index=0,
    acquisition_id=0,
    version=ome_zarr_version,  # Use same version
)
```

Example for v0.5:

```python
# IMPORTANT: The plate and all wells must use the same version
ome_zarr_version = "0.5"

# Write first field to well A/1 (v0.5 format)
nz.write_hcs_well_image(
    store="my_screen_v05.ome.zarr",
    multiscales=field_image,  # Your Multiscales image data
    plate_metadata=plate_metadata_v05,
    row_name="A",
    column_name="1",
    field_index=0,  # First field of view
    acquisition_id=0,
    version=ome_zarr_version,  # Must match plate version
)
```

### Complete Example: Drug Screening Workflow (v0.4)

```python
import ngff_zarr as nz
import numpy as np
from ngff_zarr import NgffImage, to_multiscales
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
from ngff_zarr.v04.zarr_metadata import *

# IMPORTANT: Define version once - all plate and well operations must use the same version
ome_zarr_version = "0.4"

# Create plate layout for a 96-well plate (subset)
columns = [PlateColumn(name=str(i)) for i in range(1, 13)]  # 12 columns
rows = [PlateRow(name=chr(65 + i)) for i in range(8)]       # 8 rows (A-H)

wells = []
for row_idx, row in enumerate(rows):
    for col_idx, col in enumerate(columns):
        wells.append(PlateWell(
            path=f"{row.name}/{col.name}",
            rowIndex=row_idx,
            columnIndex=col_idx
        ))

plate_metadata = Plate(
    name="Compound Screen - Plate 1",
    columns=columns,
    rows=rows,
    wells=wells,
    field_count=4,  # 4 fields per well
    version=ome_zarr_version  # Use consistent version
)

# Function to create synthetic field data
def create_field_image(treatment_effect=1.0):
    # Create synthetic microscopy data: T, C, Z, Y, X
    base_intensity = 100
    data = np.random.poisson(
        base_intensity * treatment_effect,
        size=(1, 2, 10, 512, 512)
    ).astype(np.uint16)

    ngff_image = NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.3, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        name=f"Field_effect_{treatment_effect}",
    )

    return to_multiscales(ngff_image, scale_factors=[2, 4])

# Simulate acquisition workflow
plate_store = "drug_screen_plate1.ome.zarr"

# First, create the plate structure
hcs_plate = HCSPlate(store=plate_store, plate_metadata=plate_metadata)
to_hcs_zarr(hcs_plate, plate_store)

# Iterate through wells as they are imaged
for row in ["A", "B", "C"]:  # First 3 rows for demo
    for col in ["1", "2", "3", "4"]:  # First 4 columns for demo

        # Simulate different treatment effects
        if row == "A":
            effect = 0.5  # Inhibitor
        elif row == "B":
            effect = 1.0  # Control
        else:
            effect = 2.0  # Activator

        # Acquire and write multiple fields per well
        for field in range(2):  # 2 fields for demo
            field_image = create_field_image(effect + field * 0.1)

            nz.write_hcs_well_image(
                store=plate_store,
                multiscales=field_image,
                plate_metadata=plate_metadata,
                row_name=row,
                column_name=col,
                field_index=field,
                acquisition_id=0,
                version=ome_zarr_version,  # Must match plate version
            )

            print(f"Acquired well {row}/{col}, field {field}")

print(f"Screening complete! Plate saved to: {plate_store}")

# Verify the written data
final_plate = nz.from_hcs_zarr(plate_store)
print(f"Plate: {final_plate.name}")
print(f"Wells written: {len([w for w in final_plate.wells if final_plate.get_well(final_plate.rows[w.rowIndex].name, final_plate.columns[w.columnIndex].name) is not None])}")
```

### Complete Example: v0.5 Drug Screening Workflow

```python
import ngff_zarr as nz
import numpy as np
from ngff_zarr import NgffImage, to_multiscales
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
from ngff_zarr.v04.zarr_metadata import *

# IMPORTANT: Define version once - all plate and well operations must use the same version
ome_zarr_version = "0.5"

# Create plate layout for v0.5
columns = [PlateColumn(name=str(i)) for i in range(1, 4)]  # 3 columns
rows = [PlateRow(name=chr(65 + i)) for i in range(2)]      # 2 rows (A-B)

wells = []
for row_idx, row in enumerate(rows):
    for col_idx, col in enumerate(columns):
        wells.append(PlateWell(
            path=f"{row.name}/{col.name}",
            rowIndex=row_idx,
            columnIndex=col_idx
        ))

plate_metadata = Plate(
    name="v0.5 Screening Plate",
    columns=columns,
    rows=rows,
    wells=wells,
    field_count=2,
    version=ome_zarr_version,  # Use consistent version
)

# Function to create synthetic field data
def create_field_image(treatment_effect=1.0):
    base_intensity = 100
    data = np.random.poisson(
        base_intensity * treatment_effect,
        size=(1, 2, 10, 512, 512)
    ).astype(np.uint16)

    ngff_image = NgffImage(
        data=data,
        dims=["t", "c", "z", "y", "x"],
        scale={"t": 1.0, "c": 1.0, "z": 0.3, "y": 0.325, "x": 0.325},
        translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
        name=f"Field_effect_{treatment_effect}",
    )

    return to_multiscales(ngff_image, scale_factors=[2, 4])

# Create plate structure
plate_store = "screening_v05.ome.zarr"
hcs_plate = HCSPlate(store=plate_store, plate_metadata=plate_metadata)
to_hcs_zarr(hcs_plate, plate_store)

# Write field images with v0.5
for row in ["A", "B"]:
    for col in ["1", "2", "3"]:
        effect = 1.0 if row == "A" else 2.0

        for field in range(2):
            field_image = create_field_image(effect)

            nz.write_hcs_well_image(
                store=plate_store,
                multiscales=field_image,
                plate_metadata=plate_metadata,
                row_name=row,
                column_name=col,
                field_index=field,
                acquisition_id=0,
                version=ome_zarr_version,  # Must match plate version
            )

            print(f"Acquired well {row}/{col}, field {field}")

# Read back and verify v0.5 structure
loaded_plate = nz.from_hcs_zarr(plate_store)
print(f"Plate version: {loaded_plate.metadata.version}")
print(f"Plate name: {loaded_plate.name}")
```
```

### Key Points for HCS Writing

**Individual Image Writing**: HCS datasets are typically written field-by-field as images are acquired, rather than all at once. This allows for:
- Real-time quality control during acquisition
- Incremental backup and storage
- Parallel acquisition from multiple microscopes
- Early analysis of completed wells

**Metadata Management**: The workflow with `to_hcs_zarr` and `write_hcs_well_image`:
- First creates the plate structure using `to_hcs_zarr`
- Then writes individual field images using `write_hcs_well_image`
- Updates well metadata automatically as fields are added
- Maintains proper NGFF hierarchical structure
- **v0.4**: Stores version in each metadata dict (`.zattrs["well"]["version"]`)
- **v0.5**: Stores version only at top-level OME wrapper (`.zattrs["ome"]["version"]`)

**Performance Considerations**:
- Each field is written as a complete multiscale image
- Use appropriate chunking for your analysis workflow
- Consider compression settings for long-term storage
- Multiple processes can write to different wells simultaneously

### Error Handling

The function validates inputs and provides clear error messages:

```python
# This will raise ValueError if coordinates are invalid
try:
    nz.write_hcs_well_image(
        store="plate.zarr",
        multiscales=image,
        plate_metadata=plate_metadata,
        row_name="Z",  # Invalid row
        column_name="1",
        field_index=0,
    )
except ValueError as e:
    print(f"Error: {e}")  # "Row 'Z' not found in plate metadata"
```

### Simple Complete Example (v0.4)

Here's a minimal working example for v0.4:

```python
import ngff_zarr as nz
import numpy as np
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
from ngff_zarr.v04.zarr_metadata import *

# IMPORTANT: Define version once - all plate and well operations must use the same version
ome_zarr_version = "0.4"

# Create plate layout
columns = [PlateColumn(name="1"), PlateColumn(name="2")]
rows = [PlateRow(name="A"), PlateRow(name="B")]
wells = [
    PlateWell(path="A/1", rowIndex=0, columnIndex=0),
    PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    PlateWell(path="B/1", rowIndex=1, columnIndex=0),
    PlateWell(path="B/2", rowIndex=1, columnIndex=1),
]

plate_metadata = Plate(
    columns=columns, rows=rows, wells=wells,
    name="Example Plate v0.4", field_count=1,
    version=ome_zarr_version
)

# Create synthetic image data
data = np.random.randint(0, 255, size=(1, 1, 10, 256, 256), dtype=np.uint8)
ngff_image = nz.NgffImage(
    data=data, dims=["t", "c", "z", "y", "x"],
    scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
    translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0}
)
multiscales = nz.to_multiscales(ngff_image)

# First create the plate structure
hcs_plate = HCSPlate(store="my_plate_v04.ome.zarr", plate_metadata=plate_metadata)
to_hcs_zarr(hcs_plate, "my_plate_v04.ome.zarr")

# Write images to each well (all must use same version as plate)
for well in wells:
    row_name = rows[well.rowIndex].name
    col_name = columns[well.columnIndex].name

    nz.write_hcs_well_image(
        store="my_plate_v04.ome.zarr",
        multiscales=multiscales,
        plate_metadata=plate_metadata,
        row_name=row_name,
        column_name=col_name,
        field_index=0,
        version=ome_zarr_version,  # Must match plate version
    )

# Verify the result
plate = nz.from_hcs_zarr("my_plate_v04.ome.zarr")
print(f"Created v0.4 plate with {len(plate.wells)} wells")
```

### Simple Complete Example (v0.5)

Here's a minimal working example for v0.5:

```python
import ngff_zarr as nz
import numpy as np
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
from ngff_zarr.v04.zarr_metadata import *

# IMPORTANT: Define version once - all plate and well operations must use the same version
ome_zarr_version = "0.5"

# Create plate layout
columns = [PlateColumn(name="1"), PlateColumn(name="2")]
rows = [PlateRow(name="A"), PlateRow(name="B")]
wells = [
    PlateWell(path="A/1", rowIndex=0, columnIndex=0),
    PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    PlateWell(path="B/1", rowIndex=1, columnIndex=0),
    PlateWell(path="B/2", rowIndex=1, columnIndex=1),
]

plate_metadata = Plate(
    columns=columns, rows=rows, wells=wells,
    name="Example Plate v0.5", field_count=1,
    version=ome_zarr_version
)

# Create synthetic image data
data = np.random.randint(0, 255, size=(1, 1, 10, 256, 256), dtype=np.uint8)
ngff_image = nz.NgffImage(
    data=data, dims=["t", "c", "z", "y", "x"],
    scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
    translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0}
)
multiscales = nz.to_multiscales(ngff_image)

# First create the plate structure
hcs_plate = HCSPlate(store="my_plate_v05.ome.zarr", plate_metadata=plate_metadata)
to_hcs_zarr(hcs_plate, "my_plate_v05.ome.zarr")

# Write images to each well (all must use same version as plate)
for well in wells:
    row_name = rows[well.rowIndex].name
    col_name = columns[well.columnIndex].name

    nz.write_hcs_well_image(
        store="my_plate_v05.ome.zarr",
        multiscales=multiscales,
        plate_metadata=plate_metadata,
        row_name=row_name,
        column_name=col_name,
        field_index=0,
        version=ome_zarr_version,  # Must match plate version
    )

# Verify the result
plate = nz.from_hcs_zarr("my_plate_v05.ome.zarr")
print(f"Created v0.5 plate with {len(plate.wells)} wells")
print(f"Version: {plate.metadata.version}")
```

These approaches allow you to write individual well images as they are acquired during HCS experiments, making them ideal for real-time acquisition workflows.

### Creating Plate Metadata

```python
from ngff_zarr.v04.zarr_metadata import (
    Plate, PlateColumn, PlateRow, PlateWell,
    Well, WellImage, PlateAcquisition
)

# Define plate structure
columns = [PlateColumn(name="1"), PlateColumn(name="2")]
rows = [PlateRow(name="A"), PlateRow(name="B")]

# Define wells
wells = [
    PlateWell(path="A/1", rowIndex=0, columnIndex=0),
    PlateWell(path="A/2", rowIndex=0, columnIndex=1),
    PlateWell(path="B/1", rowIndex=1, columnIndex=0),
    PlateWell(path="B/2", rowIndex=1, columnIndex=1),
]

# Create plate metadata
plate_metadata = Plate(
    name="My Screening Plate",
    columns=columns,
    rows=rows,
    wells=wells,
    field_count=2  # Number of fields per well
)
```

### Writing Well Data

```python
# Create well metadata with multiple fields
well_images = [
    WellImage(path="0"),  # Field 0
    WellImage(path="1"),  # Field 1
]

well_metadata = Well(images=well_images)

# Write to zarr (this is typically done as part of a larger workflow)
# The exact implementation depends on your specific data and requirements
```

## Validation

HCS data can be validated against the OME-Zarr specification:

```python
# Validate HCS metadata during loading
plate = nz.from_hcs_zarr("path/to/hcs.ome.zarr", validate=True)
```

This uses the appropriate HCS-specific JSON schema validation rather than the
standard image schema.

## Integration with Standard Workflows

HCS data integrates seamlessly with other NGFF-Zarr functionality:

```python
# Each field is a standard multiscale image
well = plate.get_well("A", "1")
image = well.get_image(0)

# Use standard multiscale operations
first_scale = image.images[0]
print(f"Dimensions: {first_scale.dims}")
print(f"Scale: {first_scale.scale}")
print(f"Translation: {first_scale.translation}")

# Access the dask array for computation
data_array = first_scale.data
processed = data_array.mean(axis=0)  # Example processing
```

## Common Use Cases

### Drug Screening Analysis

```python
import numpy as np

plate = nz.from_hcs_zarr("drug_screen.ome.zarr")

results = {}
for well_meta in plate.metadata.wells:
    row = plate.metadata.rows[well_meta.rowIndex].name
    col = plate.metadata.columns[well_meta.columnIndex].name

    well = plate.get_well(row, col)
    if well:
        # Analyze all fields in the well
        field_intensities = []
        for field_idx in range(len(well.images)):
            image = well.get_image(field_idx)
            if image:
                data = image.images[0].data.compute()
                mean_intensity = np.mean(data)
                field_intensities.append(mean_intensity)

        results[f"{row}/{col}"] = np.mean(field_intensities)

print("Well intensities:", results)
```

### Time Series Analysis

```python
# For plates with multiple acquisitions (time points)
plate = nz.from_hcs_zarr("time_series.ome.zarr")

if plate.metadata.acquisitions:
    well = plate.get_well("A", "1")

    time_series = []
    for acq in plate.metadata.acquisitions:
        image = well.get_image_by_acquisition(acq.id, field_index=0)
        if image:
            data = image.images[0].data.compute()
            time_series.append(data.mean())

    print(f"Time series for well A/1: {time_series}")
```

## Technical Details

### Metadata Structure

The HCS implementation uses the following metadata hierarchy:

```
Plate Root (.zattrs)
├── plate/
│   ├── name
│   ├── rows[]
│   ├── columns[]
│   ├── wells[]
│   ├── acquisitions[]
│   └── field_count
└── bioformats2raw.layout (optional)

Well Groups (e.g., A/1/.zattrs)
├── well/
│   ├── images[]
│   └── version
└── Standard multiscale metadata for each field
```

### Path Conventions

- Plate root: Contains plate metadata and well groups
- Well paths: `{row}/{column}/` (e.g., `A/1/`, `B/2/`)
- Field paths: `{row}/{column}/{field}/` (e.g., `A/1/0/`, `A/1/1/`)

### Performance Considerations

- **Lazy Loading**: Images are loaded on-demand
- **Dask Integration**: Large datasets can be processed in chunks
- **Memory Management**: Wells and images use bounded LRU caches to prevent
  memory bloat with large plates
  - Wells cache: Configurable limit (default: 500 wells per plate)
  - Images cache: Configurable limit (default: 100 images per well)
  - Automatic eviction of least recently used items when limits are exceeded
- **Parallel Access**: Multiple wells can be processed in parallel using Dask

### Cache Configuration

The HCS implementation includes intelligent caching to balance performance and
memory usage:

```python
import ngff_zarr as nz

# Configure cache sizes globally
nz.config.hcs_well_cache_size = 100   # Max wells to cache per plate
nz.config.hcs_image_cache_size = 50   # Max images to cache per well

# Or configure per operation
plate = nz.from_hcs_zarr(
    "plate.zarr",
    well_cache_size=50,    # Custom well cache size
    image_cache_size=25    # Custom image cache size
)
```

**Cache Benefits:**

- Prevents memory issues when working with large plates (1000+ wells)
- Maintains recently accessed data for fast repeated access
- Automatically manages memory without manual intervention
- Configurable limits for different hardware and use cases

## See Also

- [OME-Zarr HCS Specification](https://ngff.openmicroscopy.org/latest/#plate-md)
- [Python API Reference](./python.md)
- [Specification Features](./spec_features.md)
