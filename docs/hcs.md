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

## Writing HCS Data

NGFF-Zarr provides support for writing HCS datasets through the `write_hcs_well_image` function. This function is designed for the typical HCS workflow where individual well images (fields of view) are written as they are acquired during the experiment.

### Basic Writing Workflow

The expected workflow for writing HCS data is to:

1. Create plate metadata describing the plate layout
2. Create the plate structure using `to_hcs_zarr`
3. Write individual field images as they are acquired using `write_hcs_well_image`

### Creating Plate Metadata

```python
from ngff_zarr.v04.zarr_metadata import (
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

# Create plate metadata
plate_metadata = Plate(
    name="Drug Screening Plate",
    columns=columns,
    rows=rows,
    wells=wells,
    acquisitions=acquisitions,
    field_count=2  # Number of fields per well
)
```

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

After creating the plate structure, use `write_hcs_well_image` to write individual field images as they are acquired:

```python
# Write first field to well A/1
nz.write_hcs_well_image(
    store="my_screen.ome.zarr",
    multiscales=field_image,  # Your Multiscales image data
    plate_metadata=plate_metadata,
    row_name="A",
    column_name="1",
    field_index=0,  # First field of view
    acquisition_id=0,
)

# Write second field to the same well
nz.write_hcs_well_image(
    store="my_screen.ome.zarr",
    multiscales=field_image_2,
    plate_metadata=plate_metadata,
    row_name="A",
    column_name="1",
    field_index=1,  # Second field of view
    acquisition_id=0,
)

# Write to different well
nz.write_hcs_well_image(
    store="my_screen.ome.zarr",
    multiscales=field_image_3,
    plate_metadata=plate_metadata,
    row_name="A",
    column_name="2",
    field_index=0,
    acquisition_id=0,
)
```

### Complete Example: Drug Screening Workflow

```python
import ngff_zarr as nz
import numpy as np
from ngff_zarr import NgffImage, to_multiscales
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
from ngff_zarr.v04.zarr_metadata import *

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
    version="0.4"
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
            )

            print(f"Acquired well {row}/{col}, field {field}")

print(f"Screening complete! Plate saved to: {plate_store}")

# Verify the written data
final_plate = nz.from_hcs_zarr(plate_store)
print(f"Plate: {final_plate.name}")
print(f"Wells written: {len([w for w in final_plate.wells if final_plate.get_well(final_plate.rows[w.rowIndex].name, final_plate.columns[w.columnIndex].name) is not None])}")
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
- Handles both v0.4 and v0.5 formats

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

### Simple Complete Example

Here's a minimal working example:

```python
import ngff_zarr as nz
import numpy as np
from ngff_zarr.hcs import HCSPlate, to_hcs_zarr
from ngff_zarr.v04.zarr_metadata import *

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
    name="Example Plate", field_count=1
)

# Create synthetic image data
data = np.random.randint(0, 255, size=(1, 1, 10, 256, 256), dtype=np.uint8)
ngff_image = nz.NgffImage(
    data=data, dims=["t", "c", "z", "y", "x"],
    scale={"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.325, "x": 0.325},
    translation={"t": 0.0, "c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0}
)
multiscales = nz.to_multiscales(ngff_image)

# Write images to each well
# First create the plate structure
hcs_plate = HCSPlate(store="my_plate.ome.zarr", plate_metadata=plate_metadata)
to_hcs_zarr(hcs_plate, "my_plate.ome.zarr")

for well in wells:
    row_name = rows[well.rowIndex].name
    col_name = columns[well.columnIndex].name

    nz.write_hcs_well_image(
        store="my_plate.ome.zarr",
        multiscales=multiscales,
        plate_metadata=plate_metadata,
        row_name=row_name,
        column_name=col_name,
        field_index=0,
    )

# Verify the result
plate = nz.from_hcs_zarr("my_plate.ome.zarr")
print(f"Created plate with {len(plate.wells)} wells")
```

This approach allows you to write individual well images as they are acquired during HCS experiments, making it ideal for real-time acquisition workflows.

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
