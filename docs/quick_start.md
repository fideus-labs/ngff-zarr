<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ⚡ Quick start

## Installation

```shell
pip install "ngff-zarr[cli]"
```

## Command line conversion

```shell
ngff-zarr -i ./cthead1.png -o ./cthead1.ome.zarr
```

## NumPy array to OME-Zarr

```python
import ngff_zarr as nz
import numpy as np

# Generate pixel data (e.g. from a microscope, simulation, etc.)
data = np.random.randint(0, 256, int(1e6)).reshape((1000, 1000))

multiscales = nz.to_multiscales(data)

nz.to_ngff_zarr('example.ome.zarr', multiscales)
```

## High Content Screening (HCS)

```python
import ngff_zarr as nz

# Load HCS plate data
plate = nz.from_hcs_zarr('screening_plate.ome.zarr')

# Access a specific well
well = plate.get_well("A", "1")
image = well.get_image(0)  # First field

print(f"Image shape: {image.images[0].data.shape}")
```
