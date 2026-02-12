<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🤔 Frequently Asked Questions (FAQ)

## Performance and Memory

### Why does `to_multiscales` perform computation immediately with large datasets?

For both small and large datasets, `to_multiscales` returns a simple Python
dataclass composed of basic Python datatypes and lazy dask arrays. The lazy dask
arrays, as with all dask arrays, are a task graph that defines how to generate
those arrays and do not exist in memory. This is the case for both regular sized
datasets and very large datasets.

For very large datasets, though, data conditioning and task graph engineering is
performed during construction to improve performance and avoid running out of
system memory. This preprocessing step ensures that when you eventually compute
the arrays, the operations are optimized and memory-efficient.

If you want to avoid this behavior entirely, you can pass `cache=False` to
`to_multiscales`:

```python
multiscales = to_multiscales(image, cache=False)
```

**Warning:** Disabling caching may cause you to run out of memory when working
with very large datasets!

The lazy evaluation approach allows ngff-zarr to handle extremely large datasets
that wouldn't fit in memory, while still providing optimal performance through
intelligent task graph optimization.

## Network Storage and Authentication

### How do I read from network stores like S3, GCS, or other remote storage?

ngff-zarr can read from any network store that provides a Zarr Python compatible
interface. This includes stores from
[fsspec](https://filesystem-spec.readthedocs.io/en/latest/), which supports many
protocols including S3, Google Cloud Storage, Azure Blob Storage, and more.

You can construct network stores with authentication options and pass them
directly to ngff-zarr functions.

The following examples require the `fsspec` and `s3fs` packages:

```bash
pip install fsspec s3fs
```

```python
import zarr
from ngff_zarr import from_ngff_zarr

# S3 example with authentication using FsspecStore
s3_store = zarr.storage.FsspecStore.from_url(
    "s3://my-bucket/my-dataset.zarr",
    storage_options={
        "key": "your-access-key",
        "secret": "your-secret-key",
        "region_name": "us-west-2"
    }
)

# Read from the S3 store
multiscales = from_ngff_zarr(s3_store)
```

For public datasets, you can omit authentication:

```python
# Example using OME-Zarr Open Science Vis Datasets
s3_store = zarr.storage.FsspecStore.from_url(
    "s3://ome-zarr-scivis/v0.5/96x2/carp.ome.zarr",
    storage_options={"anon": True}  # Anonymous access for public data
)

multiscales = from_ngff_zarr(s3_store)
```

You can also pass S3 URLs directly to ngff-zarr functions, which will create the
appropriate store automatically:

```python
# Direct URL access for public datasets
multiscales = from_ngff_zarr(
    "s3://ome-zarr-scivis/v0.5/96x2/carp.ome.zarr",
    storage_options={"anon": True}
)
```

For more control over the underlying filesystem, you can use S3FileSystem
directly:

```python
import zarr
from s3fs import S3FileSystem

# Using S3FileSystem with Zarr
fs = S3FileSystem(
    key="your-access-key",
    secret="your-secret-key",
    region_name="us-west-2"
)
store = zarr.storage.FsspecStore(fs=fs, path="my-bucket/my-dataset.zarr")

multiscales = from_ngff_zarr(store)
```

**Authentication Options:**

In addition to specification of credentials explicitly,
[there are other options](https://s3fs.readthedocs.io/en/latest/#credentials).

- **Environment variables**: Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  etc.
- **IAM roles**: Use EC2 instance profiles or assume roles
- **Configuration files**: Use `~/.aws/credentials` or similar
- **Direct parameters**: Pass credentials directly to the store constructor

The same patterns work for other cloud providers (GCS, Azure) by using their
respective fsspec implementations (e.g., `gcsfs`, `adlfs`).

## Troubleshooting

### I'm getting "Invalid page offset" or "OSError" errors when converting TIFF/SVS files

This error typically indicates that the TIFF or SVS file is corrupted, truncated, or has an invalid internal structure. Common causes include:

1. **Incomplete file transfer**: The file may have been incompletely downloaded or transferred
2. **Disk errors**: Physical disk errors during file creation
3. **Software bugs**: Issues in the original software that created the file
4. **File format violations**: Non-standard TIFF structures that violate the specification

**How to diagnose:**

```bash
# Try opening the file with tifffile directly to check for errors
python3 -c "import tifffile; tif = tifffile.TiffFile('your_file.svs'); print(f'Series: {len(tif.series)}'); tif.close()"
```

If this command fails or shows errors, the file is likely corrupted.

**Possible solutions:**

1. **Re-download or re-transfer the file** from the original source
2. **Verify file integrity** using checksums if available
3. **Try repairing the file** using specialized TIFF repair tools
4. **Contact the data provider** if the file came from an external source
5. **Use the original acquisition software** to re-export the image if possible

**Recent improvements:**

As of the latest version, ngff-zarr provides better error messages when encountering corrupted files, helping you identify the issue early rather than waiting for the conversion to fail after hours of processing.

### My TIFF/SVS conversion is taking extremely long (hours/days)

Very large whole-slide imaging (WSI) files like SVS can take significant time to convert, but there are ways to optimize the process:

**Expected performance:**

- Small files (< 1GB): Minutes
- Medium files (1-10GB): 15-60 minutes
- Large files (10-50GB): 1-4 hours
- Very large files (> 50GB): Several hours to days

**Optimization tips:**

1. **Use the `--local-cluster` option** for better parallelization:
   ```bash
   ngff-zarr --local-cluster -i input.svs -o output.ozx
   ```

2. **Adjust memory target** to use more of your available RAM:
   ```bash
   # For a system with 64GB RAM, use 50GB
   ngff-zarr --memory-target 50G -i input.svs -o output.ozx
   ```

3. **Check for file corruption** before starting long conversions:
   ```bash
   # Quick validation
   python3 -c "import tifffile; tif = tifffile.TiffFile('input.svs'); print(f'{len(tif.series)} series found'); tif.close()"
   ```

4. **Use SSD storage** for both input and output if possible, as I/O can be a bottleneck

5. **Monitor system resources** during conversion to identify bottlenecks:
   ```bash
   # In a separate terminal
   htop  # or 'top' on macOS/Linux
   ```

If conversion is taking unexpectedly long and then fails with an error, the file is likely corrupted. See the section above for diagnosis and solutions.
