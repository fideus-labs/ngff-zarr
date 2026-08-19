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

ngff-zarr reads remote OME-Zarr stores directly from their URLs through
zarrista's async API, backed by
[obstore](https://developmentseed.org/obstore/latest/). Supported protocols:
http(s), S3, Google Cloud Storage, and Azure Blob Storage.

The remote backend requires Python >= 3.11 and is installed with the `remote`
extra:

```bash
pip install "ngff-zarr[remote]"
```

Pass the URL directly to ngff-zarr functions; authentication and other
filesystem options go in `storage_options`:

```python
from ngff_zarr import from_ngff_zarr

# S3 example with authentication
multiscales = from_ngff_zarr(
    "s3://my-bucket/my-dataset.zarr",
    storage_options={
        "key": "your-access-key",
        "secret": "your-secret-key",
        "region_name": "us-west-2",
    },
)
```

For public datasets, use anonymous access:

```python
# Example using OME-Zarr Open Science Vis Datasets
multiscales = from_ngff_zarr(
    "s3://ome-zarr-scivis/v0.5/96x2/carp.ome.zarr",
    storage_options={"anon": True},  # Anonymous access for public data
)
```

fsspec-style option names (`key`, `secret`, `anon`, `token`, `endpoint_url`,
`region_name`, ...) are accepted and translated to their obstore equivalents;
obstore-native option names also work and are passed through as-is.

**Authentication Options:**

- **Environment variables**: Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  etc.
- **IAM roles**: Use EC2 instance profiles via the instance metadata service
- **Direct parameters**: Pass credentials in `storage_options`

The same patterns work for other cloud providers (GCS, Azure) with their
respective `storage_options`.

**Remote writes** are not supported: write to a local directory and upload
afterwards (for example with `aws s3 sync` or `rclone`).

## Zarr Backend and zarr-python Compatibility

### Does ngff-zarr require zarr-python?

No. ngff-zarr reads and writes Zarr through
[zarrista](https://github.com/developmentseed/zarrista), a Python Zarr
implementation built on the Rust [zarrs](https://zarrs.dev/) library.
zarr-python is not a dependency of ngff-zarr, and the two can be installed
side by side without conflict.

The stores ngff-zarr writes are standard OME-Zarr: version 0.4 uses the Zarr
v2 format and versions 0.5+ use Zarr v3. They remain readable by zarr-python
2 and 3 and by the other OME-Zarr implementations; ngff-zarr's test suite
verifies round-trips against both zarr-python majors.

### Why does passing a zarr-python store object raise `TypeError`?

Store objects from zarr-python (`LocalStore`, `DirectoryStore`,
`MemoryStore`, ...) are no longer accepted as inputs. Pass the path or URL a
store wraps instead:

```python
# Before
from zarr.storage import LocalStore
multiscales = from_ngff_zarr(LocalStore("image.ome.zarr"))

# After
multiscales = from_ngff_zarr("image.ome.zarr")
```

Writing targets local directory paths (or `.ozx` archive paths); reading
additionally accepts remote URLs and in-memory key-to-bytes mappings (a
plain `dict` works). See
[Read an OME-Zarr](./python.md#read-an-ome-zarr) for the full list of
accepted store types.

### Why can't ngff-zarr read a Zarr v3 store with `nthreads` in its blosc codec metadata?

Some tools wrote deprecated `nthreads`/`blocksize` keys into the blosc codec
configuration of Zarr v3 array metadata. When these keys appear in a store's
*consolidated* metadata (the root `zarr.json`), ngff-zarr tolerates and
ignores them. When they appear in an individual array's `zarr.json`, the
zarrista backend rejects the store as out-of-spec. To repair such a store,
remove the two keys from the `blosc` codec `configuration` object in each
array-level `zarr.json` document.

## File Formats

### Where can I find documentation for specific file formats like TIFF or LIF?

ngff-zarr converts most bioimaging formats to OME-Zarr. Detailed, format-specific
guides are available for:

- [TIFF and OME-TIFF Support](./tiff.md) — automatic metadata extraction and
  multi-series conversion
- [Leica Image Format (LIF) Support](./lif.md) — Leica microscopy data

## Troubleshooting

### I'm getting "Invalid argument", "Invalid page offset", or `OSError` errors when converting TIFF/SVS files

These errors usually mean the TIFF or SVS file is corrupted, truncated, or has
an invalid internal structure. With very large whole-slide images the failure
can surface only late in the conversion, once the affected data is actually
read. Common causes include:

1. **Incomplete file transfer**: the file was partially downloaded or copied
2. **Disk errors**: physical disk errors during file creation
3. **Software bugs**: issues in the software that created the file
4. **File format violations**: non-standard structures that violate the spec

**How to diagnose:**

```bash
# Try opening the file with tifffile directly to check for errors
python3 -c "import tifffile; tif = tifffile.TiffFile('your_file.svs'); print(f'Series: {len(tif.series)}'); tif.close()"
```

If this command fails or reports errors, the file is likely corrupted.

**Possible solutions:**

1. Re-download or re-transfer the file from the original source
2. Verify file integrity using checksums if available
3. Contact the data provider if the file came from an external source
4. Re-export the image from the original acquisition software if possible

### My TIFF/SVS conversion is taking extremely long (hours/days)

Large whole-slide imaging (WSI) files can take significant time to convert.
To optimize the process:

1. **Increase the memory target** to use more of your available RAM:
   ```bash
   # For a system with 64 GB RAM, target ~50 GB
   ngff-zarr --memory-target 50G -i input.svs -o output.ozx
   ```
2. **Validate the file first** to avoid a long run that ends in an error:
   ```bash
   python3 -c "import tifffile; tif = tifffile.TiffFile('input.svs'); print(f'{len(tif.series)} series found'); tif.close()"
   ```
3. **Use SSD storage** for both input and output, as I/O is often the bottleneck

If a conversion runs for a long time and then fails with an error, the source
file is likely corrupted; see the section above for diagnosis and solutions.
