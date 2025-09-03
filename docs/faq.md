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

## Naming

### Why is the project called "fizarr"? What happened to "ngff-zarr"?

**fizarr is the artist formerly known as ngff-zarr** 👑

The project was renamed from `ngff-zarr` to `fizarr` to avoid confusion between the broader Next Generation File Format (NGFF) specification and the specific OME-Zarr file format implementation.

While NGFF is a community-driven specification that encompasses various file formats and data models (see [ngff.openmicroscopy.org](https://ngff.openmicroscopy.org/)), fizarr is specifically designed to provide tooling for the **OME-Zarr file format** that emerges from the NGFF community's work.

The name "fizarr" reflects its curation by [fideus labs](https://fideus.io), whose mission is to foster trust and advance understanding from scientific and biomedical images. This clearer naming helps users understand that:

- **NGFF** = The broader specification and community
- **OME-Zarr** = The specific zarr-based file format for bioimaging data
- **fizarr** = The lean and kind implementation for working with OME-Zarr files

All functionality remains the same - fizarr continues to be a minimal-dependency, web-ready, and performant library for reading and writing OME-Zarr files, now with a name that better reflects its specific purpose and maintainership.
