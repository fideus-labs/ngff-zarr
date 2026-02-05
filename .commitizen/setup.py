"""Setup file for cz_ngff_zarr commitizen plugin."""

from setuptools import setup

setup(
    name="cz_ngff_zarr",
    version="1.0.0",
    py_modules=["cz_ngff_zarr"],
    install_requires=["commitizen"],
    entry_points={
        "commitizen.plugin": [
            "cz_ngff_zarr = cz_ngff_zarr:NgffZarrCz",
        ],
    },
)
