# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""consolidate_metadata can skip consolidation.

Consolidated metadata is incompatible with backends that run their own
consolidation, such as Icechunk, so ``to_ome_zarr`` must be able to skip it.
"""

import numpy as np
import pytest
import zarr
from ngff_zarr import from_ome_zarr, to_multiscales, to_ngff_image, to_ome_zarr
from ngff_zarr._zarrista_utils import has_consolidated_metadata
from packaging import version

needs_zarr_v3 = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.0.0b1"),
    reason="zarr version < 3.0.0b1",
)
VERSIONS = ["0.4", pytest.param("0.5", marks=needs_zarr_v3)]


@pytest.mark.parametrize("ngff_version", VERSIONS)
def test_consolidate_metadata_toggle(tmp_path, ngff_version):
    zarr_format = 2 if ngff_version == "0.4" else 3
    image = to_ngff_image(
        np.arange(64 * 64, dtype=np.uint8).reshape(64, 64), dims=["y", "x"]
    )
    multiscales = to_multiscales(image, scale_factors=[2], cache=False)

    default = str(tmp_path / "default.ome.zarr")
    disabled = str(tmp_path / "disabled.ome.zarr")
    to_ome_zarr(default, multiscales, version=ngff_version)
    to_ome_zarr(disabled, multiscales, version=ngff_version, consolidate_metadata=False)

    assert has_consolidated_metadata(default, zarr_format)
    assert not has_consolidated_metadata(disabled, zarr_format)


def _consolidated_level0(path, ngff_version):
    data = np.random.default_rng(0).random((1, 32, 32, 32)).astype(np.float32)
    image = to_ngff_image(data, dims=["c", "z", "y", "x"])
    to_ome_zarr(
        path, to_multiscales(image, scale_factors=[], cache=False), version=ngff_version
    )
    pyramid = to_multiscales(
        from_ome_zarr(path).images[0], scale_factors=[2], cache=False
    )
    return pyramid


# A zarr v2 in-place write cannot drop the pre-existing .zmetadata sidecar, so
# skipping consolidation would leave it stale. This holds for an append
# (start_level) and a plain overwrite=False rewrite alike.
@pytest.mark.parametrize("extra", [{"start_level": 1}, {}], ids=["append", "in_place"])
def test_disabling_on_in_place_v2_write_raises(tmp_path, extra):
    path = str(tmp_path / "image.ome.zarr")
    pyramid = _consolidated_level0(path, "0.4")
    with pytest.raises(ValueError, match="would be left stale"):
        to_ome_zarr(
            path,
            pyramid,
            version="0.4",
            overwrite=False,
            consolidate_metadata=False,
            **extra,
        )


# A zarr v3 in-place write rewrites the root document, dropping its consolidated
# block, so the same append instead succeeds and leaves the store unconsolidated.
@needs_zarr_v3
def test_disabling_on_in_place_v3_append_self_cleans(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    pyramid = _consolidated_level0(path, "0.5")
    to_ome_zarr(
        path,
        pyramid,
        version="0.5",
        overwrite=False,
        start_level=1,
        consolidate_metadata=False,
    )

    assert not has_consolidated_metadata(path, 3)
    assert len(from_ome_zarr(path).images) == 2
