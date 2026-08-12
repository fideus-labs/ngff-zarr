# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from unittest.mock import Mock

import pytest
from ngff_zarr import cli


@pytest.mark.parametrize(
    ("uri", "expected_storage_options"),
    [
        (
            "s3://aind-scratch-data/exaspim-ccfv4-template-dev/nifti-to-omezarr/"
            "Registration_EXASPIM_delivery_ExaSPIM_template_10specimens_30um."
            "ome.zarr",
            {"anon": True},
        ),
        ("https://example.com/image.ome.zarr", None),
    ],
    ids=("anonymous-s3", "default-https"),
)
def test_cli_configures_remote_storage_options(
    monkeypatch, uri, expected_storage_options
):
    """Only S3 CLI inputs opt into anonymous remote access."""
    multiscales = object()
    from_ome_zarr = Mock(return_value=multiscales)
    render = Mock()
    monkeypatch.setattr(cli, "from_ome_zarr", from_ome_zarr)
    monkeypatch.setattr(cli, "_multiscales_to_ngff_zarr", render)

    cli.main(["--quiet", "-i", uri])

    from_ome_zarr.assert_called_once_with(uri, storage_options=expected_storage_options)
    assert render.call_args.args[4] is multiscales
