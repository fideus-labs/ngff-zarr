# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from unittest.mock import Mock

import pytest
from ngff_zarr import cli

S3_URI = (
    "s3://aind-scratch-data/exaspim-ccfv4-template-dev/nifti-to-omezarr/"
    "Registration_EXASPIM_delivery_ExaSPIM_template_10specimens_30um."
    "ome.zarr"
)


@pytest.mark.parametrize(
    ("argv", "expected_input", "expected_storage_options"),
    [
        pytest.param(
            ["-i", S3_URI],
            S3_URI,
            {"anon": True},
            id="anonymous-s3",
        ),
        pytest.param(
            ["-i", S3_URI.replace("s3://", "S3://")],
            S3_URI,
            {"anon": True},
            id="normalized-s3-scheme",
        ),
        pytest.param(
            [
                "--storage-options",
                '{"anon": false, "requester_pays": true}',
                "-i",
                S3_URI,
            ],
            S3_URI,
            {"anon": False, "requester_pays": True},
            id="authenticated-requester-pays-s3",
        ),
        pytest.param(
            ["-i", "https://example.com/image.ome.zarr"],
            "https://example.com/image.ome.zarr",
            None,
            id="default-https",
        ),
    ],
)
@pytest.mark.skipif(
    not hasattr(cli.zarr.storage, "FsspecStore"),
    reason="zarr-python 3 storage options path",
)
def test_cli_configures_remote_storage_options(
    monkeypatch, argv, expected_input, expected_storage_options
):
    """S3 defaults to anonymous access but accepts explicit fsspec options."""
    multiscales = object()
    from_ome_zarr = Mock(return_value=multiscales)
    render = Mock()
    monkeypatch.setattr(cli, "from_ome_zarr", from_ome_zarr)
    monkeypatch.setattr(cli, "_multiscales_to_ngff_zarr", render)

    cli.main(["--quiet", *argv])

    from_ome_zarr.assert_called_once_with(
        expected_input, storage_options=expected_storage_options
    )
    assert render.call_args.args[4] is multiscales


@pytest.mark.parametrize(
    ("extra_args", "expected_storage_options"),
    [
        pytest.param([], {"anon": True}, id="anonymous-s3"),
        pytest.param(
            [
                "--storage-options",
                '{"anon": false, "profile": "research"}',
            ],
            {"anon": False, "profile": "research"},
            id="authenticated-s3",
        ),
    ],
)
def test_cli_configures_zarr2_s3_store(
    monkeypatch, extra_args, expected_storage_options
):
    """Zarr 2 receives S3 options through FSStore instead of the v3-only API."""
    multiscales = object()
    legacy_store = object()
    from_ome_zarr = Mock(return_value=multiscales)
    fs_store = Mock(return_value=legacy_store)
    render = Mock()
    monkeypatch.delattr(cli.zarr.storage, "FsspecStore", raising=False)
    monkeypatch.setattr(cli.zarr.storage, "FSStore", fs_store, raising=False)
    monkeypatch.setattr(cli, "from_ome_zarr", from_ome_zarr)
    monkeypatch.setattr(cli, "_multiscales_to_ngff_zarr", render)

    cli.main(["--quiet", "-i", S3_URI, *extra_args])

    fs_store.assert_called_once_with(S3_URI, mode="r", **expected_storage_options)
    from_ome_zarr.assert_called_once_with(legacy_store, storage_options=None)
    assert render.call_args.args[4] is multiscales


@pytest.mark.parametrize("storage_options", ["not-json", "[]"])
def test_cli_rejects_invalid_storage_options(storage_options, capsys):
    """Storage options must decode to a JSON object."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--quiet",
                "-i",
                S3_URI,
                "--storage-options",
                storage_options,
            ]
        )

    assert "storage options must be" in capsys.readouterr().err
