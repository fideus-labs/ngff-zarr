#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Test script to test tensorstore compression directly."""

import tempfile
import sys
from pathlib import Path
from ngff_zarr import to_ngff_zarr, cli_input_to_ngff_image, detect_cli_io_backend
import numcodecs


def test_tensorstore_compression():
    if sys.platform == 'win32':
        print('Skipping tensorstore test on Windows')
        return

    test_input_file = Path('../mcp/test/data/input/MR-head.nrrd')
    if test_input_file.exists():
        backend = detect_cli_io_backend([str(test_input_file)])
        ngff_image = cli_input_to_ngff_image(backend, [str(test_input_file)])

        # Need to convert to multiscales first
        from ngff_zarr import to_multiscales
        multiscales = to_multiscales(ngff_image)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f'{temp_dir}/test.ome.zarr'
            print('Testing compression with tensorstore...')

            try:
                compressor = numcodecs.Blosc(cname='lz4', clevel=5)
                to_ngff_zarr(
                    output_path,
                    multiscales,
                    version='0.4',
                    use_tensorstore=True,
                    compressor=compressor
                )
                print('Success!')
            except Exception as e:
                print(f'Error: {e}')
                import traceback
                traceback.print_exc()
    else:
        print('Test file not found')


if __name__ == "__main__":
    test_tensorstore_compression()
