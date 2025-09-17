<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# 🔨 Development

Contributions are welcome and appreciated!

We are glad you are here and appreciate your contribution. Please keep in mind
our
[community participation guidelines](https://github.com/InsightSoftwareConsortium/ITK/blob/master/CODE_OF_CONDUCT.md).

## Get the source code

```shell
git clone https://github.com/thewtex/ngff-zarr
cd ngff-zarr
```

## Install dependencies

First install [pixi]. Then, install project dependencies:

```shell
pixi install -a
pixi run pre-commit-install
```

## Run the test suite

```shell
pixi run test
```

## Build the documentation

If needed, build and update the documentation:

```shell
pixi run dev-docs
```

## Update test data

If needed, update the testing data.

1. Generate new test data tarball and compute its sha256 hash

```shell
pixi run hash-data
```

2. Upload the data to [web3.storage](https://web3.storage)

3. Upload the `test_data_ipfs_cid` (from web3.storage web UI) and
   `test_data_sha256` (`sh256sum ../data.tar.gz`) variables in _test/\_data.py_.

## Submit the patch

We use the standard [GitHub flow].

[pixi]: https://pixi.sh
[GitHub flow]: https://docs.github.com/en/get-started/using-github/github-flow
