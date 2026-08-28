## mcp-v0.14.0 (2026-08-28)

### 🐛 Bug Fixes

- **py,ts**: gate the RFC-4 orientation rules on 0.9.dev1 ([7205c2e](https://github.com/fideus-labs/ngff-zarr/commit/7205c2e85e84f8bb1f07fa5a188b7440634e9846))

## mcp-v0.13.0 (2026-08-27)

### BREAKING CHANGE

- the old names are removed rather than kept as aliases. ([d3e48f5](https://github.com/fideus-labs/ngff-zarr/commit/d3e48f53500d1f0726ff3e4334d03f231a58b62b))
- `ByDimensionItem.input_axes` and `.output_axes` are now
`inputAxes` and `outputAxes`, in the Python dataclass and the TypeScript
interface. ([65a0e10](https://github.com/fideus-labs/ngff-zarr/commit/65a0e103b234b6138d0d5759802ae4b4c9efb007))
- ngff-zarr now requires Python >= 3.11. Python 3.10
users should remain on the previous release series. ([6ea8c96](https://github.com/fideus-labs/ngff-zarr/commit/6ea8c96802a593ad52edcb1b73dad251c5543ccf))
- installing ngff-zarr no longer installs zarr-python.
Code that imported zarr relying on ngff-zarr's dependency tree must
declare it explicitly. The remote extra no longer ships the fsspec
backends (fsspec, aiohttp, requests, s3fs, gcsfs, adlfs); remote URL
targets for writes are not supported. ([9f76343](https://github.com/fideus-labs/ngff-zarr/commit/9f76343dbb43a377cee16e4eb9db7553c59f0c53))

### ♻️ Refactoring

- **py,ts**: name the resamplers after what they take ([d3e48f5](https://github.com/fideus-labs/ngff-zarr/commit/d3e48f53500d1f0726ff3e4334d03f231a58b62b))

### ✨ Features

- **py,ts**: support the RFC-5 projectAxis transformation ([367eb06](https://github.com/fideus-labs/ngff-zarr/commit/367eb069674e4fa6588474ca2387c3d61dcb0d23))
- **py,ts**: convert every RFC-5 transformation type to ITK ([1d4f063](https://github.com/fideus-labs/ngff-zarr/commit/1d4f063f09b6b1704500137be37651fddbd6752b))
- **py,ts**: exact frame conversion and hardened ITK-Wasm decoding ([444c879](https://github.com/fideus-labs/ngff-zarr/commit/444c879c881b2c8f248838125100dff12b00ecbe))
- convert coordinate transformations between RFC-5 and ITK ([16ec87b](https://github.com/fideus-labs/ngff-zarr/commit/16ec87bd0348ba8eec08df0c736ab6d7833e21db))
- **py,ts**: offer 0.9.dev1 as an upgrade target and document it ([c256ea3](https://github.com/fideus-labs/ngff-zarr/commit/c256ea3af4816a09c9bcfdf17fccf3403187d184))
- **py,ts**: add the OME-Zarr 0.9.dev1 version and its metadata model ([264c8bd](https://github.com/fideus-labs/ngff-zarr/commit/264c8bdd46a5caa8c7a515446f509998e9dc8f16))
- **py,ts**: track the OME-Zarr 0.6rc0 schemas and version tag ([69b6e14](https://github.com/fideus-labs/ngff-zarr/commit/69b6e14c64c0338731177f62ec53380482e3e833))
- **mcp**: deprecate use_tensorstore and drop runtime zarr-python usage ([41c2cdc](https://github.com/fideus-labs/ngff-zarr/commit/41c2cdcdf0e3c16224b87e2a477bf3400eade968))

### 🐛 Bug Fixes

- **py,ts**: refuse to write a transform the reader would reject ([36d3e92](https://github.com/fideus-labs/ngff-zarr/commit/36d3e924c08de5c697900a9d1eb522dd8a31539c))
- **py,ts**: hold projectAxis to the arity its schema declares ([71d52e7](https://github.com/fideus-labs/ngff-zarr/commit/71d52e7ef3d12a90f9f754d75a1f873fbe40555d))
- **py,ts**: refuse the conversions that returned a wrong matrix silently ([967e438](https://github.com/fideus-labs/ngff-zarr/commit/967e4387c1dc4b1c21c5db30c116b2a8c54372f9))
- **py,ts**: follow the byDimension axis key rename ([945b590](https://github.com/fideus-labs/ngff-zarr/commit/945b590ab9e8746ff0b5618e26a326a7d0aad003))
- **py,ts**: recover an ITK affine exactly and bind sub-grid axes by name ([f756056](https://github.com/fideus-labs/ngff-zarr/commit/f756056f6bc3e1e6157ce61981484ef86b787034))
- **py,ts**: stop applying the v0.6 mapAxis arity to a 0.9.dev1 store ([ed4aee4](https://github.com/fideus-labs/ngff-zarr/commit/ed4aee4d8d0f9893d879686a6ec6a633450164ee))
- **py,ts**: carry the v0.6 transform rules over to 0.9.dev1 ([3393a2d](https://github.com/fideus-labs/ngff-zarr/commit/3393a2dda5761f4675095c0d89aa4c1a0dcfecfa))
- **py,ts**: gate the axes each version reads and writes ([a022dca](https://github.com/fideus-labs/ngff-zarr/commit/a022dca968a7f7fc13886e374e92df58d5f44ba5))
- **py,ts**: correct the axis model and gate the RFC-3 rules by version ([e418a39](https://github.com/fideus-labs/ngff-zarr/commit/e418a39dbb0f278a8eaaadab1d7d71e845515fa6))
- **py,ts**: write byDimension and top-level transforms the 0.6rc0 schema accepts ([65a0e10](https://github.com/fideus-labs/ngff-zarr/commit/65a0e103b234b6138d0d5759802ae4b4c9efb007))

## mcp-v0.12.0 (2026-08-21)

### ♻️ Refactoring

- **py,ts**: keep the RFC-4 vocabulary in one place per port ([02afb3f](https://github.com/fideus-labs/ngff-zarr/commit/02afb3f900ceb01b315f0e8bd2b2caeb99b68ecf))

### ✨ Features

- **mcp**: support stateless Streamable HTTP transport ([034d804](https://github.com/fideus-labs/ngff-zarr/commit/034d804911477a96a954bee4ac6c1fe94a3216c5))

### 🐛 Bug Fixes

- **py,ts**: reject a non-string orientation value and pin the schema both ways ([81b095d](https://github.com/fideus-labs/ngff-zarr/commit/81b095ddd75d02d2811659091854e6f14bfee5ed))
- fix ci PyPI publish failure on metadata 2.5 wheels ([90c5bee](https://github.com/fideus-labs/ngff-zarr/commit/90c5bee484e8f128023a4cf7da71d2d320db9532))
- RFC 4 orientation is optional per axis, null equals absent, type must be anatomical ([f4bf3dc](https://github.com/fideus-labs/ngff-zarr/commit/f4bf3dcb4446678e17135a95305b379c3fb9fe0a))
- **ci**: use supported 'prek update' in autoupdate workflow ([8f2ee83](https://github.com/fideus-labs/ngff-zarr/commit/8f2ee83af4af48f096fc0272abdeb303646c3768))
- reject RFC 4 orientation on non-spatial axes and duplicate anatomical axes ([45de14c](https://github.com/fideus-labs/ngff-zarr/commit/45de14c0ed62a0f0d68d56790f1a71482cb9c221))

## mcp-v0.11.0 (2026-07-17)

### ✨ Features

- **py,ts**: write an image and its transformation into one store ([be3f6cd](https://github.com/fideus-labs/ngff-zarr/commit/be3f6cd54d622559fac1049e0c8ea6194e7c1ef4))

## mcp-v0.10.0 (2026-07-09)

### ♻️ Refactoring

- **py,ts**: move v0.6 axis types to v06 and set axes_types on NgffImage ([54eaed4](https://github.com/fideus-labs/ngff-zarr/commit/54eaed4d2e4b23de93f68a3c15723fc14893244e))

### ✨ Features

- **py,ts**: carry the v0.6 discrete axis flag and address review feedback ([5203864](https://github.com/fideus-labs/ngff-zarr/commit/5203864efefcd1ba38b42c3e4976c542ae984fcf))
- **py,ts**: support OME-Zarr v0.6 displacement and coordinate fields ([c181cc1](https://github.com/fideus-labs/ngff-zarr/commit/c181cc189edc142f2add7f0df3a3ad865421f0f8))

## mcp-v0.9.0 (2026-07-03)

### BREAKING CHANGE

- The `enabled_rfcs` parameter (Python `to_ngff_zarr`/
`to_ome_zarr`), the `--enable-rfc` CLI flag, the `enabledRfcs` TypeScript write
option, and the MCP `enable_rfc4`/`enabled_rfcs` fields are removed. RFC-4
anatomical orientation is written automatically when orientation metadata is
present; supply it via `NgffImage.axes_orientations`, the new `orientation=`
argument to `to_multiscales`, or the `--orientation` CLI flag. ([abda351](https://github.com/fideus-labs/ngff-zarr/commit/abda351de31305f87a6119102e27b55e6baaecdb))

### ♻️ Refactoring

- **py,ts**: write RFC-4 anatomical orientation automatically, drop enabled-rfcs API ([abda351](https://github.com/fideus-labs/ngff-zarr/commit/abda351de31305f87a6119102e27b55e6baaecdb))

### ✨ Features

- **py,ts**: add OME-Zarr v0.5 structural validation and schema support ([598ed91](https://github.com/fideus-labs/ngff-zarr/commit/598ed91e6f3b15533d9a50a59b8cd68b3674e22f))
- **py,ts**: extend RFC-4 vocabulary with subject-local tissue orientations ([0cc4a91](https://github.com/fideus-labs/ngff-zarr/commit/0cc4a9146fb9113d0a7c289f7787abdd825e362d))

### 🐛 Bug Fixes

- **mcp**: add py.typed marker and fix type errors ([3bb65ec](https://github.com/fideus-labs/ngff-zarr/commit/3bb65ec1f5f95a641866a32d06a0bcc35816eb7d))
- **mcp**: cover import-untyped in s3fs type-ignore ([278b3e8](https://github.com/fideus-labs/ngff-zarr/commit/278b3e88ecfddf703ad40e6546ef410acceb4fac))
- **ci**: remove OTLP import from daily-doc-updater workflow ([45394c8](https://github.com/fideus-labs/ngff-zarr/commit/45394c8ccb09b5ad3687d8dd834c94af45d69469))
- stop pre-commit from mangling gh-aw generated files ([d6f1201](https://github.com/fideus-labs/ngff-zarr/commit/d6f120144884bc6a3089e5286a624ed0316f90ba))

## mcp-v0.8.0 (2026-04-15)

### ♻️ Refactoring

- rename `Multiscales` to `NgffMultiscales` across py, ts, mcp ([1d0c572](https://github.com/fideus-labs/ngff-zarr/commit/1d0c5724fdd7ba86679f57b3343332a44d1aff5d))

### ✨ Features

- **py,ts**: rename from_ngff_zarr/fromNgffZarr to from_ome_zarr/fromOmeZarr ([84a20b2](https://github.com/fideus-labs/ngff-zarr/commit/84a20b28bce135e4009197b7671fa9ad21e772a0))
- **py,ts**: rename to_ngff_zarr/toNgffZarr to to_ome_zarr/toOmeZarr ([9431546](https://github.com/fideus-labs/ngff-zarr/commit/94315466d03b7c3e73a3e42fc7e236c428f1d95c))
- update default OME-Zarr version from 0.4 to 0.5 ([4e60c07](https://github.com/fideus-labs/ngff-zarr/commit/4e60c077b3adcf6f4ef827a1cc112ca8e340f22c))
- **py,ts**: add bidirectional anatomical orientation ↔ ITK direction matrix conversion ([e427ca4](https://github.com/fideus-labs/ngff-zarr/commit/e427ca4e5e131a97b9b566a5fd79ac6020110dd9))

### 🐛 Bug Fixes

- resolve merge conflicts with main branch ([af2a8eb](https://github.com/fideus-labs/ngff-zarr/commit/af2a8eb4c5e4aa995d7afd79e92a574549db3dfc))

## mcp-v0.7.0 (2026-03-20)

### ✨ Features

- update default OME-Zarr version from 0.4 to 0.5 ([4e60c07](https://github.com/fideus-labs/ngff-zarr/commit/4e60c077b3adcf6f4ef827a1cc112ca8e340f22c))
- **py,ts**: add bidirectional anatomical orientation ↔ ITK direction matrix conversion ([e427ca4](https://github.com/fideus-labs/ngff-zarr/commit/e427ca4e5e131a97b9b566a5fd79ac6020110dd9))

### 🐛 Bug Fixes

- resolve merge conflicts with main branch ([af2a8eb](https://github.com/fideus-labs/ngff-zarr/commit/af2a8eb4c5e4aa995d7afd79e92a574549db3dfc))

## mcp-v0.6.0 (2026-03-04)

### ♻️ Refactoring

- move re imports to module level for better performance ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- use descriptive CI workflow filenames ([777b70e](https://github.com/fideus-labs/ngff-zarr/commit/777b70e00cd4838cb9f30be4b087c374deac304d))

### ✨ Features

- **py,ts**: relax well image path constraints per ome/ngff-spec#71 ([32b7723](https://github.com/fideus-labs/ngff-zarr/commit/32b7723a79698272dea694348f24505ff8efa6d9))
- add scope-filtered changelogs with GitHub commit links ([2f68564](https://github.com/fideus-labs/ngff-zarr/commit/2f6856470ed894e90fa16d733d560ce0a0560073))
- **ci**: add automated GitHub Release workflow with Commitizen changelog integration ([af9d7c2](https://github.com/fideus-labs/ngff-zarr/commit/af9d7c25004f2cdbde2ba6b4a9672c4882fda620))
- **py,ts**: add OMERO metadata computation from NgffImage ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- add input validation for OMERO metadata computation ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- add Leica Image Format (LIF) support ([3954a1c](https://github.com/fideus-labs/ngff-zarr/commit/3954a1c7e8cd24fccf9e9c8817c499792164cf13))
- add support for dask>=2025.12.0 ([b8ee6ca](https://github.com/fideus-labs/ngff-zarr/commit/b8ee6ca15f296e94d21075169572a302013147a4))
- add write_hcs_well_image ([709d310](https://github.com/fideus-labs/ngff-zarr/commit/709d310d2478b6d46e14dcd0e0db648542186905))
- **mcp**: populate method_type, method_metadata ([7a54145](https://github.com/fideus-labs/ngff-zarr/commit/7a541458fcd53f2c65b6702e9b386cb766f97b72))
- **mcp**: convert_to_ome_zarr support for store inputs ([68fb972](https://github.com/fideus-labs/ngff-zarr/commit/68fb972a658a24add46ddaeb4880a920f5085b67))
- **mcp**: updates for ngff-zarr 0.14, 0.15 features ([f9e2c60](https://github.com/fideus-labs/ngff-zarr/commit/f9e2c60a6295fc913d885febf2a8e135a4afb528))
- provide RAS, LPS convenience mappings ([81746be](https://github.com/fideus-labs/ngff-zarr/commit/81746bee6875c04e1837b5c51172cb728eb347c0))
- **from_ngff_zarr**: add support for storage_options ([12d00f8](https://github.com/fideus-labs/ngff-zarr/commit/12d00f805f5e83f9c048ea9c7a51a3fd4b38eed1))
- **from_ngff_zarr**: add support for storage_options ([d27fd19](https://github.com/fideus-labs/ngff-zarr/commit/d27fd19930e559b10916802d09c4762da9e8d39a))

### 🐛 Bug Fixes

- **ci**: avoid duplicate pixi binary install in release workflow ([07dbd36](https://github.com/fideus-labs/ngff-zarr/commit/07dbd3642a3dc44fb06f48523fcf0221d1d00607))
- **mcp**: add ngff_zarr to mypy ignore_missing_imports overrides ([9923409](https://github.com/fideus-labs/ngff-zarr/commit/9923409e9c6e01f6b5629f142d53f7d31dc38942))
- **ci**: correct new contributor detection and filter bots ([836992d](https://github.com/fideus-labs/ngff-zarr/commit/836992dd0094aaf0c8036200e7de606a628094da))
- **ci**: guard printf against names starting with dash ([7ef5d40](https://github.com/fideus-labs/ngff-zarr/commit/7ef5d406a6c6c8495e7b0652cedb8ec2656d82dc))
- **deps**: pin dask to <2025.11.0 ([8402988](https://github.com/fideus-labs/ngff-zarr/commit/8402988650196b1417c5d4f4e7397d57a460f36d))
- incremental downsampling to achieve exact 1x, 2x, 3x, 4x sizes ([a1a00cd](https://github.com/fideus-labs/ngff-zarr/commit/a1a00cd61ea7ee73d1de55a09f9a5b6411547a71))
- **rfc4**: correct LPS RAS direction ([0c00023](https://github.com/fideus-labs/ngff-zarr/commit/0c000233cc28d8c4d45f4a62ce2ca8c4587d05bd))
- **mcp**: remove memory_target option ([a846e4c](https://github.com/fideus-labs/ngff-zarr/commit/a846e4c5813762e8aa7a6680f8da996406005c6c))
- **mcp**: add ngff-zarr tensorstore dep ([4746645](https://github.com/fideus-labs/ngff-zarr/commit/4746645d8468db9c7836b58fb4c0398d1c2294ec))
- **mcp**: fix shadowing of zarr module in utils.py ([c709501](https://github.com/fideus-labs/ngff-zarr/commit/c709501e4316faf4c2d55d650ef7e96f9649bf2e))
- **mcp**: workaround dandi omero compatibility ([317c982](https://github.com/fideus-labs/ngff-zarr/commit/317c982fb455d81d0b397ec8c583bfa1a9c9e037))
- **rfc.py**: add missing values ([6f7ffca](https://github.com/fideus-labs/ngff-zarr/commit/6f7ffca93ee792d2d9da77a11072012f33d6f887))
