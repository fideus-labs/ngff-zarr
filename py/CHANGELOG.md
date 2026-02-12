## py-v0.26.0 (2026-02-12)

### ⚡ Performance

- **py**: wire pyramid reuse into TIFFFILE CLI code path ([48cbc57](https://github.com/fideus-labs/ngff-zarr/commit/48cbc57c34e3213a0e63968e2e128561cc07b47f))
- **py**: reuse existing pyramid levels from pyramidal TIFFs ([9ed34a7](https://github.com/fideus-labs/ngff-zarr/commit/9ed34a7d37acfde094cb75ad90a737b79e93dc5b))

### ✨ Features

- **py**: add scale_strategy parameter to to_ngff_zarr() ([1f146d8](https://github.com/fideus-labs/ngff-zarr/commit/1f146d8d3e84cd25310d0c2e4db95f5dcaf3c946))

### 🐛 Bug Fixes

- **py**: to_multiscale fails with non-prime scale factors ([a9bb0ed](https://github.com/fideus-labs/ngff-zarr/commit/a9bb0edb479243cc73ad67a9740d6b551bfb83cf))
- **py**: remove unused previous_scale_factors variable in _dask_image ([fa32c47](https://github.com/fideus-labs/ngff-zarr/commit/fa32c47f684ff00291db1d0967267184f36dff3c))
- **py**: handle zarr Group from pyramidal TIFFs in to_ngff_image ([85ce8ab](https://github.com/fideus-labs/ngff-zarr/commit/85ce8ab22eb4b84050281eaab6d79b3eafa50dbc))
- **py**: reduce dask task explosion for large tiled TIFF files ([dd1b197](https://github.com/fideus-labs/ngff-zarr/commit/dd1b197296ebb668b7d275d95371359a99674d23))

## py-v0.25.0 (2026-02-08)

### ✨ Features

- **py**: add nibabel NIfTI output support to CLI ([393a0e3](https://github.com/fideus-labs/ngff-zarr/commit/393a0e3087c4313f60b34ac50141e7e0bf250e26))
- **py**: add multi-format output support to CLI ([3aa37fd](https://github.com/fideus-labs/ngff-zarr/commit/3aa37fdd8cfb8703397e48c1a02f42cf2603a3e0))
- **py**: add TIFF series selection and OME metadata extraction ([ba4c088](https://github.com/fideus-labs/ngff-zarr/commit/ba4c088fa2989186481c5d481ff0d5db3ebe6c29))
- add scope-filtered changelogs with GitHub commit links ([2f68564](https://github.com/fideus-labs/ngff-zarr/commit/2f6856470ed894e90fa16d733d560ce0a0560073))

### 🐛 Bug Fixes

- **py**: remove zarr v2 DirectoryStore usage in CLI input handler ([9108543](https://github.com/fideus-labs/ngff-zarr/commit/910854303077c13c49687d04fbed7e41e5a884c6))
- **py**: set PYTHONIOENCODING=utf-8 in CLI output tests for Windows ([9208ce2](https://github.com/fideus-labs/ngff-zarr/commit/9208ce212adc0aa8c400bb1cdb75b198257d41b0))
- restore accidentally modified commented code ([48da920](https://github.com/fideus-labs/ngff-zarr/commit/48da920523988c7be54147a9bcd0f2df95b5a898))
- improve absolute factor tracking and add shape validation ([fd9da2c](https://github.com/fideus-labs/ngff-zarr/commit/fd9da2c447d4b042fb14a1c25a09d9eb35ea3be9))
- preserve non-spatial metadata in dask_image downsampling ([c277f2f](https://github.com/fideus-labs/ngff-zarr/commit/c277f2f734232417d7c77b67dc09136f30c9571e))
- **py**: balance security and usability in series name sanitization ([2deae9d](https://github.com/fideus-labs/ngff-zarr/commit/2deae9d5743cfe0010131991c2ed28d600893770))
- **py**: improve series name sanitization for security ([b55c077](https://github.com/fideus-labs/ngff-zarr/commit/b55c077bdf9ca3418351949523faacd659aeb392))
- **py**: address PR review comments for TIFF series handling ([7efdebe](https://github.com/fideus-labs/ngff-zarr/commit/7efdebeaf7b3e9b47e79a28aa879de131cc18799))
- **py**: handle TIFF S-axis (sample/RGB) and unsupported axes ([750eed5](https://github.com/fideus-labs/ngff-zarr/commit/750eed5bd75497ace33e8bfda074d5291944a5f4))

## py-v0.24.0 (2026-02-05)

### ♻️ Refactoring

- **py**: extract HCS plate detection to helper function ([9a434db](https://github.com/fideus-labs/ngff-zarr/commit/9a434db59870d3ad5743e8faf366cecafaea3408))

### ✨ Features

- **py**: add HCS plate detection to NGFF version detection ([9e3eedc](https://github.com/fideus-labs/ngff-zarr/commit/9e3eedc56dcc541424646cd5d048c297e26fc4ac))

### 🐛 Bug Fixes

- **py**: remove invalid zarr_kwargs from LocalStore instantiation ([d0c0bd4](https://github.com/fideus-labs/ngff-zarr/commit/d0c0bd4fd4cb0c358f565dc9e9b2fc1df23a1827))
- **py**: handle None axes_units in ngff_image_to_itk_image ([363eeda](https://github.com/fideus-labs/ngff-zarr/commit/363eeda7adc6e5b62b92f933c064446c4acbb796))

## py-v0.23.0 (2026-02-03)

### ♻️ Refactoring

- move re imports to module level for better performance ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- deduplicate dask.array.to_zarr ([ffac15c](https://github.com/fideus-labs/ngff-zarr/commit/ffac15c3ca9b0dd3d31e4581063fee4e6d8ba00c))
- refactor zarr_kwargs checks prior to da.array.to_zarr ([d9fa9fc](https://github.com/fideus-labs/ngff-zarr/commit/d9fa9fc371ea1556b98989a6fdf24f47b296f2ce))
- add metadata reading into class methods ([6de9008](https://github.com/fideus-labs/ngff-zarr/commit/6de90083e6c2a5234be46ad7afe9db5bf4a8ae6f))
- version converters for metadata classes ([e775b3e](https://github.com/fideus-labs/ngff-zarr/commit/e775b3eb8f63cab25244f0663e9f9b98a469821d))
- use descriptive CI workflow filenames ([777b70e](https://github.com/fideus-labs/ngff-zarr/commit/777b70e00cd4838cb9f30be4b087c374deac304d))

### ⚡ Performance

- **py**: avoid loading data into memory for omero computation ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))

### ✨ Features

- **ci**: add automated GitHub Release workflow with Commitizen changelog integration ([af9d7c2](https://github.com/fideus-labs/ngff-zarr/commit/af9d7c25004f2cdbde2ba6b4a9672c4882fda620))
- **py,ts**: add OMERO metadata computation from NgffImage ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- add input validation for OMERO metadata computation ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- **py**: jsonFirst flag in the ZIP comment for RFC-9 OZX files ([2d54351](https://github.com/fideus-labs/ngff-zarr/commit/2d5435166b35220ffb7972d16c7e3d49dfa2c01d))
- add Leica Image Format (LIF) support ([3954a1c](https://github.com/fideus-labs/ngff-zarr/commit/3954a1c7e8cd24fccf9e9c8817c499792164cf13))
- add support for dask>=2025.12.0 ([b8ee6ca](https://github.com/fideus-labs/ngff-zarr/commit/b8ee6ca15f296e94d21075169572a302013147a4))
- **hcs**: add helpers and docs for writing .ozx ([58ddb99](https://github.com/fideus-labs/ngff-zarr/commit/58ddb9966c24ba59a8a9735eab40a321eaf5c206))
- **py**: implement RFC-9 Zip file support ([fca333f](https://github.com/fideus-labs/ngff-zarr/commit/fca333fd0398a4774e1a3cc1c01bc8ab0e364081))
- **nibabel**: extract omero metadata from nibabel ([5c83475](https://github.com/fideus-labs/ngff-zarr/commit/5c83475491dd00c5c9ed2f5b30399cefd6eac2ef))
- **nibabel**: extract orientation from nifti header ([c80c7f7](https://github.com/fideus-labs/ngff-zarr/commit/c80c7f767412cc2afa50f245bc3c693dadb6e682))
- **nibabel**: better pixel data typing ([5a24cc0](https://github.com/fideus-labs/ngff-zarr/commit/5a24cc0f5f16c618ef60ffe6546fc26f7b44373d))
- **py**: nibabel io backend ([2360f14](https://github.com/fideus-labs/ngff-zarr/commit/2360f149091833a889c714f0ba97350e77f1a883))
- add write_hcs_well_image ([709d310](https://github.com/fideus-labs/ngff-zarr/commit/709d310d2478b6d46e14dcd0e0db648542186905))
- **py**: add hcs caching support ([c415d2d](https://github.com/fideus-labs/ngff-zarr/commit/c415d2dfe0bf8032ea7f41c99ee6c045c69797de))
- **py**: add high content screening support ([c13deb2](https://github.com/fideus-labs/ngff-zarr/commit/c13deb2e23bfc0e9b1409aab72b7d833834b4070))
- **py**: add RFC-4 validation ([9bd61ed](https://github.com/fideus-labs/ngff-zarr/commit/9bd61edf1af27c2b56c63df7fe56288734b2c178))
- add method metadata support ([e04e39a](https://github.com/fideus-labs/ngff-zarr/commit/e04e39a3cc493bcfa0476870dfa8d3dc1d52b330))
- **py**: add method type metadata ([28ee37b](https://github.com/fideus-labs/ngff-zarr/commit/28ee37be80b853398f0761297304a18ff5ac3c3b))
- provide RAS, LPS convenience mappings ([81746be](https://github.com/fideus-labs/ngff-zarr/commit/81746bee6875c04e1837b5c51172cb728eb347c0))
- **from_ngff_zarr**: add support for storage_options ([12d00f8](https://github.com/fideus-labs/ngff-zarr/commit/12d00f805f5e83f9c048ea9c7a51a3fd4b38eed1))
- **from_ngff_zarr**: add support for storage_options ([d27fd19](https://github.com/fideus-labs/ngff-zarr/commit/d27fd19930e559b10916802d09c4762da9e8d39a))

### 🐛 Bug Fixes

- **py**: handle unknown axis fields in Zarr metadata gracefully ([cb7e619](https://github.com/fideus-labs/ngff-zarr/commit/cb7e61903d359690b9376b39b10f4616f65a214e))
- **py**: better downsampling with many-channel inputs ([a36240a](https://github.com/fideus-labs/ngff-zarr/commit/a36240a5912593e39dad14accf7959de625cbdb8))
- **py**: fix cwd for pixi build-docs command ([301f21f](https://github.com/fideus-labs/ngff-zarr/commit/301f21f5460bd26e89b7d8f7122decfe209ca3d4))
- **py**: omero metadata with OME-Zarr version 0.5 ([225a5db](https://github.com/fideus-labs/ngff-zarr/commit/225a5db985d146467efff23443e8c3082cb7169d))
- **cli**: only process spatial dims with IMAGEIO backend ([d3f32c2](https://github.com/fideus-labs/ngff-zarr/commit/d3f32c285d48973052464f9ead56b285a9ce65d5))
- update dask kwargs ([7f88bf2](https://github.com/fideus-labs/ngff-zarr/commit/7f88bf23271fdcbfc676edd8d68ea24bc8741124))
- improve error message for RFC-9 zipped OME-Zarr version check ([59b3745](https://github.com/fideus-labs/ngff-zarr/commit/59b374591f54c6255a94cf84685f58e41f94ad02))
- prevent wasm imread error on windows ([c1cc8b7](https://github.com/fideus-labs/ngff-zarr/commit/c1cc8b75da0af6750da91f9079f47af0d2689c6b))
- **py**: Remove debug print statements from _itkwasm.py ([a04384a](https://github.com/fideus-labs/ngff-zarr/commit/a04384a66fcf784e38b29061db0c08d9df3e91d1))
- Always use written data for next pyramid level ([0e3465f](https://github.com/fideus-labs/ngff-zarr/commit/0e3465fb372991e5c625cb1aed3f1e7fbb5fa705))
- Always use written data for next pyramid level ([0e3465f](https://github.com/fideus-labs/ngff-zarr/commit/0e3465fb372991e5c625cb1aed3f1e7fbb5fa705))
- **py**: remove duplicate .scale_factors check in to_ngff_zarr ([4cf9181](https://github.com/fideus-labs/ngff-zarr/commit/4cf91817fc57f0bb0b585821845c2a18a6eee0e3))
- **deps**: pin dask to <2025.11.0 ([8402988](https://github.com/fideus-labs/ngff-zarr/commit/8402988650196b1417c5d4f4e7397d57a460f36d))
- incremental downsampling to achieve exact 1x, 2x, 3x, 4x sizes ([a1a00cd](https://github.com/fideus-labs/ngff-zarr/commit/a1a00cd61ea7ee73d1de55a09f9a5b6411547a71))
- **hcs**: better 0.5 metadata support ([0008055](https://github.com/fideus-labs/ngff-zarr/commit/0008055d7e4862b579d4c666d06744df46dace3f))
- **cli**: default to OME-Zarr 0.5 for .ozx ([b321cdb](https://github.com/fideus-labs/ngff-zarr/commit/b321cdb0c2c6fc92f36f12f3dc64a5b3b3180c21))
- **py**: improve sharding chunk shape ([04517a2](https://github.com/fideus-labs/ngff-zarr/commit/04517a228e99f2a0a87611c0204ea9fd25d55e1d))
- **py**: improved chunk sizes for dask/zarr ([0b4c7ca](https://github.com/fideus-labs/ngff-zarr/commit/0b4c7cab0e6ab8012b8d075233f932d2eae0af24))
- **py**: multi-channel / component gaussian downsampling ([77ea912](https://github.com/fideus-labs/ngff-zarr/commit/77ea9126f59ba89597be36802384a1b06418daef))
- **py**: update testing data hashes for nifti image input ([18424b3](https://github.com/fideus-labs/ngff-zarr/commit/18424b3889d7b621225f10d19ea807c4a80d9380))
- **nibabel**: add 'c' scale_dict translation_dict values ([242ae66](https://github.com/fideus-labs/ngff-zarr/commit/242ae66279ae75bc57c807a2608f8fd3577f7fd6))
- **nibabel**: remove zero shear criteria to add anatomical orientation ([4cfff56](https://github.com/fideus-labs/ngff-zarr/commit/4cfff56d65caff4dd268bde3617d5c50c37570a8))
- **cli**: add check that input and output dirs are not the same ([5553ff9](https://github.com/fideus-labs/ngff-zarr/commit/5553ff92e36cf87539fe70a80199d156103bc4f9))
- **hcs**: address writing to sparse plate layouts ([777fad0](https://github.com/fideus-labs/ngff-zarr/commit/777fad0823e4473abe5c516b94e49c6468845355))
- **hcs.py**: use zarr format 2 with NGFF version 0.4 ([b01bb30](https://github.com/fideus-labs/ngff-zarr/commit/b01bb30ba4eb83b9b21bd9b69111bca7e2ff430f))
- **hcs.py**: add missing to_ngff_zarr import ([c4bd422](https://github.com/fideus-labs/ngff-zarr/commit/c4bd422f257bb3a474dff567239c20b316ad4ef1))
- **rfc4**: correct LPS RAS direction ([0c00023](https://github.com/fideus-labs/ngff-zarr/commit/0c000233cc28d8c4d45f4a62ce2ca8c4587d05bd))
- **py**: lps uses superior-to-inferior ([a9ef1c4](https://github.com/fideus-labs/ngff-zarr/commit/a9ef1c457058c6178a14f3b313321eb9d4c5fb2d))
- **py**: sharding, compression codec, tensorstore ([a33c333](https://github.com/fideus-labs/ngff-zarr/commit/a33c333dcad97afd62cc6a8e15e4ba482608b05d))
- **py**: use _handle_large_array_writing without scale_factors ([3df833c](https://github.com/fideus-labs/ngff-zarr/commit/3df833c3cc6a8fdced9076a897b7ed36425d126a))
- **py**: support compressor kwargs with _write_with_tensorstore ([d0d15ee](https://github.com/fideus-labs/ngff-zarr/commit/d0d15ee2c9569dc6f9e7aa96563f7b7c7ee8a3f8))
- **from_ngff_zarr**: handle invalid OMERO metadata ([e32dbe5](https://github.com/fideus-labs/ngff-zarr/commit/e32dbe5d59e6116f9d32353bf356d19d691f726f))
- **rfc.py**: add missing values ([6f7ffca](https://github.com/fideus-labs/ngff-zarr/commit/6f7ffca93ee792d2d9da77a11072012f33d6f887))
