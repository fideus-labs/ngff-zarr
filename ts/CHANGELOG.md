## ts-v0.7.3 (2026-02-09)

### 🐛 Bug Fixes

- **ts**: fizarrita bump for more reliable shape estimation ([283c925](https://github.com/fideus-labs/ngff-zarr/commit/283c925caf6350655e7b4a40647bb55b4ab4e21d))

## ts-v0.7.2 (2026-02-08)

### 🐛 Bug Fixes

- **ts**: bump fizarrita for better chunk shape detection ([261da2f](https://github.com/fideus-labs/ngff-zarr/commit/261da2fbd2dd0c101db9604c69a8e08ed66db4c6))

## ts-v0.7.1 (2026-02-08)

### 🐛 Bug Fixes

- **ts**: bump fizarrita for edge chunk handling ([ad84810](https://github.com/fideus-labs/ngff-zarr/commit/ad8481021552b26b8e06d755e574aa15ca54a774))

## ts-v0.7.0 (2026-02-08)

### ✨ Features

- **ts**: export fizarrita worker pools ([3310bc8](https://github.com/fideus-labs/ngff-zarr/commit/3310bc85acc70826b0e92ae639abf824e9332703))

## ts-v0.6.0 (2026-02-08)

### ⚡ Performance

- **ts**: integrate fizarrita worker pool for parallel zarr codec operations ([26dca7e](https://github.com/fideus-labs/ngff-zarr/commit/26dca7e89d5c43ae1302ff68cf9b1e5610ede2f6))

### ✨ Features

- add scope-filtered changelogs with GitHub commit links ([2f68564](https://github.com/fideus-labs/ngff-zarr/commit/2f6856470ed894e90fa16d733d560ce0a0560073))

### 🐛 Bug Fixes

- **ts**: add bytes+blosc codecs to all zarr.create() calls ([d822eee](https://github.com/fideus-labs/ngff-zarr/commit/d822eeefd6f80c8ee1faf8bb2c086c0218a809a0))

## ts-v0.5.1 (2026-02-05)

### 🐛 Bug Fixes

- **ts**: support exactOptionalPropertyTypes for OMERO options

## ts-v0.5.0 (2026-02-05)

### ✨ Features

- add javascript omero computation benchmark ([28c4a32](https://github.com/fideus-labs/ngff-zarr/commit/28c4a32f2ef76acca99e3f16c6e27e7519b4a2a6))
- **ts**: add WebWorker-based OMERO computation with Comlink ([82f39a4](https://github.com/fideus-labs/ngff-zarr/commit/82f39a4a8d93c06eb338c8fd117cd7f6bf9ebaee))
- **ci**: add automated GitHub Release workflow with Commitizen changelog integration ([af9d7c2](https://github.com/fideus-labs/ngff-zarr/commit/af9d7c25004f2cdbde2ba6b4a9672c4882fda620))

### 🐛 Bug Fixes

- **ts**: export dataTypeToComponentType in the package ([e5b2d53](https://github.com/fideus-labs/ngff-zarr/commit/e5b2d5325d6f2af2c28d8b2c6cc1d012f30f241b))
- **ts**: respect requested chunk size ([58fd0af](https://github.com/fideus-labs/ngff-zarr/commit/58fd0affde391103d6d5bfd583a6eee8c8492c1b))
- **ts**: export itkImageToNgffImage from package ([c93b311](https://github.com/fideus-labs/ngff-zarr/commit/c93b31172c70761750c4acfee877f4e7859e1d99))

## ts-v0.4.0 (2026-02-03)

### ♻️ Refactoring

- move re imports to module level for better performance ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- **ts**: switch to glasbey_hv for omero colors ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- **ts**: metadata extract per Python PR #250 ([178dd11](https://github.com/fideus-labs/ngff-zarr/commit/178dd11a3a23728e3918bf85bf02fdee6b109da6))
- **ts**: factor out calculateIncrementalFactor ([848580e](https://github.com/fideus-labs/ngff-zarr/commit/848580eadf3e33c17bb6d33e87ce533a04b81047))
- remove LazyArray ([67c9b4b](https://github.com/fideus-labs/ngff-zarr/commit/67c9b4bf276c7f3e7d03f523613ae1784eb558df))
- **ts**: use zarr.Array for NgffImage.data ([644ff17](https://github.com/fideus-labs/ngff-zarr/commit/644ff17f13ac43f459819a83f71c8f6159a21e1b))
- **ts**: fromNgffZarr simplifications ([113ebfa](https://github.com/fideus-labs/ngff-zarr/commit/113ebfa85dcc3b6448fcb1972698cf46bb297ffe))
- use descriptive CI workflow filenames ([777b70e](https://github.com/fideus-labs/ngff-zarr/commit/777b70e00cd4838cb9f30be4b087c374deac304d))
- **ts**: simplify toNgffZarr, fromNgffZarr ([2bb93e7](https://github.com/fideus-labs/ngff-zarr/commit/2bb93e71e66988cdc551c7279bbbd93d532d5420))
- **ts**: rename ZarrReader to OMEZarrReader ([b1d5f39](https://github.com/fideus-labs/ngff-zarr/commit/b1d5f39922748c4c8d1ae1c1f3bfea9b3ddc206c))
- **ts**: use @std/cli package ([d71107d](https://github.com/fideus-labs/ngff-zarr/commit/d71107daff2d8fa299acddd03c00dab6d37a0cc8))
- **ts**: rename DaskArray to LazyArray ([eb9dea5](https://github.com/fideus-labs/ngff-zarr/commit/eb9dea576674828719e623f93d85e39cc85228bf))
- **ts**: update package org/name to @fideus-labs/ngff-zarr ([8161dba](https://github.com/fideus-labs/ngff-zarr/commit/8161dba4d4e437cd7a090a6bd30bb058c63a3086))

### ⚡ Performance

- **ts**: use reservoir sampling for memory-efficient quantile computation ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))

### ✨ Features

- **py,ts**: add OMERO metadata computation from NgffImage ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- add input validation for OMERO metadata computation ([43d0d0a](https://github.com/fideus-labs/ngff-zarr/commit/43d0d0a4302eb8cdfb08a7d70ab7f146e367c35f))
- **ts**: add RFC-9 (.ozx) zipped OME-Zarr support ([4130015](https://github.com/fideus-labs/ngff-zarr/commit/4130015de7418969c987275b1f574150fb5cea9d))
- add Leica Image Format (LIF) support ([3954a1c](https://github.com/fideus-labs/ngff-zarr/commit/3954a1c7e8cd24fccf9e9c8817c499792164cf13))
- **ts**: add rfc4 support ([3add9b7](https://github.com/fideus-labs/ngff-zarr/commit/3add9b71cdd300f2683e026e2eedd6ca72142bb4))
- **ts**: add RFC 4 validation ([ebfc914](https://github.com/fideus-labs/ngff-zarr/commit/ebfc914193f529e485a09f3de848e43362108053))
- add support for dask>=2025.12.0 ([b8ee6ca](https://github.com/fideus-labs/ngff-zarr/commit/b8ee6ca15f296e94d21075169572a302013147a4))
- **ts**: add path option to ItkImageToNgffImageOptions ([0687f11](https://github.com/fideus-labs/ngff-zarr/commit/0687f11ff7c9f39b87fa9c8029fb2717d0bb0ee5))
- **ts**: add support for downsampling more types ([107a129](https://github.com/fideus-labs/ngff-zarr/commit/107a12956d473e6a15a35cb20c409098ebb754b1))
- **ts**: add downsampleItkWasm method ([3a7602c](https://github.com/fideus-labs/ngff-zarr/commit/3a7602c0166efb5f9b5deca72ddd1534b22b6640))
- **ts**: initial toMultiscales ([c6c1a12](https://github.com/fideus-labs/ngff-zarr/commit/c6c1a12d5c26465926569e577cff45cb5414d7c7))
- add write_hcs_well_image ([709d310](https://github.com/fideus-labs/ngff-zarr/commit/709d310d2478b6d46e14dcd0e0db648542186905))
- **ts**: add basic hcs support ([5c4de7d](https://github.com/fideus-labs/ngff-zarr/commit/5c4de7d0ae1547d10257c0941c5671e8b3716055))
- **ts**: add zod schemas based on spec json schemas ([41d7ecc](https://github.com/fideus-labs/ngff-zarr/commit/41d7ecc8a387266e98df4272b261a4b246e991c3))
- **ts**: add RFC-4 zod schema support ([8157bc0](https://github.com/fideus-labs/ngff-zarr/commit/8157bc0c8e5a882171368892f6382c6128724c8c))
- **ts**: add ngffImageToItkImage ([3f24317](https://github.com/fideus-labs/ngff-zarr/commit/3f2431708af12f4f4e53b755c21c992b9b1261ce))
- **ts**: itk_image_to_ngff_image ([2545cab](https://github.com/fideus-labs/ngff-zarr/commit/2545cabe0420d25a9d51a55f45e7777fb082c36c))
- **ts**: write array data ([583fc71](https://github.com/fideus-labs/ngff-zarr/commit/583fc7168b4b358fd21c12c665395a9f1fe79422))
- **ts**: add BigInt64Array, BigUint64Array to TypedArray ([a5e8de8](https://github.com/fideus-labs/ngff-zarr/commit/a5e8de889c043a4cb7fcf4a459ed2d9968c97bd1))
- **ts**: initial toNgffZarr implementation ([e3e3535](https://github.com/fideus-labs/ngff-zarr/commit/e3e353509a1db7a83febbd5d797f4b314e94e0ae))
- **fromNgffZarr**: support store type inputs ([f347930](https://github.com/fideus-labs/ngff-zarr/commit/f347930f619d1a4a6ce4f2beb4aa211375c13280))
- provide RAS, LPS convenience mappings ([81746be](https://github.com/fideus-labs/ngff-zarr/commit/81746bee6875c04e1837b5c51172cb728eb347c0))
- **ts**: bump deno and zarrita ([eb6b828](https://github.com/fideus-labs/ngff-zarr/commit/eb6b82893097c0a89b0ba9a14d79190fcca97d4a))
- **from_ngff_zarr**: add support for storage_options ([12d00f8](https://github.com/fideus-labs/ngff-zarr/commit/12d00f805f5e83f9c048ea9c7a51a3fd4b38eed1))
- **from_ngff_zarr**: add support for storage_options ([d27fd19](https://github.com/fideus-labs/ngff-zarr/commit/d27fd19930e559b10916802d09c4762da9e8d39a))

### 🐛 Bug Fixes

- **ts**: bump @zarrita/storage for ozx sharded store support ([6bd3e51](https://github.com/fideus-labs/ngff-zarr/commit/6bd3e51df9b2fdb3b5e85b5be8c0b2f972bdfcc7))
- **ts**: add many channel to_multiscales support ([6066741](https://github.com/fideus-labs/ngff-zarr/commit/6066741094d51b1bd1d26a4f52ccd5efc8cb2a23))
- **ts**: write omero metadata inside ome namespace for OME-Zarr 0.5 ([cecb313](https://github.com/fideus-labs/ngff-zarr/commit/cecb3132b49aa8989ae8275a46e75dae6e1c7553))
- **ts**: detect ome property with version 0.5 data ([cbab67a](https://github.com/fideus-labs/ngff-zarr/commit/cbab67ace270d02b0ed92b48eb79de170c8d977d))
- **deps**: pin dask to <2025.11.0 ([8402988](https://github.com/fideus-labs/ngff-zarr/commit/8402988650196b1417c5d4f4e7397d57a460f36d))
- **ts**: add back full component type support ([2f7d59f](https://github.com/fideus-labs/ngff-zarr/commit/2f7d59f2d7196ab2f9b8f02c67b78e2eac0a99fb))
- **ts**: time and channel handling ([472f038](https://github.com/fideus-labs/ngff-zarr/commit/472f038ad7ddefe3f0da8849bf2751d07035e817))
- **ts**: fix vector order indexing and bin-shrink isotroic scaling ([dc64bed](https://github.com/fideus-labs/ngff-zarr/commit/dc64beda247d588112c64fea6b40f22542432cc2))
- incremental downsampling to achieve exact 1x, 2x, 3x, 4x sizes ([a1a00cd](https://github.com/fideus-labs/ngff-zarr/commit/a1a00cd61ea7ee73d1de55a09f9a5b6411547a71))
- **ts**: fix chunk size downsample rounding ([51afc13](https://github.com/fideus-labs/ngff-zarr/commit/51afc130612201b4162f808aa464563e67e3047d))
- **ts**: itk ngff image shape corrections ([67b7932](https://github.com/fideus-labs/ngff-zarr/commit/67b79326109bc5c6cbffcbbae713cc85f63dbafb))
- **ts**: add missing translation offset ([8e020c4](https://github.com/fideus-labs/ngff-zarr/commit/8e020c483e2d05e9c86150e539d507b9bb9e7474))
- **ts**: use a crop radius of zeros ([3194551](https://github.com/fideus-labs/ngff-zarr/commit/319455138b0e678de458af16876c58168e066023))
- **ts**: compute sigma correctly for gaussian filtering ([811ab7f](https://github.com/fideus-labs/ngff-zarr/commit/811ab7f18b6a0e32157ee041a93b53c1b275d952))
- **ts**: support all types, shape order in itkImageToZarr ([c024183](https://github.com/fideus-labs/ngff-zarr/commit/c024183c0eb26d3f87b97a0289887d2e98a904cd))
- **ts**: write images to `scale${index}` ([fc6459a](https://github.com/fideus-labs/ngff-zarr/commit/fc6459a30b25a5b07a2f30a0babd1b7b3fbad951))
- **ts**: more direct typing for publish to JSR ([96254c4](https://github.com/fideus-labs/ngff-zarr/commit/96254c4c7b82324acf62f5bed2683e2c1f29b83b))
- **ts**: export explicit metadata types ([ef279da](https://github.com/fideus-labs/ngff-zarr/commit/ef279da5a9f981128c9a228b873bfa8c02f06b60))
- **ts**: bump downsample dep for Windows path support ([c6cb628](https://github.com/fideus-labs/ngff-zarr/commit/c6cb62849a7332947e090657dca5fb764966027e))
- **ts**: use AnatomicalOrientation in OrientationSchema ([63aa143](https://github.com/fideus-labs/ngff-zarr/commit/63aa1436dff1a66b2f8e5bb17a816ccb4dd0a4c8))
- **ts**: use anatomical value enum ([fede698](https://github.com/fideus-labs/ngff-zarr/commit/fede698840193befd6db53496d47c28027e2d959))
- **ts**: use zod anatomical literal for RFC-4 ([032e213](https://github.com/fideus-labs/ngff-zarr/commit/032e2137164f90811a94da606f086957b4206c63))
- **verify_against_baseline.ts**: remove duplicate code ([15cf899](https://github.com/fideus-labs/ngff-zarr/commit/15cf899d15e4e6bde31706f0ad65f1223966483a))
- **verify_against_baseline.ts**: improve path normalization ([037e08a](https://github.com/fideus-labs/ngff-zarr/commit/037e08ae499dbc02b24bcf2eef1e2e870726e8b0))
- **verify_against_baseline.ts**: use deno's path utilities ([3480075](https://github.com/fideus-labs/ngff-zarr/commit/348007560ce06b48f6e343a71953ea845d75654e))
- **rfc4**: correct LPS RAS direction ([0c00023](https://github.com/fideus-labs/ngff-zarr/commit/0c000233cc28d8c4d45f4a62ce2ca8c4587d05bd))
- **to_ngff_zarr.ts**: remove unused targetTypedArrayConstructor ([1ecfc17](https://github.com/fideus-labs/ngff-zarr/commit/1ecfc17687eb7a383339cf87c209524396e223ad))
- **to_ngff_zarr**: throw with unknown dtype ([f315a97](https://github.com/fideus-labs/ngff-zarr/commit/f315a977dd7a5cff86060c1d02fcec412bd4c153))
- **to_ngff_zarr**: use null to write full array ([c16efbf](https://github.com/fideus-labs/ngff-zarr/commit/c16efbfd49fc55902111741f36f46ef3c3e5f19a))
- **rfc.py**: add missing values ([6f7ffca](https://github.com/fideus-labs/ngff-zarr/commit/6f7ffca93ee792d2d9da77a11072012f33d6f887))
- **ts**: Windows FileSystemStore path normalization ([f5d26b6](https://github.com/fideus-labs/ngff-zarr/commit/f5d26b62307eaa53922c0fbe5ca26008e59f437f))
- **ts**: add zod schema types ([dc7549e](https://github.com/fideus-labs/ngff-zarr/commit/dc7549efed53d4bef932272c9b3ee1a255c23865))
- **ts**: remove unused playwright.config.js ([bdf63a6](https://github.com/fideus-labs/ngff-zarr/commit/bdf63a692f325a8ad5c5641d165b3db3540d1d10))
- **ts**: remove unused dask_array.ts ([a65c6d7](https://github.com/fideus-labs/ngff-zarr/commit/a65c6d7fa38c4de22caea0a1a3b40513d0568e10))
- **ts**: remove invalid methods from MethodsSchema ([d7bd8cb](https://github.com/fideus-labs/ngff-zarr/commit/d7bd8cb64c2b661de928c052d01bcce90eec3728))
- **ts**: fix deno task build ([ae150d0](https://github.com/fideus-labs/ngff-zarr/commit/ae150d09aeaa425244dc1d8f808398a09b0a8d00))
- **ts**: remove ITK_* and DASK_IMAGE_* methods ([727f132](https://github.com/fideus-labs/ngff-zarr/commit/727f132f787b11c0a287e4a00a85b0da9ef03e4a))
