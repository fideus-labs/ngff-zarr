## ts-v0.22.0 (2026-06-26)

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

- **ts**: add temporary OME-Zarr 0.6.dev4 version support ([9d887f9](https://github.com/fideus-labs/ngff-zarr/commit/9d887f9e222602b38afd4191aac593ee3d21772d))
- **ts**: add OME-Zarr v0.6 (RFC-5) coordinate transformations support ([859519b](https://github.com/fideus-labs/ngff-zarr/commit/859519b944afd0aa0f2b12f0ac8e2ee4f84ab53f))

### 🐛 Bug Fixes

- **ts**: deno fmt with deno 2.9.0 ([4eeb5b9](https://github.com/fideus-labs/ngff-zarr/commit/4eeb5b926338186036cc7ac62257ef6fe7a624f9))
- format ts CHANGELOG ([93ee09f](https://github.com/fideus-labs/ngff-zarr/commit/93ee09f7627fe73d63ee8b0e0a66729b717aaf53))

## ts-v0.20.0 (2026-06-18)

### ✨ Features

- **py,ts**: add OME-Zarr v0.5 structural validation and schema support ([598ed91](https://github.com/fideus-labs/ngff-zarr/commit/598ed91e6f3b15533d9a50a59b8cd68b3674e22f))
- **py,ts**: extend RFC-4 vocabulary with subject-local tissue orientations ([0cc4a91](https://github.com/fideus-labs/ngff-zarr/commit/0cc4a9146fb9113d0a7c289f7787abdd825e362d))

## ts-v0.19.0 (2026-06-04)

### ✨ Features

- **ts**: add OME-Zarr v0.4 structural validation ([7c0e3e2](https://github.com/fideus-labs/ngff-zarr/commit/7c0e3e222f6ceb62e2aa7d0e97ba367be3695da5))

### 🐛 Bug Fixes

- **ci**: remove OTLP import from daily-doc-updater workflow ([45394c8](https://github.com/fideus-labs/ngff-zarr/commit/45394c8ccb09b5ad3687d8dd834c94af45d69469))
- **ts**: honor enabledRfcs in browser toOmeZarr ([62c1ffc](https://github.com/fideus-labs/ngff-zarr/commit/62c1ffc7fa4b47734f25d2eed740289f24263601))
- stop pre-commit from mangling gh-aw generated files ([d6f1201](https://github.com/fideus-labs/ngff-zarr/commit/d6f120144884bc6a3089e5286a624ed0316f90ba))

## ts-v0.21.1 (2026-06-25)

### 🐛 Bug Fixes

- format ts CHANGELOG ([93ee09f](https://github.com/fideus-labs/ngff-zarr/commit/93ee09f7627fe73d63ee8b0e0a66729b717aaf53))

## ts-v0.21.0 (2026-06-25)

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

- **ts**: add temporary OME-Zarr 0.6.dev4 version support ([9d887f9](https://github.com/fideus-labs/ngff-zarr/commit/9d887f9e222602b38afd4191aac593ee3d21772d))
- **ts**: add OME-Zarr v0.6 (RFC-5) coordinate transformations support ([859519b](https://github.com/fideus-labs/ngff-zarr/commit/859519b944afd0aa0f2b12f0ac8e2ee4f84ab53f))

## ts-v0.20.0 (2026-06-18)

### ✨ Features

- **py,ts**: add OME-Zarr v0.5 structural validation and schema support ([598ed91](https://github.com/fideus-labs/ngff-zarr/commit/598ed91e6f3b15533d9a50a59b8cd68b3674e22f))
- **py,ts**: extend RFC-4 vocabulary with subject-local tissue orientations ([0cc4a91](https://github.com/fideus-labs/ngff-zarr/commit/0cc4a9146fb9113d0a7c289f7787abdd825e362d))

## ts-v0.19.0 (2026-06-04)

### ✨ Features

- **ts**: add OME-Zarr v0.4 structural validation ([7c0e3e2](https://github.com/fideus-labs/ngff-zarr/commit/7c0e3e222f6ceb62e2aa7d0e97ba367be3695da5))

### 🐛 Bug Fixes

- **ci**: remove OTLP import from daily-doc-updater workflow ([45394c8](https://github.com/fideus-labs/ngff-zarr/commit/45394c8ccb09b5ad3687d8dd834c94af45d69469))
- **ts**: honor enabledRfcs in browser toOmeZarr ([62c1ffc](https://github.com/fideus-labs/ngff-zarr/commit/62c1ffc7fa4b47734f25d2eed740289f24263601))
- stop pre-commit from mangling gh-aw generated files ([d6f1201](https://github.com/fideus-labs/ngff-zarr/commit/d6f120144884bc6a3089e5286a624ed0316f90ba))

## ts-v0.18.1 (2026-04-16)

### 🐛 Bug Fixes

- add pypi-prerelease-mode: if-necessary-or-explicit to ts/pixi.lock ([a84b8a8](https://github.com/fideus-labs/ngff-zarr/commit/a84b8a871ba157ed83524f6cd2f0d0d1101b1299))

## ts-v0.18.0 (2026-04-15)

### ♻️ Refactoring

- rename `Multiscales` to `NgffMultiscales` across py, ts, mcp ([1d0c572](https://github.com/fideus-labs/ngff-zarr/commit/1d0c5724fdd7ba86679f57b3343332a44d1aff5d))

### ✨ Features

- **py,ts**: rename from_ngff_zarr/fromNgffZarr to from_ome_zarr/fromOmeZarr ([84a20b2](https://github.com/fideus-labs/ngff-zarr/commit/84a20b28bce135e4009197b7671fa9ad21e772a0))
- **py,ts**: rename to_ngff_zarr/toNgffZarr to to_ome_zarr/toOmeZarr ([9431546](https://github.com/fideus-labs/ngff-zarr/commit/94315466d03b7c3e73a3e42fc7e236c428f1d95c))
- update default OME-Zarr version from 0.4 to 0.5 ([4e60c07](https://github.com/fideus-labs/ngff-zarr/commit/4e60c077b3adcf6f4ef827a1cc112ca8e340f22c))
- **py,ts**: add bidirectional anatomical orientation ↔ ITK direction matrix conversion ([e427ca4](https://github.com/fideus-labs/ngff-zarr/commit/e427ca4e5e131a97b9b566a5fd79ac6020110dd9))

### 🐛 Bug Fixes

- resolve merge conflicts with main branch ([af2a8eb](https://github.com/fideus-labs/ngff-zarr/commit/af2a8eb4c5e4aa995d7afd79e92a574549db3dfc))
- **ts**: fix TypeScript test failures - handle v0.5 ome namespace in metadata access ([44b5cfb](https://github.com/fideus-labs/ngff-zarr/commit/44b5cfb74cbf1fb8eec7d3490844b5ee0bfcbfc1))

## ts-v0.15.1 (2026-03-10)

### ♻️ Refactoring

- **ts**: named WellImage export ([5a647e6](https://github.com/fideus-labs/ngff-zarr/commit/5a647e670e021e31bf5c3d4f4ee4cf6621a074c3))

### 🐛 Bug Fixes

- **ts**: use number | undefined in WellImage interface for exactOptionalPropertyTypes ([5feb06a](https://github.com/fideus-labs/ngff-zarr/commit/5feb06afcf051e487e7c4c8e209ef075b808f594))
- **ts**: resolve duplicate WellImage identifier and use ZodType annotation ([7d6d8ba](https://github.com/fideus-labs/ngff-zarr/commit/7d6d8ba2533941ef60ed6c59ebece19656cb9301))
- **ts**: add explicit type annotation to WellImageSchema for JSR public API ([c177cd3](https://github.com/fideus-labs/ngff-zarr/commit/c177cd31490cbd099eeabf7cd10ef979a33918d8))

## ts-v0.15.0 (2026-03-04)

### ✨ Features

- **py,ts**: relax well image path constraints per ome/ngff-spec#71 ([32b7723](https://github.com/fideus-labs/ngff-zarr/commit/32b7723a79698272dea694348f24505ff8efa6d9))

### 🐛 Bug Fixes

- **ci**: avoid duplicate pixi binary install in release workflow ([07dbd36](https://github.com/fideus-labs/ngff-zarr/commit/07dbd3642a3dc44fb06f48523fcf0221d1d00607))

## ts-v0.14.0 (2026-02-24)

### ✨ Features

- **ts**: add configurable worker pool size via config.workerPoolSize ([3b308dd](https://github.com/fideus-labs/ngff-zarr/commit/3b308dd11af49d610c801d5074ed4a04f40aa8a0))

## ts-v0.13.0 (2026-02-20)

### ⚡ Performance

- **ts**: cap worker pool to 16 ([66834bb](https://github.com/fideus-labs/ngff-zarr/commit/66834bb90d0418db60dcb0304ad4540107f701e4))

## ts-v0.12.3 (2026-02-18)

### 🐛 Bug Fixes

- **ts**: address character replacement for inline worker ([1824b25](https://github.com/fideus-labs/ngff-zarr/commit/1824b25e50663d452ffda46e1ef5ad4ea648ffb1))

## ts-v0.12.2 (2026-02-18)

### 🐛 Bug Fixes

- **ts**: always build bundle for npm package ([38de129](https://github.com/fideus-labs/ngff-zarr/commit/38de129da66dee80b26c37f2b5dd819547fb1b32))

## ts-v0.12.1 (2026-02-18)

### 🐛 Bug Fixes

- **ts**: handle padded edge chunks and fix biased reservoir merge ([492a3a2](https://github.com/fideus-labs/ngff-zarr/commit/492a3a29dceb78bd1572ccf3e8993a8c8d4d4632))
- **ts**: add explicit return type to defaultCodecs public function ([d488baa](https://github.com/fideus-labs/ngff-zarr/commit/d488baae644c5b5f0c17e9bf95e23fd6f7f170a3))

## ts-v0.12.0 (2026-02-17)

### ✨ Features

- **ts**: add 2D support and comprehensive tests for direction matrix handling ([83ff58f](https://github.com/fideus-labs/ngff-zarr/commit/83ff58f4d503e080197793d7c164a8fbace2565d))
- **ts**: derive anatomical orientation from ITK direction matrix ([d3e55d1](https://github.com/fideus-labs/ngff-zarr/commit/d3e55d1d69463d26522e849c7d12525064d906f3))
- **ts**: add codecs option to toMultiscales for skipping blosc compression ([5bebee3](https://github.com/fideus-labs/ngff-zarr/commit/5bebee309c2d5ebc2768e65aac33936b978b910a))

### 🐛 Bug Fixes

- **ci**: correct new contributor detection and filter bots ([836992d](https://github.com/fideus-labs/ngff-zarr/commit/836992dd0094aaf0c8036200e7de606a628094da))

## ts-v0.11.0 (2026-02-15)

### ♻️ Refactoring

- **ts**: extract chunk fallback logic and rename parameter for clarity ([605b1bb](https://github.com/fideus-labs/ngff-zarr/commit/605b1bbfc6151a6f4da8568437dbb148ab3178b6))

### ✨ Features

- **ts**: add onProgress callback to computeOmeroFromNgffImage and toNgffZarrOzx ([4d49305](https://github.com/fideus-labs/ngff-zarr/commit/4d49305b68124c7f7f472ba3195b65ac7edd8c82))
- **ts**: export ngffImageToItkImage from browser entry point ([423c796](https://github.com/fideus-labs/ngff-zarr/commit/423c796e7350b252223b9b4d7ee4190c8b4f8999))

## ts-v0.10.0 (2026-02-15)

### ✨ Features

- **ts**: add support for general zarrita stores to fromNgffZarr ([fafaeb4](https://github.com/fideus-labs/ngff-zarr/commit/fafaeb44078c6dc515b73cfd49dcd3661caa7a86))
- **ts**: add support general zarrita stores to fromNgffZarr ([29f5bd0](https://github.com/fideus-labs/ngff-zarr/commit/29f5bd00f48c8746a41af6e0f80692e1a7017251))

### 🐛 Bug Fixes

- **ts**: reorder store type checks to avoid inconsistent behavior ([4bef960](https://github.com/fideus-labs/ngff-zarr/commit/4bef960e05bd547c7f61bc1d38b5701bdc669155))

## ts-v0.9.0 (2026-02-14)

### ♻️ Refactoring

- **ts**: apply PR review feedback for type safety and performance ([46da7a2](https://github.com/fideus-labs/ngff-zarr/commit/46da7a2bb49f7815ffb30261ed8fec1457aefdd1))

### ✨ Features

- **ts**: replace Comlink OMERO computation with unified worker-pool implementation ([e11eb63](https://github.com/fideus-labs/ngff-zarr/commit/e11eb6348de386636f5bf4d6d0fd24904cdbdc4b))

### 🐛 Bug Fixes

- **ts**: resolve CI formatting and type-checking failures ([a94d022](https://github.com/fideus-labs/ngff-zarr/commit/a94d022c33a4868eb22fb93652cb466cc70a9f64))
- **ci**: guard printf against names starting with dash ([7ef5d40](https://github.com/fideus-labs/ngff-zarr/commit/7ef5d406a6c6c8495e7b0652cedb8ec2656d82dc))

## ts-v0.8.0 (2026-02-12)

### ✨ Features

- **ts**: always use highest resolution for OMERO statistics ([c3ae65f](https://github.com/fideus-labs/ngff-zarr/commit/c3ae65fbfb0214e6a861df2523f26b9066cca400))

### 🐛 Bug Fixes

- **ts**: handle 2D multi-component images in itkImageToNgffImage ([962a017](https://github.com/fideus-labs/ngff-zarr/commit/962a01780874cc2162ac706f26b377408722f331))

## ts-v0.7.5 (2026-02-11)

### 🐛 Bug Fixes

- **ts**: do not always set method undefined with fromNgffZarr ([20bd29a](https://github.com/fideus-labs/ngff-zarr/commit/20bd29ab44a0cbf51c86f95b134d183d353380e6))

## ts-v0.7.4 (2026-02-10)

### ⚡ Performance

- **ts**: add cache option to fromNgffZarr with fizarrita 1.2.0 ([cd84d9e](https://github.com/fideus-labs/ngff-zarr/commit/cd84d9edc4057b3d6fc0c802924da1ab58452de8))

### 🐛 Bug Fixes

- **ts**: use spec-compliant blosc codec shuffle string and typesize ([938cfb6](https://github.com/fideus-labs/ngff-zarr/commit/938cfb6fff73c5af6ae11cf47e8bd4961c0672a4))

## ts-v0.7.3 (2026-02-09)

### 🐛 Bug Fixes

- **ts**: fizarrita bump for more reliable shape estimation ([283c925](https://github.com/fideus-labs/ngff-zarr/commit/283c925caf6350655e7b4a40647bb55b4ab4e21d))

## ts-v0.7.2 (2026-02-08)

### 🐛 Bug Fixes

- **ts**: bump fizarrita for better chunk shape detection ([261da2f](https://github.com/fideus-labs/ngff-zarr/commit/261da2fbd2dd0c101db9604c69a8e08ed66db4c6))

## ts-v0.7.1 (2026-02-08)

### 🐛 Bug Fixes

- **ts**: bump fizarrita for edge chunk handling ([ad84810](https://github.com/fideus-labs/ngff-zarr/commit/ad8481021552b26b8e06d755e574aa15ca54a774))

## ts-v0.17.0 (2026-03-20)

### ✨ Features

- update default OME-Zarr version from 0.4 to 0.5 ([4e60c07](https://github.com/fideus-labs/ngff-zarr/commit/4e60c077b3adcf6f4ef827a1cc112ca8e340f22c))

### 🐛 Bug Fixes

- resolve merge conflicts with main branch ([af2a8eb](https://github.com/fideus-labs/ngff-zarr/commit/af2a8eb4c5e4aa995d7afd79e92a574549db3dfc))
- **ts**: fix TypeScript test failures - handle v0.5 ome namespace in metadata access ([44b5cfb](https://github.com/fideus-labs/ngff-zarr/commit/44b5cfb74cbf1fb8eec7d3490844b5ee0bfcbfc1))

## ts-v0.16.0 (2026-03-11)

### ✨ Features

- **py,ts**: add bidirectional anatomical orientation ↔ ITK direction matrix conversion ([e427ca4](https://github.com/fideus-labs/ngff-zarr/commit/e427ca4e5e131a97b9b566a5fd79ac6020110dd9))

## ts-v0.15.1 (2026-03-10)

### ♻️ Refactoring

- **ts**: named WellImage export ([5a647e6](https://github.com/fideus-labs/ngff-zarr/commit/5a647e670e021e31bf5c3d4f4ee4cf6621a074c3))

### 🐛 Bug Fixes

- **ts**: use number | undefined in WellImage interface for exactOptionalPropertyTypes ([5feb06a](https://github.com/fideus-labs/ngff-zarr/commit/5feb06afcf051e487e7c4c8e209ef075b808f594))
- **ts**: resolve duplicate WellImage identifier and use ZodType annotation ([7d6d8ba](https://github.com/fideus-labs/ngff-zarr/commit/7d6d8ba2533941ef60ed6c59ebece19656cb9301))
- **ts**: add explicit type annotation to WellImageSchema for JSR public API ([c177cd3](https://github.com/fideus-labs/ngff-zarr/commit/c177cd31490cbd099eeabf7cd10ef979a33918d8))

## ts-v0.15.0 (2026-03-04)

### ✨ Features

- **py,ts**: relax well image path constraints per ome/ngff-spec#71 ([32b7723](https://github.com/fideus-labs/ngff-zarr/commit/32b7723a79698272dea694348f24505ff8efa6d9))

### 🐛 Bug Fixes

- **ci**: avoid duplicate pixi binary install in release workflow ([07dbd36](https://github.com/fideus-labs/ngff-zarr/commit/07dbd3642a3dc44fb06f48523fcf0221d1d00607))

## ts-v0.14.0 (2026-02-24)

### ✨ Features

- **ts**: add configurable worker pool size via config.workerPoolSize ([3b308dd](https://github.com/fideus-labs/ngff-zarr/commit/3b308dd11af49d610c801d5074ed4a04f40aa8a0))

## ts-v0.13.0 (2026-02-20)

### ⚡ Performance

- **ts**: cap worker pool to 16 ([66834bb](https://github.com/fideus-labs/ngff-zarr/commit/66834bb90d0418db60dcb0304ad4540107f701e4))

## ts-v0.12.3 (2026-02-18)

### 🐛 Bug Fixes

- **ts**: address character replacement for inline worker ([1824b25](https://github.com/fideus-labs/ngff-zarr/commit/1824b25e50663d452ffda46e1ef5ad4ea648ffb1))

## ts-v0.12.2 (2026-02-18)

### 🐛 Bug Fixes

- **ts**: always build bundle for npm package ([38de129](https://github.com/fideus-labs/ngff-zarr/commit/38de129da66dee80b26c37f2b5dd819547fb1b32))

## ts-v0.12.1 (2026-02-18)

### 🐛 Bug Fixes

- **ts**: handle padded edge chunks and fix biased reservoir merge ([492a3a2](https://github.com/fideus-labs/ngff-zarr/commit/492a3a29dceb78bd1572ccf3e8993a8c8d4d4632))
- **ts**: add explicit return type to defaultCodecs public function ([d488baa](https://github.com/fideus-labs/ngff-zarr/commit/d488baae644c5b5f0c17e9bf95e23fd6f7f170a3))

## ts-v0.12.0 (2026-02-17)

### ✨ Features

- **ts**: add 2D support and comprehensive tests for direction matrix handling ([83ff58f](https://github.com/fideus-labs/ngff-zarr/commit/83ff58f4d503e080197793d7c164a8fbace2565d))
- **ts**: derive anatomical orientation from ITK direction matrix ([d3e55d1](https://github.com/fideus-labs/ngff-zarr/commit/d3e55d1d69463d26522e849c7d12525064d906f3))
- **ts**: add codecs option to toMultiscales for skipping blosc compression ([5bebee3](https://github.com/fideus-labs/ngff-zarr/commit/5bebee309c2d5ebc2768e65aac33936b978b910a))

### 🐛 Bug Fixes

- **ci**: correct new contributor detection and filter bots ([836992d](https://github.com/fideus-labs/ngff-zarr/commit/836992dd0094aaf0c8036200e7de606a628094da))

## ts-v0.11.0 (2026-02-15)

### ♻️ Refactoring

- **ts**: extract chunk fallback logic and rename parameter for clarity ([605b1bb](https://github.com/fideus-labs/ngff-zarr/commit/605b1bbfc6151a6f4da8568437dbb148ab3178b6))

### ✨ Features

- **ts**: add onProgress callback to computeOmeroFromNgffImage and toNgffZarrOzx ([4d49305](https://github.com/fideus-labs/ngff-zarr/commit/4d49305b68124c7f7f472ba3195b65ac7edd8c82))
- **ts**: export ngffImageToItkImage from browser entry point ([423c796](https://github.com/fideus-labs/ngff-zarr/commit/423c796e7350b252223b9b4d7ee4190c8b4f8999))

## ts-v0.10.0 (2026-02-15)

### ✨ Features

- **ts**: add support for general zarrita stores to fromNgffZarr ([fafaeb4](https://github.com/fideus-labs/ngff-zarr/commit/fafaeb44078c6dc515b73cfd49dcd3661caa7a86))
- **ts**: add support general zarrita stores to fromNgffZarr ([29f5bd0](https://github.com/fideus-labs/ngff-zarr/commit/29f5bd00f48c8746a41af6e0f80692e1a7017251))

### 🐛 Bug Fixes

- **ts**: reorder store type checks to avoid inconsistent behavior ([4bef960](https://github.com/fideus-labs/ngff-zarr/commit/4bef960e05bd547c7f61bc1d38b5701bdc669155))

## ts-v0.9.0 (2026-02-14)

### ♻️ Refactoring

- **ts**: apply PR review feedback for type safety and performance ([46da7a2](https://github.com/fideus-labs/ngff-zarr/commit/46da7a2bb49f7815ffb30261ed8fec1457aefdd1))

### ✨ Features

- **ts**: replace Comlink OMERO computation with unified worker-pool implementation ([e11eb63](https://github.com/fideus-labs/ngff-zarr/commit/e11eb6348de386636f5bf4d6d0fd24904cdbdc4b))

### 🐛 Bug Fixes

- **ts**: resolve CI formatting and type-checking failures ([a94d022](https://github.com/fideus-labs/ngff-zarr/commit/a94d022c33a4868eb22fb93652cb466cc70a9f64))
- **ci**: guard printf against names starting with dash ([7ef5d40](https://github.com/fideus-labs/ngff-zarr/commit/7ef5d406a6c6c8495e7b0652cedb8ec2656d82dc))

## ts-v0.8.0 (2026-02-12)

### ✨ Features

- **ts**: always use highest resolution for OMERO statistics ([c3ae65f](https://github.com/fideus-labs/ngff-zarr/commit/c3ae65fbfb0214e6a861df2523f26b9066cca400))

### 🐛 Bug Fixes

- **ts**: handle 2D multi-component images in itkImageToNgffImage ([962a017](https://github.com/fideus-labs/ngff-zarr/commit/962a01780874cc2162ac706f26b377408722f331))

## ts-v0.7.5 (2026-02-11)

### 🐛 Bug Fixes

- **ts**: do not always set method undefined with fromNgffZarr ([20bd29a](https://github.com/fideus-labs/ngff-zarr/commit/20bd29ab44a0cbf51c86f95b134d183d353380e6))

## ts-v0.7.4 (2026-02-10)

### ⚡ Performance

- **ts**: add cache option to fromNgffZarr with fizarrita 1.2.0 ([cd84d9e](https://github.com/fideus-labs/ngff-zarr/commit/cd84d9edc4057b3d6fc0c802924da1ab58452de8))

### 🐛 Bug Fixes

- **ts**: use spec-compliant blosc codec shuffle string and typesize ([938cfb6](https://github.com/fideus-labs/ngff-zarr/commit/938cfb6fff73c5af6ae11cf47e8bd4961c0672a4))

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
