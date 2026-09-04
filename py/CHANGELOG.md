## py-v0.46.0 (2026-09-04)

### ♻️ Refactoring

- **py**: keep the 0.9 development model out of the package exports ([ffb1619](https://github.com/fideus-labs/ngff-zarr/commit/ffb161960a57cf693b28ecdfabd22a1163b157d2))
- rename convert_field_block to convert_itk_field_block ([f9c3b5c](https://github.com/fideus-labs/ngff-zarr/commit/f9c3b5c923e09554aedcd4a179460b759388d3e5))
- **py**: name the axis and field transformation types ([6cdb8a9](https://github.com/fideus-labs/ngff-zarr/commit/6cdb8a927e41a91652e083833cf8376d511dfb60))

### ✨ Features

- **py**: add consolidate_metadata option to to_ome_zarr ([2de3c7a](https://github.com/fideus-labs/ngff-zarr/commit/2de3c7aec584ad3f21a67c2babf1145035611d2b))
- **py**: add consolidate_metadata option to to_ome_zarr ([3f31a1c](https://github.com/fideus-labs/ngff-zarr/commit/3f31a1ceb9f7407325257871e4d51baf48642a56))
- **py**: make the version-scoped metadata API public per version ([8d206b8](https://github.com/fideus-labs/ngff-zarr/commit/8d206b81875d9ea225da4e10458d053e43f4be09))
- **py**: convert a field block between ITK's convention and RFC-5's ([25767d9](https://github.com/fideus-labs/ngff-zarr/commit/25767d908e0fb9a6f8dd2f4af5d15105308f150a))
- **py**: declare a field image as an RFC-5 field transform ([bece32f](https://github.com/fideus-labs/ngff-zarr/commit/bece32f7cb53eb86a02abcf9775eb462f6341d7c))
- **py**: open an array in a remote store ([dfc13e6](https://github.com/fideus-labs/ngff-zarr/commit/dfc13e63489647499c43570b6f0310e8fc675864))
- **py**: read a displacement field one window per block ([329a084](https://github.com/fideus-labs/ngff-zarr/commit/329a084f890ac33508e089215d3555c98baa9d5b))

### 🐛 Bug Fixes

- **py**: v09 __all__ keeps its v06 re-exports; test covers v04 and required names ([53dd4e1](https://github.com/fideus-labs/ngff-zarr/commit/53dd4e16e8c5f5d59d63cc478cdb82d33830d48d))
- **py,ts**: carry a displacement range through a non-linear outer stage ([76c0ae0](https://github.com/fideus-labs/ngff-zarr/commit/76c0ae0bdf0e44d34f6c70c37e680d98a0b3a5c5))
- **py**: refuse a write through an empty stepped slice ([fc47c5c](https://github.com/fideus-labs/ngff-zarr/commit/fc47c5cd9ac5074452c37f76208e342c23ec8eb2))
- **py**: serve stepped and negative-step slices from the lazy adapter ([525919f](https://github.com/fideus-labs/ngff-zarr/commit/525919fb89b1b64d97acf14446484be200e16e34))
- **py**: refuse the multiscale-level transforms a version would drop ([2fd62bd](https://github.com/fideus-labs/ngff-zarr/commit/2fd62bd2747637adca4d86f6e6dc5ee9c6bfb3f5))
- **py**: resolve the bucket region an S3 read needs ([1c1e615](https://github.com/fideus-labs/ngff-zarr/commit/1c1e61578a2acc2a83ee4be55d7d07cfba7f9e4f))
- **py**: refuse a transform whose vector does not span its axes ([29c3ef2](https://github.com/fideus-labs/ngff-zarr/commit/29c3ef21763cc7b7af7aeb4bb8edd07b538ba9d7))
- **py,ts**: let only a field prove the grid stays on its domain ([318099c](https://github.com/fideus-labs/ngff-zarr/commit/318099cbc1426213ed6f1b2338df52d8d3b54288))
- **py,ts**: bound a displacement stage of an ITK transform by its values ([1753989](https://github.com/fideus-labs/ngff-zarr/commit/1753989a550f15e5d6fbd80205b0f8b9c40e5443))
- **py,ts**: size a field's region by what the field displaces ([bc9f983](https://github.com/fideus-labs/ngff-zarr/commit/bc9f983c0a74f52dfd0ef53f1dde5b3332c09e39))
- **py**: reject the codec and chunk arguments the writer cannot honor ([c350f2d](https://github.com/fideus-labs/ngff-zarr/commit/c350f2dc02e93e207fdf64ef0cadf9bec4b87f0b))
- **py**: reject codec chains the write engine cannot encode ([e0127a4](https://github.com/fideus-labs/ngff-zarr/commit/e0127a431aee11c45551128de1569356a5345a3f))
- **py**: carry upstream's re-synced RFC-4 schema descriptions ([3d95b71](https://github.com/fideus-labs/ngff-zarr/commit/3d95b71bbc75682d51cf3591c80a5d68ee2053ab))

## py-v0.45.0 (2026-08-27)

### ✨ Features

- **py**: validate 0.9.dev1 stores against the bundled schemas ([98d38b8](https://github.com/fideus-labs/ngff-zarr/commit/98d38b882c6ad776f0d64689c5b32ad51880c21d))

### 🐛 Bug Fixes

- **py**: treat a version that is not a string as no version at all ([2309416](https://github.com/fideus-labs/ngff-zarr/commit/2309416908b48deae5bfd804f1b425802aeea4ac))
- **py**: word the byDimension axes as upstream words them ([e95535f](https://github.com/fideus-labs/ngff-zarr/commit/e95535fb9ce372fa1d607e8df44659acfe388364))
- **py**: let a malformed 0.9 document reach the schema that describes it ([144b418](https://github.com/fideus-labs/ngff-zarr/commit/144b418d818e9afa13c57f4e149a711613236930))
- **py**: correct two schema descriptions the validator prints ([6eb7936](https://github.com/fideus-labs/ngff-zarr/commit/6eb793679d2767236b151eb4695966676d90ec2e))
- **py,ts**: gate the RFC-4 orientation rules on 0.9.dev1 ([7205c2e](https://github.com/fideus-labs/ngff-zarr/commit/7205c2e85e84f8bb1f07fa5a188b7440634e9846))

## py-v0.44.0 (2026-08-27)

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
- to_ngff_zarr, from_ngff_zarr, to_hcs_zarr,
write_hcs_well_image, and upgrade_ome_zarr no longer accept zarr-python
store objects (MemoryStore, LocalStore, ...). Pass a local directory
path instead; reads additionally accept remote URL strings, .ozx
archives, and key-to-bytes mappings. ([a65596f](https://github.com/fideus-labs/ngff-zarr/commit/a65596f30f12a102ec0590a21360ea6be0dd9c58))

### ♻️ Refactoring

- **py,ts**: name the resamplers after what they take ([d3e48f5](https://github.com/fideus-labs/ngff-zarr/commit/d3e48f53500d1f0726ff3e4334d03f231a58b62b))
- **py**: adopt enum.StrEnum now that Python >= 3.11 is required ([6201bb2](https://github.com/fideus-labs/ngff-zarr/commit/6201bb2a7b5d12923aebc2a682e8a3bb22a78311))
- **py**: remove legacy zarr-python fallback branches ([a65596f](https://github.com/fideus-labs/ngff-zarr/commit/a65596f30f12a102ec0590a21360ea6be0dd9c58))
- **py**: replace zarr-python type helpers with self-contained store types ([07beb3c](https://github.com/fideus-labs/ngff-zarr/commit/07beb3c1e0741a52c8664a6fd0ed0dce77d0fcb9))

### ⚡ Performance

- **py**: walk each dask layer once when guarding an overwrite ([635ae78](https://github.com/fideus-labs/ngff-zarr/commit/635ae7896b07ef0a592ce027b33b7999c90ff0fb))
- **py**: name the transform once in the graph instead of once per block ([0aee194](https://github.com/fideus-labs/ngff-zarr/commit/0aee19431657df53d3898d906dcb4a32b5300982))
- **py**: read each moving chunk once across resample blocks ([edca3c1](https://github.com/fideus-labs/ngff-zarr/commit/edca3c1479f286a8e1743ba21073f1430b7c23b4))

### ✨ Features

- **py,ts**: support the RFC-5 projectAxis transformation ([367eb06](https://github.com/fideus-labs/ngff-zarr/commit/367eb069674e4fa6588474ca2387c3d61dcb0d23))
- **py**: open a store array for region reads and writes ([51ec893](https://github.com/fideus-labs/ngff-zarr/commit/51ec893732607bacaafc1ba863a3aee4f5663a05))
- **py**: carry root attributes beside the OME metadata ([aea7a15](https://github.com/fideus-labs/ngff-zarr/commit/aea7a15e72a23df68ae1640183d5f2e0e7f3b449))
- **py**: add start_level to to_ome_zarr ([a0ee542](https://github.com/fideus-labs/ngff-zarr/commit/a0ee542004eea33bf4a74612cdfc657ada6979df))
- **py**: add metadata_only to to_ome_zarr ([2b6dfa1](https://github.com/fideus-labs/ngff-zarr/commit/2b6dfa1cc8c0e652caade1575b151ac2ba330f88))
- **py**: add Methods.DASK_BIN_SHRINK ([80036d6](https://github.com/fideus-labs/ngff-zarr/commit/80036d66d4d55d8929b8331c29be732f485ac4e3))
- **py,ts**: convert every RFC-5 transformation type to ITK ([1d4f063](https://github.com/fideus-labs/ngff-zarr/commit/1d4f063f09b6b1704500137be37651fddbd6752b))
- **py**: convert ITK displacement fields to and from RFC-5 displacements ([297c201](https://github.com/fideus-labs/ngff-zarr/commit/297c201f476073ae09e44ebc66480d468adae871))
- **py,ts**: exact frame conversion and hardened ITK-Wasm decoding ([444c879](https://github.com/fideus-labs/ngff-zarr/commit/444c879c881b2c8f248838125100dff12b00ecbe))
- convert coordinate transformations between RFC-5 and ITK ([16ec87b](https://github.com/fideus-labs/ngff-zarr/commit/16ec87bd0348ba8eec08df0c736ab6d7833e21db))
- **py,ts**: offer 0.9.dev1 as an upgrade target and document it ([c256ea3](https://github.com/fideus-labs/ngff-zarr/commit/c256ea3af4816a09c9bcfdf17fccf3403187d184))
- **py,ts**: add the OME-Zarr 0.9.dev1 version and its metadata model ([264c8bd](https://github.com/fideus-labs/ngff-zarr/commit/264c8bdd46a5caa8c7a515446f509998e9dc8f16))
- **py,ts**: track the OME-Zarr 0.6rc0 schemas and version tag ([69b6e14](https://github.com/fideus-labs/ngff-zarr/commit/69b6e14c64c0338731177f62ec53380482e3e833))
- **py**: require Python >= 3.11 ([6ea8c96](https://github.com/fideus-labs/ngff-zarr/commit/6ea8c96802a593ad52edcb1b73dad251c5543ccf))
- **py**: drop zarr-python runtime dependency in favor of zarrista ([9f76343](https://github.com/fideus-labs/ngff-zarr/commit/9f76343dbb43a377cee16e4eb9db7553c59f0c53))
- **py**: read remote stores through zarrista and obstore ([8341f56](https://github.com/fideus-labs/ngff-zarr/commit/8341f56733ebe9d37de312901b80df52f577028f))
- **py**: read OZX zip archives through zarrista ([656eeb7](https://github.com/fideus-labs/ngff-zarr/commit/656eeb7ee4d48b4083b2a9cfea48c5e594fb1459))
- **py**: read tifffile aszarr stores through the store reader ([f477399](https://github.com/fideus-labs/ngff-zarr/commit/f4773996159ea38f7870eb9289fb20455f2e47ba))
- **py**: route remaining local readers through zarrista ([b9f832d](https://github.com/fideus-labs/ngff-zarr/commit/b9f832db12e03c4d7354aefa4f83b78f1fbefaa5))
- **py**: route from_ngff_zarr reads through zarrista ([dfaa48f](https://github.com/fideus-labs/ngff-zarr/commit/dfaa48ffb1995bad57fa4bf6a14206a889bc0ba9))
- **py**: add internal zarr v2 bytes-mapping store reader ([99d3679](https://github.com/fideus-labs/ngff-zarr/commit/99d36798de621f1916d76b53a74eef3fdfd75bc8))
- **py**: resolve compat-layer store handles for RFC-9 zip creation ([fad75b3](https://github.com/fideus-labs/ngff-zarr/commit/fad75b337d58f9d61d3a101c43c68b259abfe3ad))
- **py**: upgrade OME-Zarr stores in place through zarrista ([7d70727](https://github.com/fideus-labs/ngff-zarr/commit/7d707274c16cb8caeaa823acb1f5e9b776187a37))
- **py**: write HCS plates through zarrista for path stores ([61f69bd](https://github.com/fideus-labs/ngff-zarr/commit/61f69bdfe4f5baefbd10570e89013bb0eea65f9e))
- **py**: route the to_multiscales disk cache through zarrista ([ddb0a58](https://github.com/fideus-labs/ngff-zarr/commit/ddb0a5849cdd07a28081c721e08f48dde9e48019))
- **py**: write arrays through zarrista for path stores ([217b68a](https://github.com/fideus-labs/ngff-zarr/commit/217b68ab950bc357482a39005650d90baca4ea3f))
- **py**: write group metadata through zarrista for path stores ([0d53be1](https://github.com/fideus-labs/ngff-zarr/commit/0d53be19f762f6c0de423c48db980f4177940be5))
- **py**: add use_zarrista_for backend dispatch helper ([d7a2598](https://github.com/fideus-labs/ngff-zarr/commit/d7a25986580676a77e1e0403044a85635dbc2e06))
- **py**: deprecate use_tensorstore in favor of the zarrista writer ([8326fad](https://github.com/fideus-labs/ngff-zarr/commit/8326fad2db1dacc694c89264851377e8fcca60d8))
- **py**: implement zarrista writer for the tensorstore fast path ([ff1e31f](https://github.com/fideus-labs/ngff-zarr/commit/ff1e31fdb0a04601067e05cb2cee3d54d4b05c67))
- **py**: add zarrista dependency and compatibility layer ([4796ab8](https://github.com/fideus-labs/ngff-zarr/commit/4796ab8ab4095f5bb5e0b65b875824a2f9ba828b))
- **py**: add itk_transform_resample for out-of-core resampling ([822d90f](https://github.com/fideus-labs/ngff-zarr/commit/822d90f1a8fff0c327984772c66e59276004c9f9))

### 🐛 Bug Fixes

- **py,ts**: refuse to write a transform the reader would reject ([36d3e92](https://github.com/fideus-labs/ngff-zarr/commit/36d3e924c08de5c697900a9d1eb522dd8a31539c))
- **py,ts**: hold projectAxis to the arity its schema declares ([71d52e7](https://github.com/fideus-labs/ngff-zarr/commit/71d52e7ef3d12a90f9f754d75a1f873fbe40555d))
- **py**: keep the requested factor when it reaches the target size ([0627c36](https://github.com/fideus-labs/ngff-zarr/commit/0627c36a1d13f92d39ce6b33b1437ae6c36ddafd))
- **py**: clamp a small array's shards and keep the consolidation state ([664ebb5](https://github.com/fideus-labs/ngff-zarr/commit/664ebb51600630929f0e2f9c5e0f4ecf63bdc2c3))
- **py**: read non-default axis types back onto the image ([f302cdc](https://github.com/fideus-labs/ngff-zarr/commit/f302cdccacc46bf54908518b684c96ee75bef529))
- **py**: guard itkwasm block sizes and trim bin-shrink remainders ([3c0188f](https://github.com/fideus-labs/ngff-zarr/commit/3c0188f610f487475f901ee722d644870da7e760))
- **py**: keep foreign root attributes across an overwrite ([91e4fac](https://github.com/fideus-labs/ngff-zarr/commit/91e4fac11212e1aeea82ff3d157d8434e2883949))
- **py**: write supplied scale levels as given ([e31124e](https://github.com/fideus-labs/ngff-zarr/commit/e31124e7b3a5f97780dd2aa3ba49778b1574b36a))
- **py**: refuse to overwrite the store a multiscales still reads from ([e862300](https://github.com/fideus-labs/ngff-zarr/commit/e862300bd603f8ee9e9d23a9a00e2c6d720898eb))
- **py,ts**: refuse the conversions that returned a wrong matrix silently ([967e438](https://github.com/fideus-labs/ngff-zarr/commit/967e4387c1dc4b1c21c5db30c116b2a8c54372f9))
- **py,ts**: follow the byDimension axis key rename ([945b590](https://github.com/fideus-labs/ngff-zarr/commit/945b590ab9e8746ff0b5618e26a326a7d0aad003))
- **py,ts**: recover an ITK affine exactly and bind sub-grid axes by name ([f756056](https://github.com/fideus-labs/ngff-zarr/commit/f756056f6bc3e1e6157ce61981484ef86b787034))
- **py,ts**: stop applying the v0.6 mapAxis arity to a 0.9.dev1 store ([ed4aee4](https://github.com/fideus-labs/ngff-zarr/commit/ed4aee4d8d0f9893d879686a6ec6a633450164ee))
- **py,ts**: carry the v0.6 transform rules over to 0.9.dev1 ([3393a2d](https://github.com/fideus-labs/ngff-zarr/commit/3393a2dda5761f4675095c0d89aa4c1a0dcfecfa))
- **py,ts**: gate the axes each version reads and writes ([a022dca](https://github.com/fideus-labs/ngff-zarr/commit/a022dca968a7f7fc13886e374e92df58d5f44ba5))
- **py,ts**: correct the axis model and gate the RFC-3 rules by version ([e418a39](https://github.com/fideus-labs/ngff-zarr/commit/e418a39dbb0f278a8eaaadab1d7d71e845515fa6))
- **py**: substitute only the 0.6 tags earlier releases wrote ([e8dec2a](https://github.com/fideus-labs/ngff-zarr/commit/e8dec2a08ee89eb4f470294142e7ada06cdc5a9f))
- **py**: validate the rest of a store that carries a superseded 0.6 tag ([edfd283](https://github.com/fideus-labs/ngff-zarr/commit/edfd2831683042f85bc3710288d84736e6037e20))
- **py,ts**: write byDimension and top-level transforms the 0.6rc0 schema accepts ([65a0e10](https://github.com/fideus-labs/ngff-zarr/commit/65a0e103b234b6138d0d5759802ae4b4c9efb007))
- **py**: make large-image cache paths unique per call ([05182ce](https://github.com/fideus-labs/ngff-zarr/commit/05182ce3f86c442969c6b8405db411043ee7bd44))
- **py**: tolerate the replace window when reading group documents ([1c87020](https://github.com/fideus-labs/ngff-zarr/commit/1c870202086ba6a51503c1766b168a7f2280c078))
- **py**: retry the atomic group-document replace on Windows ([ff16e17](https://github.com/fideus-labs/ngff-zarr/commit/ff16e173362a96ca1250e241e6045c643e345b95))
- **py**: address review findings in the zarrista read/write paths ([205fe87](https://github.com/fideus-labs/ngff-zarr/commit/205fe8714633328808766fd9f88b404162749435))
- **py**: convert TIFF inputs without zarr-python ([be129ab](https://github.com/fideus-labs/ngff-zarr/commit/be129ab7926215bed3d5c238a1b51071cca94660))
- **py**: name the read/write API to_ome_zarr and from_ome_zarr ([d3ccfcf](https://github.com/fideus-labs/ngff-zarr/commit/d3ccfcf0fd3c302937fcb87158e5f7fc25795a4c))
- **py**: render the v0.6 dataset transform instead of normalizing it ([40ec3e2](https://github.com/fideus-labs/ngff-zarr/commit/40ec3e2f74a8991eb9097e86cb3a84174264f025))
- **py**: validate the coordinate-systems metadata model ([4fedb2d](https://github.com/fideus-labs/ngff-zarr/commit/4fedb2d1335199888eeaeb3cea16468353541f41))
- **py**: carry RFC-4 orientation through block geometry ([4fb33c0](https://github.com/fideus-labs/ngff-zarr/commit/4fb33c0b4a11ae13d1641d2ebad8888eb54120d3))

## py-v0.43.0 (2026-08-21)

### ♻️ Refactoring

- **py,ts**: keep the RFC-4 vocabulary in one place per port ([02afb3f](https://github.com/fideus-labs/ngff-zarr/commit/02afb3f900ceb01b315f0e8bd2b2caeb99b68ecf))
- **py**: move transform validation onto the dataclasses ([813409c](https://github.com/fideus-labs/ngff-zarr/commit/813409c210ad2b9563680060fbc26b66eb3d993c))

### ✨ Features

- **py**: model the mapAxis, byDimension and bijection transforms ([bf28616](https://github.com/fideus-labs/ngff-zarr/commit/bf28616a7426fd64c815c3fef9c5fb5158e8cae6))

### 🐛 Bug Fixes

- **py,ts**: reject a non-string orientation value and pin the schema both ways ([81b095d](https://github.com/fideus-labs/ngff-zarr/commit/81b095ddd75d02d2811659091854e6f14bfee5ed))
- **py**: treat only a null or empty orientation as undefined ([db2840b](https://github.com/fideus-labs/ngff-zarr/commit/db2840b42b83a4387ae15e8c784e85e753fc07ca))
- **py**: resolve the intrinsic system by name and validate any orientation ([0c13821](https://github.com/fideus-labs/ngff-zarr/commit/0c138216a1d0d612b6a94db94a9ba7526afa2eae))
- **py**: read the axes wherever the version keeps them ([00cec97](https://github.com/fideus-labs/ngff-zarr/commit/00cec973be269554a5afbdaed482c36a26a5c229))
- **py**: narrow the version fallback and document validate() ([3cbb94c](https://github.com/fideus-labs/ngff-zarr/commit/3cbb94cd12b8648cb9f3d6224fc0653ba0796715))
- **py**: resolve cross-file JSON Schema references during validation ([f8bebf6](https://github.com/fideus-labs/ngff-zarr/commit/f8bebf64ef6cca65e27acf71b41d67aca4cdd1fa))
- **py**: preserve configured S3 CLI access ([82c3c80](https://github.com/fideus-labs/ngff-zarr/commit/82c3c80d44f91015329d28a6dd66c2c8bc8b3104))
- **py**: inspect public S3 stores anonymously ([8dc5596](https://github.com/fideus-labs/ngff-zarr/commit/8dc5596cb0411a98ed566a98057d9702d59472c2))
- **py**: reject transformation payloads that omit required fields ([522b552](https://github.com/fideus-labs/ngff-zarr/commit/522b55226713351b7f2540f52cb9807e95fa65ab))
- **py**: bound the mapAxis arity to 2 through 5 indices ([8515126](https://github.com/fideus-labs/ngff-zarr/commit/8515126c98daab7ef6e7a49d353722593bfa369e))
- **py**: preserve transform names and tighten axis validation ([2e0a1bd](https://github.com/fideus-labs/ngff-zarr/commit/2e0a1bd525008c775bc11cc5a17086fb5befe650))
- fix ci PyPI publish failure on metadata 2.5 wheels ([90c5bee](https://github.com/fideus-labs/ngff-zarr/commit/90c5bee484e8f128023a4cf7da71d2d320db9532))

## py-v0.42.0 (2026-08-13)

### ✨ Features

- **py**: add itk_transform_resample_bounding_box for out-of-core resampling ([ca87ef5](https://github.com/fideus-labs/ngff-zarr/commit/ca87ef59261585fb74c6c139a801a44298fec563))

### 🐛 Bug Fixes

- **py**: correct the declared value type of float32 ITK transforms ([86e3e69](https://github.com/fideus-labs/ngff-zarr/commit/86e3e6900bc7b79c556ffb3e78025481d15c9830))
- **py**: preserve the explicit codec chain on the TensorStore path ([1f85381](https://github.com/fideus-labs/ngff-zarr/commit/1f8538109f3410eb9d4c295846d39e17b653c42d))
- **py**: keep "pad" as the default scale strategy ([e21c914](https://github.com/fideus-labs/ngff-zarr/commit/e21c914230daea0a90e986d6d76b1ace785681a9))
- **py**: match zarr-python's codec handling on the TensorStore path ([979bb21](https://github.com/fideus-labs/ngff-zarr/commit/979bb21cb75ef1bc9c28cf3d9ad206f1af5d862b))
- **py**: default to the exact scale strategy and stop mutating the caller ([210a01f](https://github.com/fideus-labs/ngff-zarr/commit/210a01ff7cd7b77dbaa2ba24886751aea4d914c3))
- **py**: stop inheriting backend defaults and leaking into the caller ([6121aad](https://github.com/fideus-labs/ngff-zarr/commit/6121aadbc8026579d4246be838461a3ec05db777))

## py-v0.41.1 (2026-08-07)

### 🐛 Bug Fixes

- **py**: normalize generated axes to the spec order (t, c, z, y, x) ([358a491](https://github.com/fideus-labs/ngff-zarr/commit/358a491de374e4e3dd07ac0f956e975748fab982))
- **py**: normalize accelerated itkwasm downsampling output ([f5b9d94](https://github.com/fideus-labs/ngff-zarr/commit/f5b9d9494018d0804ab6d08ac4c65d4ddd487715))

## py-v0.41.0 (2026-08-04)

### ✨ Features

- **py**: add `ngff-zarr conformance` RFC 4 verdict subcommand ([7830d24](https://github.com/fideus-labs/ngff-zarr/commit/7830d244f907cf7de9308127f2be85bec27d774f))
- **py**: add ngff-zarr upgrade CLI subcommand ([0a426f9](https://github.com/fideus-labs/ngff-zarr/commit/0a426f9256e925f62274aab3f50b208ac3b59abd))
- **py**: add upgrade_ome_zarr() OME-Zarr spec-version upgrade ([c3bb7ed](https://github.com/fideus-labs/ngff-zarr/commit/c3bb7ed9efc2550597f474f7d8178306663043c2))

### 🐛 Bug Fixes

- **py**: count itemsize once in memory_usage, not once per dimension ([6b7fba7](https://github.com/fideus-labs/ngff-zarr/commit/6b7fba750867505fcf8ec8aab46a27da002fe6d1))
- RFC 4 orientation is optional per axis, null equals absent, type must be anatomical ([f4bf3dc](https://github.com/fideus-labs/ngff-zarr/commit/f4bf3dcb4446678e17135a95305b379c3fb9fe0a))
- **py**: correct v0.5 to_version return type to cross-version union ([dae2de5](https://github.com/fideus-labs/ngff-zarr/commit/dae2de50e101b6e8d267eb0dc5e73c187af980cf))
- **py**: install prek hooks with --overwrite ([af76338](https://github.com/fideus-labs/ngff-zarr/commit/af763383b461d0f969ef727388b0c9e7183a34ef))
- **ci**: use supported 'prek update' in autoupdate workflow ([8f2ee83](https://github.com/fideus-labs/ngff-zarr/commit/8f2ee83af4af48f096fc0272abdeb303646c3768))
- reject RFC 4 orientation on non-spatial axes and duplicate anatomical axes ([45de14c](https://github.com/fideus-labs/ngff-zarr/commit/45de14c0ed62a0f0d68d56790f1a71482cb9c221))

## py-v0.39.0 (2026-07-17)

### ♻️ Refactoring

- **py**: enumerate shard divisors in O(sqrt) and refresh clamp comments ([431aa04](https://github.com/fideus-labs/ngff-zarr/commit/431aa0474fc9559c7da552e9bcfb233bf96cf669))

### ✨ Features

- **py,ts**: write an image and its transformation into one store ([be3f6cd](https://github.com/fideus-labs/ngff-zarr/commit/be3f6cd54d622559fac1049e0c8ea6194e7c1ef4))

### 🐛 Bug Fixes

- **py**: floor sharded inner chunk to avoid 2*prime collapse ([3bf12dd](https://github.com/fideus-labs/ngff-zarr/commit/3bf12dd3996e67104013099f6cf2a608596a4e61))
- **py**: clamp chunk/shard sizes to the axis instead of snapping to a divisor ([7cab47b](https://github.com/fideus-labs/ngff-zarr/commit/7cab47b0509e08737704f996266a52d1f654ac46))
- **py**: step region writes by the band width so the whole array is written ([9ccf5c8](https://github.com/fideus-labs/ngff-zarr/commit/9ccf5c84460a76fda149f3bda61d9e8a7b940752))

## py-v0.38.0 (2026-07-09)

### ♻️ Refactoring

- **py,ts**: move v0.6 axis types to v06 and set axes_types on NgffImage ([54eaed4](https://github.com/fideus-labs/ngff-zarr/commit/54eaed4d2e4b23de93f68a3c15723fc14893244e))

### ✨ Features

- **py,ts**: carry the v0.6 discrete axis flag and address review feedback ([5203864](https://github.com/fideus-labs/ngff-zarr/commit/5203864efefcd1ba38b42c3e4976c542ae984fcf))
- **py,ts**: support OME-Zarr v0.6 displacement and coordinate fields ([c181cc1](https://github.com/fideus-labs/ngff-zarr/commit/c181cc189edc142f2add7f0df3a3ad865421f0f8))

## py-v0.37.1 (2026-07-03)

### 🐛 Bug Fixes

- **py**: add py.typed marker for PEP 561 compliance ([5ccf7da](https://github.com/fideus-labs/ngff-zarr/commit/5ccf7da3eb2634f43554ebdea5b438bea731b0c9))

## py-v0.37.0 (2026-06-25)

### BREAKING CHANGE

- The `enabled_rfcs` parameter (Python `to_ngff_zarr`/
`to_ome_zarr`), the `--enable-rfc` CLI flag, the `enabledRfcs` TypeScript write
option, and the MCP `enable_rfc4`/`enabled_rfcs` fields are removed. RFC-4
anatomical orientation is written automatically when orientation metadata is
present; supply it via `NgffImage.axes_orientations`, the new `orientation=`
argument to `to_multiscales`, or the `--orientation` CLI flag. ([abda351](https://github.com/fideus-labs/ngff-zarr/commit/abda351de31305f87a6119102e27b55e6baaecdb))

### ♻️ Refactoring

- **py,ts**: write RFC-4 anatomical orientation automatically, drop enabled-rfcs API ([abda351](https://github.com/fideus-labs/ngff-zarr/commit/abda351de31305f87a6119102e27b55e6baaecdb))
- update to refactored read/write functions ([39318c0](https://github.com/fideus-labs/ngff-zarr/commit/39318c07cacbba81238174ade43febf1426117f2))
- Introduce CoordinateSystemIdentifier metadata class ([52df14f](https://github.com/fideus-labs/ngff-zarr/commit/52df14fecb4406a05254dd053f3b11e6039c138d))
- improve typing of omero parser ([25ef680](https://github.com/fideus-labs/ngff-zarr/commit/25ef68023a16b74ceefc1baff9155d39d1618774))
- Make rfc4 checks compliant with v0.6 metadata structure ([83d0f52](https://github.com/fideus-labs/ngff-zarr/commit/83d0f52a9480086167264f207a7726f425442120))
- add metadata reading into class methods ([7f0923c](https://github.com/fideus-labs/ngff-zarr/commit/7f0923c91a6b987297a1964347ea026d54c39579))

### ✨ Features

- 0.6.dev4 on write ([7db2227](https://github.com/fideus-labs/ngff-zarr/commit/7db2227d11b02ced680531fddee4c0bf01876057))
- allow reading dev4 versions ([50a9434](https://github.com/fideus-labs/ngff-zarr/commit/50a943448196d5d3019ce1675c7f22cb2394b603))
- pin higher zarr ([4d5269f](https://github.com/fideus-labs/ngff-zarr/commit/4d5269f60128dee0cfdb884607a61d77af73b941))
- import missing metadata class ([b6bd757](https://github.com/fideus-labs/ngff-zarr/commit/b6bd757190c5cc19011015cddbdb7fc33440a6fb))
- move up metadata conversion to retain method metadata ([4c0b829](https://github.com/fideus-labs/ngff-zarr/commit/4c0b829e4ea959da4d8ec08ab3a09bf187ca2bc1))
- introduce convenience method to validate intrinsic coordinate system ([c4347d5](https://github.com/fideus-labs/ngff-zarr/commit/c4347d5de982d73872f6b785b0c9c4e2703cbe9c))
- update to 0.6-style input output convention ([9446760](https://github.com/fideus-labs/ngff-zarr/commit/9446760f3a12fa74f35c873dba12f927d0262aae))
- use correct call to class attribute ([7d70b7e](https://github.com/fideus-labs/ngff-zarr/commit/7d70b7e405ede0234f4fe1d5392ae4530bdbcb52))
- add escape for missing coordinate systems ([ca1d0ae](https://github.com/fideus-labs/ngff-zarr/commit/ca1d0ae2300ed4a192368da7f0c4c095c3017f51))
- add missing, declared conversion from v06 to v04 ([b37e210](https://github.com/fideus-labs/ngff-zarr/commit/b37e210e676f6300dff3b5338cb16884ffb6f5e3))
- always treat scale/translation variables as class instances ([f2d9cbd](https://github.com/fideus-labs/ngff-zarr/commit/f2d9cbd010badd8ab058bf4a435186ff20b83d96))
- update to latest schemas ([bfea702](https://github.com/fideus-labs/ngff-zarr/commit/bfea7027db13ca90f37975c56b969a89c3d054f8))
- add schemas (preliminary) ([300f389](https://github.com/fideus-labs/ngff-zarr/commit/300f38955510b382a201f93bb1d4e119d676edb9))
- Add utility parser function for transforms ([eb6915a](https://github.com/fideus-labs/ngff-zarr/commit/eb6915ae8e0df05e0980607afaaa0091999b6567))
- Support 0.6 reading and writing ([21a54c6](https://github.com/fideus-labs/ngff-zarr/commit/21a54c67150c5ad8ca83163121aeb07eb1429e4a))

### 🐛 Bug Fixes

- **py**: re-open LIF source at compute time to survive closed file handle ([618c838](https://github.com/fideus-labs/ngff-zarr/commit/618c838de1e3bf5dce4c572234dd9325dc9901c7))
- **py**: map big-endian dtypes to Zarr v3 core dtype ([04890f6](https://github.com/fideus-labs/ngff-zarr/commit/04890f60291212c836063590f04250f101459d5d))
- **py**: carry metadata extra passthrough through v0.5/v0.6 conversion ([0315ad0](https://github.com/fideus-labs/ngff-zarr/commit/0315ad0a48e2e816985729c9a9efdefac486b3cf))
- get omero metadata from correct location ([a8bc4f4](https://github.com/fideus-labs/ngff-zarr/commit/a8bc4f401a502c0a43997d7fafae340cd38ad2b4))
- fix type annotation ([a656eba](https://github.com/fideus-labs/ngff-zarr/commit/a656ebae38fb8de0b15b72a4e8f34d2779179f55))

## py-v0.36.2 (2026-06-19)

### 🐛 Bug Fixes

- use >= in to_multiscales loop to handle exact chunk-size multiples ([53d6f08](https://github.com/fideus-labs/ngff-zarr/commit/53d6f088584ff8c4cf7d161cce03272d2ff6bcc6))

## py-v0.36.1 (2026-06-18)

### 🐛 Bug Fixes

- **py**: support reading remote OME-Zarr out of the box via a [remote] extra ([35cbd65](https://github.com/fideus-labs/ngff-zarr/commit/35cbd658c69c96e73a4f6e58f462ee711583b484))

## py-v0.36.0 (2026-06-18)

### ✨ Features

- **py,ts**: add OME-Zarr v0.5 structural validation and schema support ([598ed91](https://github.com/fideus-labs/ngff-zarr/commit/598ed91e6f3b15533d9a50a59b8cd68b3674e22f))
- **py,ts**: extend RFC-4 vocabulary with subject-local tissue orientations ([0cc4a91](https://github.com/fideus-labs/ngff-zarr/commit/0cc4a9146fb9113d0a7c289f7787abdd825e362d))

## py-v0.35.1 (2026-06-11)

### 🐛 Bug Fixes

- **py**: give actionable errors for corrupt TIFF/SVS conversions ([5797e3b](https://github.com/fideus-labs/ngff-zarr/commit/5797e3b4ec81595950a80df8b178a39c6ce4bc7a))

## py-v0.35.0 (2026-06-04)

### ♻️ Refactoring

- **py**: cull all None metadata fields recursively ([159add6](https://github.com/fideus-labs/ngff-zarr/commit/159add6ae428634cacce8ba13d5a06104c048cd1))

### ✨ Features

- **py**: add OME-Zarr v0.4 structural validation ([d6de82b](https://github.com/fideus-labs/ngff-zarr/commit/d6de82b25ea5207c6b3e201013914a2d6062f5e3))

### 🐛 Bug Fixes

- **py**: degrade gracefully when stdout cannot be reconfigured ([89bfcb2](https://github.com/fideus-labs/ngff-zarr/commit/89bfcb2a8ef39176927d0912c5778e77b56cf8d5))
- **py**: prevent CLI crash on legacy Windows terminal encodings ([b842050](https://github.com/fideus-labs/ngff-zarr/commit/b842050fc127c9dbbcf635a58445538bf840e845))
- **ci**: remove OTLP import from daily-doc-updater workflow ([45394c8](https://github.com/fideus-labs/ngff-zarr/commit/45394c8ccb09b5ad3687d8dd834c94af45d69469))
- **py**: create requested .ozx for multi-series TIFFs, warn on truncated reads ([25cc73a](https://github.com/fideus-labs/ngff-zarr/commit/25cc73a5153f9974ba43e69c52e20342481bcd33))
- **py**: strip deprecated nthreads/blocksize from blosc codec config when reading zarr v3 files ([572c19a](https://github.com/fideus-labs/ngff-zarr/commit/572c19a6fda86382a6bb7ae3c1b6da9950115033))
- **test**: handle pre-existing file cleanup in overwrite directory test fixture ([53fc468](https://github.com/fideus-labs/ngff-zarr/commit/53fc468500228501450283689c1874ce6178d688))
- **py**: handle pre-existing directory at .ozx destination on Windows ([8c85fe4](https://github.com/fideus-labs/ngff-zarr/commit/8c85fe40610491038807609391225034c045e3da))

## py-v0.34.2 (2026-05-28)

### 🐛 Bug Fixes

- stop pre-commit from mangling gh-aw generated files ([d6f1201](https://github.com/fideus-labs/ngff-zarr/commit/d6f120144884bc6a3089e5286a624ed0316f90ba))

## py-v0.34.1 (2026-05-27)

### 🐛 Bug Fixes

- **from-ome-zarr**: better error routing ([095ce7c](https://github.com/fideus-labs/ngff-zarr/commit/095ce7c1e7cacae75727c11bd27e3d187218cc45))
- **py**: add helpful error messages for GroupNotFoundError in from_ngff_zarr ([89552ac](https://github.com/fideus-labs/ngff-zarr/commit/89552ac488588c94ab5cabfde07b852b85705748))
- **py**: use arr.chunksize for v0.5 chunks forwarding ([9dcdf07](https://github.com/fideus-labs/ngff-zarr/commit/9dcdf076012b907bbbceb576cd81f5e546f8c5a9))
- **py**: resolve data loss and chunks regressions from issue #488 ([c93fe01](https://github.com/fideus-labs/ngff-zarr/commit/c93fe012b8451e1409aa679376419f456ea04416))
- **py**: apply _find_optimal_chunk_size to spatial dims only in z-slabs path ([0fbdb0c](https://github.com/fideus-labs/ngff-zarr/commit/0fbdb0c3f8bbe7af2efe09614595e66246fecdd0))
- **py**: remove dask.array.optimize to prevent chunk structure fusion ([ab3bbfd](https://github.com/fideus-labs/ngff-zarr/commit/ab3bbfd50ba88493a68028329906d45ed96c93a3))
- **py**: align 2D strip and 1D segment cache chunks with region write boundaries ([f398ce3](https://github.com/fideus-labs/ngff-zarr/commit/f398ce35e8c501baac3aac5d1d2b8ad42507d65f))
- **py**: align cache zarr chunks with dask data chunks to prevent PerformanceWarning ([8b2fdd6](https://github.com/fideus-labs/ngff-zarr/commit/8b2fdd62db316fa815f92da37a670c0f663e70a5))
- **py**: prevent zarr_kwargs mutation across scale loop iterations in _handle_large_array_writing ([f1f88d7](https://github.com/fideus-labs/ngff-zarr/commit/f1f88d752fdb54f767a60bfc7548cf396c0fb7b7))
- **py**: remove retry_if_failed from pooch downloader ([2f9bda2](https://github.com/fideus-labs/ngff-zarr/commit/2f9bda2f909740e77a2be5707b4f8726d4beee64))
- **py**: add retry with longer timeout to test data download ([8d9b490](https://github.com/fideus-labs/ngff-zarr/commit/8d9b49031eb9d65f5fdd46b56313889c19330375))
- **py**: use create_array for zarr v3 compatibility in rfc4 validation tests ([6b9c45f](https://github.com/fideus-labs/ngff-zarr/commit/6b9c45fb4e0210a247e1d1a781343182204e71f3))
- **ci**: lower imagecodecs lower bound for Python 3.10 support ([74765a8](https://github.com/fideus-labs/ngff-zarr/commit/74765a8c10e27ddce7ed3c0ebc8e15b01501a9aa))
- **ci**: pin imagecodecs to v2026.3.6 to prevent macOS Intel build failure ([3452bea](https://github.com/fideus-labs/ngff-zarr/commit/3452bea724b4c5089985143e17dc1ae410dbbc80))

## py-v0.40.0 (2026-07-24)

### ✨ Features

- **py**: add ngff-zarr upgrade CLI subcommand ([0a426f9](https://github.com/fideus-labs/ngff-zarr/commit/0a426f9256e925f62274aab3f50b208ac3b59abd))
- **py**: add upgrade_ome_zarr() OME-Zarr spec-version upgrade ([c3bb7ed](https://github.com/fideus-labs/ngff-zarr/commit/c3bb7ed9efc2550597f474f7d8178306663043c2))

### 🐛 Bug Fixes

- **py**: correct v0.5 to_version return type to cross-version union ([dae2de5](https://github.com/fideus-labs/ngff-zarr/commit/dae2de50e101b6e8d267eb0dc5e73c187af980cf))
- **py**: install prek hooks with --overwrite ([af76338](https://github.com/fideus-labs/ngff-zarr/commit/af763383b461d0f969ef727388b0c9e7183a34ef))
- **ci**: use supported 'prek update' in autoupdate workflow ([8f2ee83](https://github.com/fideus-labs/ngff-zarr/commit/8f2ee83af4af48f096fc0272abdeb303646c3768))
- reject RFC 4 orientation on non-spatial axes and duplicate anatomical axes ([45de14c](https://github.com/fideus-labs/ngff-zarr/commit/45de14c0ed62a0f0d68d56790f1a71482cb9c221))

## py-v0.39.0 (2026-07-17)

### ♻️ Refactoring

- **py**: enumerate shard divisors in O(sqrt) and refresh clamp comments ([431aa04](https://github.com/fideus-labs/ngff-zarr/commit/431aa0474fc9559c7da552e9bcfb233bf96cf669))

### ✨ Features

- **py,ts**: write an image and its transformation into one store ([be3f6cd](https://github.com/fideus-labs/ngff-zarr/commit/be3f6cd54d622559fac1049e0c8ea6194e7c1ef4))

### 🐛 Bug Fixes

- **py**: floor sharded inner chunk to avoid 2*prime collapse ([3bf12dd](https://github.com/fideus-labs/ngff-zarr/commit/3bf12dd3996e67104013099f6cf2a608596a4e61))
- **py**: clamp chunk/shard sizes to the axis instead of snapping to a divisor ([7cab47b](https://github.com/fideus-labs/ngff-zarr/commit/7cab47b0509e08737704f996266a52d1f654ac46))
- **py**: step region writes by the band width so the whole array is written ([9ccf5c8](https://github.com/fideus-labs/ngff-zarr/commit/9ccf5c84460a76fda149f3bda61d9e8a7b940752))

## py-v0.38.0 (2026-07-09)

### ♻️ Refactoring

- **py,ts**: move v0.6 axis types to v06 and set axes_types on NgffImage ([54eaed4](https://github.com/fideus-labs/ngff-zarr/commit/54eaed4d2e4b23de93f68a3c15723fc14893244e))

### ✨ Features

- **py,ts**: carry the v0.6 discrete axis flag and address review feedback ([5203864](https://github.com/fideus-labs/ngff-zarr/commit/5203864efefcd1ba38b42c3e4976c542ae984fcf))
- **py,ts**: support OME-Zarr v0.6 displacement and coordinate fields ([c181cc1](https://github.com/fideus-labs/ngff-zarr/commit/c181cc189edc142f2add7f0df3a3ad865421f0f8))

## py-v0.37.1 (2026-07-03)

### 🐛 Bug Fixes

- **py**: add py.typed marker for PEP 561 compliance ([5ccf7da](https://github.com/fideus-labs/ngff-zarr/commit/5ccf7da3eb2634f43554ebdea5b438bea731b0c9))

## py-v0.37.0 (2026-06-25)

### BREAKING CHANGE

- The `enabled_rfcs` parameter (Python `to_ngff_zarr`/
`to_ome_zarr`), the `--enable-rfc` CLI flag, the `enabledRfcs` TypeScript write
option, and the MCP `enable_rfc4`/`enabled_rfcs` fields are removed. RFC-4
anatomical orientation is written automatically when orientation metadata is
present; supply it via `NgffImage.axes_orientations`, the new `orientation=`
argument to `to_multiscales`, or the `--orientation` CLI flag. ([abda351](https://github.com/fideus-labs/ngff-zarr/commit/abda351de31305f87a6119102e27b55e6baaecdb))

### ♻️ Refactoring

- **py,ts**: write RFC-4 anatomical orientation automatically, drop enabled-rfcs API ([abda351](https://github.com/fideus-labs/ngff-zarr/commit/abda351de31305f87a6119102e27b55e6baaecdb))
- update to refactored read/write functions ([39318c0](https://github.com/fideus-labs/ngff-zarr/commit/39318c07cacbba81238174ade43febf1426117f2))
- Introduce CoordinateSystemIdentifier metadata class ([52df14f](https://github.com/fideus-labs/ngff-zarr/commit/52df14fecb4406a05254dd053f3b11e6039c138d))
- improve typing of omero parser ([25ef680](https://github.com/fideus-labs/ngff-zarr/commit/25ef68023a16b74ceefc1baff9155d39d1618774))
- Make rfc4 checks compliant with v0.6 metadata structure ([83d0f52](https://github.com/fideus-labs/ngff-zarr/commit/83d0f52a9480086167264f207a7726f425442120))
- add metadata reading into class methods ([7f0923c](https://github.com/fideus-labs/ngff-zarr/commit/7f0923c91a6b987297a1964347ea026d54c39579))

### ✨ Features

- 0.6.dev4 on write ([7db2227](https://github.com/fideus-labs/ngff-zarr/commit/7db2227d11b02ced680531fddee4c0bf01876057))
- allow reading dev4 versions ([50a9434](https://github.com/fideus-labs/ngff-zarr/commit/50a943448196d5d3019ce1675c7f22cb2394b603))
- pin higher zarr ([4d5269f](https://github.com/fideus-labs/ngff-zarr/commit/4d5269f60128dee0cfdb884607a61d77af73b941))
- import missing metadata class ([b6bd757](https://github.com/fideus-labs/ngff-zarr/commit/b6bd757190c5cc19011015cddbdb7fc33440a6fb))
- move up metadata conversion to retain method metadata ([4c0b829](https://github.com/fideus-labs/ngff-zarr/commit/4c0b829e4ea959da4d8ec08ab3a09bf187ca2bc1))
- introduce convenience method to validate intrinsic coordinate system ([c4347d5](https://github.com/fideus-labs/ngff-zarr/commit/c4347d5de982d73872f6b785b0c9c4e2703cbe9c))
- update to 0.6-style input output convention ([9446760](https://github.com/fideus-labs/ngff-zarr/commit/9446760f3a12fa74f35c873dba12f927d0262aae))
- use correct call to class attribute ([7d70b7e](https://github.com/fideus-labs/ngff-zarr/commit/7d70b7e405ede0234f4fe1d5392ae4530bdbcb52))
- add escape for missing coordinate systems ([ca1d0ae](https://github.com/fideus-labs/ngff-zarr/commit/ca1d0ae2300ed4a192368da7f0c4c095c3017f51))
- add missing, declared conversion from v06 to v04 ([b37e210](https://github.com/fideus-labs/ngff-zarr/commit/b37e210e676f6300dff3b5338cb16884ffb6f5e3))
- always treat scale/translation variables as class instances ([f2d9cbd](https://github.com/fideus-labs/ngff-zarr/commit/f2d9cbd010badd8ab058bf4a435186ff20b83d96))
- update to latest schemas ([bfea702](https://github.com/fideus-labs/ngff-zarr/commit/bfea7027db13ca90f37975c56b969a89c3d054f8))
- add schemas (preliminary) ([300f389](https://github.com/fideus-labs/ngff-zarr/commit/300f38955510b382a201f93bb1d4e119d676edb9))
- Add utility parser function for transforms ([eb6915a](https://github.com/fideus-labs/ngff-zarr/commit/eb6915ae8e0df05e0980607afaaa0091999b6567))
- Support 0.6 reading and writing ([21a54c6](https://github.com/fideus-labs/ngff-zarr/commit/21a54c67150c5ad8ca83163121aeb07eb1429e4a))

### 🐛 Bug Fixes

- **py**: re-open LIF source at compute time to survive closed file handle ([618c838](https://github.com/fideus-labs/ngff-zarr/commit/618c838de1e3bf5dce4c572234dd9325dc9901c7))
- **py**: map big-endian dtypes to Zarr v3 core dtype ([04890f6](https://github.com/fideus-labs/ngff-zarr/commit/04890f60291212c836063590f04250f101459d5d))
- **py**: carry metadata extra passthrough through v0.5/v0.6 conversion ([0315ad0](https://github.com/fideus-labs/ngff-zarr/commit/0315ad0a48e2e816985729c9a9efdefac486b3cf))
- get omero metadata from correct location ([a8bc4f4](https://github.com/fideus-labs/ngff-zarr/commit/a8bc4f401a502c0a43997d7fafae340cd38ad2b4))
- fix type annotation ([a656eba](https://github.com/fideus-labs/ngff-zarr/commit/a656ebae38fb8de0b15b72a4e8f34d2779179f55))

## py-v0.36.2 (2026-06-19)

### 🐛 Bug Fixes

- use >= in to_multiscales loop to handle exact chunk-size multiples ([53d6f08](https://github.com/fideus-labs/ngff-zarr/commit/53d6f088584ff8c4cf7d161cce03272d2ff6bcc6))

## py-v0.36.1 (2026-06-18)

### 🐛 Bug Fixes

- **py**: support reading remote OME-Zarr out of the box via a [remote] extra ([35cbd65](https://github.com/fideus-labs/ngff-zarr/commit/35cbd658c69c96e73a4f6e58f462ee711583b484))

## py-v0.36.0 (2026-06-18)

### ✨ Features

- **py,ts**: add OME-Zarr v0.5 structural validation and schema support ([598ed91](https://github.com/fideus-labs/ngff-zarr/commit/598ed91e6f3b15533d9a50a59b8cd68b3674e22f))
- **py,ts**: extend RFC-4 vocabulary with subject-local tissue orientations ([0cc4a91](https://github.com/fideus-labs/ngff-zarr/commit/0cc4a9146fb9113d0a7c289f7787abdd825e362d))

## py-v0.35.1 (2026-06-11)

### 🐛 Bug Fixes

- **py**: give actionable errors for corrupt TIFF/SVS conversions ([5797e3b](https://github.com/fideus-labs/ngff-zarr/commit/5797e3b4ec81595950a80df8b178a39c6ce4bc7a))

## py-v0.35.0 (2026-06-04)

### ♻️ Refactoring

- **py**: cull all None metadata fields recursively ([159add6](https://github.com/fideus-labs/ngff-zarr/commit/159add6ae428634cacce8ba13d5a06104c048cd1))

### ✨ Features

- **py**: add OME-Zarr v0.4 structural validation ([d6de82b](https://github.com/fideus-labs/ngff-zarr/commit/d6de82b25ea5207c6b3e201013914a2d6062f5e3))

### 🐛 Bug Fixes

- **py**: degrade gracefully when stdout cannot be reconfigured ([89bfcb2](https://github.com/fideus-labs/ngff-zarr/commit/89bfcb2a8ef39176927d0912c5778e77b56cf8d5))
- **py**: prevent CLI crash on legacy Windows terminal encodings ([b842050](https://github.com/fideus-labs/ngff-zarr/commit/b842050fc127c9dbbcf635a58445538bf840e845))
- **ci**: remove OTLP import from daily-doc-updater workflow ([45394c8](https://github.com/fideus-labs/ngff-zarr/commit/45394c8ccb09b5ad3687d8dd834c94af45d69469))
- **py**: create requested .ozx for multi-series TIFFs, warn on truncated reads ([25cc73a](https://github.com/fideus-labs/ngff-zarr/commit/25cc73a5153f9974ba43e69c52e20342481bcd33))
- **py**: strip deprecated nthreads/blocksize from blosc codec config when reading zarr v3 files ([572c19a](https://github.com/fideus-labs/ngff-zarr/commit/572c19a6fda86382a6bb7ae3c1b6da9950115033))
- **test**: handle pre-existing file cleanup in overwrite directory test fixture ([53fc468](https://github.com/fideus-labs/ngff-zarr/commit/53fc468500228501450283689c1874ce6178d688))
- **py**: handle pre-existing directory at .ozx destination on Windows ([8c85fe4](https://github.com/fideus-labs/ngff-zarr/commit/8c85fe40610491038807609391225034c045e3da))

## py-v0.34.2 (2026-05-28)

### 🐛 Bug Fixes

- stop pre-commit from mangling gh-aw generated files ([d6f1201](https://github.com/fideus-labs/ngff-zarr/commit/d6f120144884bc6a3089e5286a624ed0316f90ba))

## py-v0.34.1 (2026-05-27)

### 🐛 Bug Fixes

- **from-ome-zarr**: better error routing ([095ce7c](https://github.com/fideus-labs/ngff-zarr/commit/095ce7c1e7cacae75727c11bd27e3d187218cc45))
- **py**: add helpful error messages for GroupNotFoundError in from_ngff_zarr ([89552ac](https://github.com/fideus-labs/ngff-zarr/commit/89552ac488588c94ab5cabfde07b852b85705748))
- **py**: use arr.chunksize for v0.5 chunks forwarding ([9dcdf07](https://github.com/fideus-labs/ngff-zarr/commit/9dcdf076012b907bbbceb576cd81f5e546f8c5a9))
- **py**: resolve data loss and chunks regressions from issue #488 ([c93fe01](https://github.com/fideus-labs/ngff-zarr/commit/c93fe012b8451e1409aa679376419f456ea04416))
- **py**: apply _find_optimal_chunk_size to spatial dims only in z-slabs path ([0fbdb0c](https://github.com/fideus-labs/ngff-zarr/commit/0fbdb0c3f8bbe7af2efe09614595e66246fecdd0))
- **py**: remove dask.array.optimize to prevent chunk structure fusion ([ab3bbfd](https://github.com/fideus-labs/ngff-zarr/commit/ab3bbfd50ba88493a68028329906d45ed96c93a3))
- **py**: align 2D strip and 1D segment cache chunks with region write boundaries ([f398ce3](https://github.com/fideus-labs/ngff-zarr/commit/f398ce35e8c501baac3aac5d1d2b8ad42507d65f))
- **py**: align cache zarr chunks with dask data chunks to prevent PerformanceWarning ([8b2fdd6](https://github.com/fideus-labs/ngff-zarr/commit/8b2fdd62db316fa815f92da37a670c0f663e70a5))
- **py**: prevent zarr_kwargs mutation across scale loop iterations in _handle_large_array_writing ([f1f88d7](https://github.com/fideus-labs/ngff-zarr/commit/f1f88d752fdb54f767a60bfc7548cf396c0fb7b7))
- **py**: remove retry_if_failed from pooch downloader ([2f9bda2](https://github.com/fideus-labs/ngff-zarr/commit/2f9bda2f909740e77a2be5707b4f8726d4beee64))
- **py**: add retry with longer timeout to test data download ([8d9b490](https://github.com/fideus-labs/ngff-zarr/commit/8d9b49031eb9d65f5fdd46b56313889c19330375))
- **py**: use create_array for zarr v3 compatibility in rfc4 validation tests ([6b9c45f](https://github.com/fideus-labs/ngff-zarr/commit/6b9c45fb4e0210a247e1d1a781343182204e71f3))
- **ci**: lower imagecodecs lower bound for Python 3.10 support ([74765a8](https://github.com/fideus-labs/ngff-zarr/commit/74765a8c10e27ddce7ed3c0ebc8e15b01501a9aa))
- **ci**: pin imagecodecs to v2026.3.6 to prevent macOS Intel build failure ([3452bea](https://github.com/fideus-labs/ngff-zarr/commit/3452bea724b4c5089985143e17dc1ae410dbbc80))

## py-v0.34.0 (2026-04-15)

### ♻️ Refactoring

- rename `Multiscales` to `NgffMultiscales` across py, ts, mcp ([1d0c572](https://github.com/fideus-labs/ngff-zarr/commit/1d0c5724fdd7ba86679f57b3343332a44d1aff5d))

### ✨ Features

- **py,ts**: rename from_ngff_zarr/fromNgffZarr to from_ome_zarr/fromOmeZarr ([84a20b2](https://github.com/fideus-labs/ngff-zarr/commit/84a20b28bce135e4009197b7671fa9ad21e772a0))
- **py,ts**: rename to_ngff_zarr/toNgffZarr to to_ome_zarr/toOmeZarr ([9431546](https://github.com/fideus-labs/ngff-zarr/commit/94315466d03b7c3e73a3e42fc7e236c428f1d95c))
- **py**: add --codec CLI option and shared codec utilities across all packages ([b1c60ee](https://github.com/fideus-labs/ngff-zarr/commit/b1c60ee51482bce4cf5db4ac76f64c91d7195373))
- update default OME-Zarr version from 0.4 to 0.5 ([4e60c07](https://github.com/fideus-labs/ngff-zarr/commit/4e60c077b3adcf6f4ef827a1cc112ca8e340f22c))
- **py,ts**: add bidirectional anatomical orientation ↔ ITK direction matrix conversion ([e427ca4](https://github.com/fideus-labs/ngff-zarr/commit/e427ca4e5e131a97b9b566a5fd79ac6020110dd9))

### 🐛 Bug Fixes

- **py**: handle both v2 and v3 array metadata in dimension separator test ([575a814](https://github.com/fideus-labs/ngff-zarr/commit/575a8141792cf1daa14e18809036f960125a7d46))
- **py**: open dimension separator test array directly by path ([4345a0a](https://github.com/fideus-labs/ngff-zarr/commit/4345a0a055af6f0ffe9a1e150bc25519ab3384b0))
- **py**: add use_consolidated=False to open_group in dimension separator test ([e1a6405](https://github.com/fideus-labs/ngff-zarr/commit/e1a64052670c7e71c50c72eda061a79a454796d2))
- **py**: use open_group instead of open_array in dimension separator test ([617bc33](https://github.com/fideus-labs/ngff-zarr/commit/617bc334d8e9a8f1bb88841a50f149098613a405))
- **py**: fall back to zarr format 2 when auto-detecting version yields empty attrs ([2a66039](https://github.com/fideus-labs/ngff-zarr/commit/2a660391dc8ffdb9cb206b9e28aaa1eb2a31b293))
- resolve merge conflicts with main branch ([af2a8eb](https://github.com/fideus-labs/ngff-zarr/commit/af2a8eb4c5e4aa995d7afd79e92a574549db3dfc))
- **py**: add zarr v3 skip marker to test_bin_shrink_map_blocks_fast_path.py ([22a70f2](https://github.com/fideus-labs/ngff-zarr/commit/22a70f2c8a7f6c7b58afaf7f43272d810e725a0c))
- **py**: fix CI failures - update test metadata access for v0.5 and add zarr v3 skip markers ([61d21e5](https://github.com/fideus-labs/ngff-zarr/commit/61d21e569beb21b040206df3c212ee3f59d0537b))
- **py**: fix test_legacy_zarr_without_type to preserve list cardinality and reliably persist mutations ([a2c1dd0](https://github.com/fideus-labs/ngff-zarr/commit/a2c1dd082f427bfcd72cd02725e1fcecda7620b0))
- **py**: improve cli support for additional protocols ([07cc062](https://github.com/fideus-labs/ngff-zarr/commit/07cc062a6d4ceb60a45373fb333a6a2eb2da4733))
- **py**: resolve relative paths to absolute in CLI for Windows compatibility ([101eb2a](https://github.com/fideus-labs/ngff-zarr/commit/101eb2a0d2d733d7ebe35b1cd15d9beb53aacc9e))
- **py**: correctly calculate regions in to_ngff_zarr ([76772df](https://github.com/fideus-labs/ngff-zarr/commit/76772df11b7cfc860b38bc3c9ab5b872453e4481))
- **py**: always use RGB colors for S-axis images, override OME-XML ([b34c700](https://github.com/fideus-labs/ngff-zarr/commit/b34c700606e3b08809aed3f89244109b72840ffb))
- **py**: make level-selection test deterministic and CI-safe ([71e916d](https://github.com/fideus-labs/ngff-zarr/commit/71e916da84a6bbbc98ad0a8e51b118dc3d73a3ad))
- **py**: fix wrong OZX display for brightfield RGB images ([6f4f66c](https://github.com/fideus-labs/ngff-zarr/commit/6f4f66c34c67630a4db583d370513defa301b24e))

## py-v0.31.1 (2026-03-10)

### ♻️ Refactoring

- **py**: deterministic ozx content order ([597cd4f](https://github.com/fideus-labs/ngff-zarr/commit/597cd4f7be54dbb47bd8b53418f7fec5029ee2e5))

### ⚡ Performance

- **py**: add map_blocks fast path for bin-shrink downsampling ([54dd014](https://github.com/fideus-labs/ngff-zarr/commit/54dd014dbb96bbd71e58ec9c27369b18f30f4aef))
- **py**: avoid loading full file into memory for ozx ([04e72af](https://github.com/fideus-labs/ngff-zarr/commit/04e72af0d83cdf1c4174a312b67c874fe3195f68))
- **py**: stream ozx zip creation to avoid loading entire dataset into memory ([bef151e](https://github.com/fideus-labs/ngff-zarr/commit/bef151e7968c4c290c7b58594feca141491b0be4))

### 🐛 Bug Fixes

- **py**: fix _itkwasm_chunk_bin_shrink_nd dimensionality math and transposed_dims initialization ([5a711b8](https://github.com/fideus-labs/ngff-zarr/commit/5a711b85037388b701971ff83c958968c1e3ec17))

## py-v0.33.0 (2026-03-20)

### ✨ Features

- update default OME-Zarr version from 0.4 to 0.5 ([4e60c07](https://github.com/fideus-labs/ngff-zarr/commit/4e60c077b3adcf6f4ef827a1cc112ca8e340f22c))

### 🐛 Bug Fixes

- **py**: handle both v2 and v3 array metadata in dimension separator test ([575a814](https://github.com/fideus-labs/ngff-zarr/commit/575a8141792cf1daa14e18809036f960125a7d46))
- **py**: open dimension separator test array directly by path ([4345a0a](https://github.com/fideus-labs/ngff-zarr/commit/4345a0a055af6f0ffe9a1e150bc25519ab3384b0))
- **py**: add use_consolidated=False to open_group in dimension separator test ([e1a6405](https://github.com/fideus-labs/ngff-zarr/commit/e1a64052670c7e71c50c72eda061a79a454796d2))
- **py**: use open_group instead of open_array in dimension separator test ([617bc33](https://github.com/fideus-labs/ngff-zarr/commit/617bc334d8e9a8f1bb88841a50f149098613a405))
- **py**: fall back to zarr format 2 when auto-detecting version yields empty attrs ([2a66039](https://github.com/fideus-labs/ngff-zarr/commit/2a660391dc8ffdb9cb206b9e28aaa1eb2a31b293))
- resolve merge conflicts with main branch ([af2a8eb](https://github.com/fideus-labs/ngff-zarr/commit/af2a8eb4c5e4aa995d7afd79e92a574549db3dfc))
- **py**: add zarr v3 skip marker to test_bin_shrink_map_blocks_fast_path.py ([22a70f2](https://github.com/fideus-labs/ngff-zarr/commit/22a70f2c8a7f6c7b58afaf7f43272d810e725a0c))
- **py**: fix CI failures - update test metadata access for v0.5 and add zarr v3 skip markers ([61d21e5](https://github.com/fideus-labs/ngff-zarr/commit/61d21e569beb21b040206df3c212ee3f59d0537b))
- **py**: fix test_legacy_zarr_without_type to preserve list cardinality and reliably persist mutations ([a2c1dd0](https://github.com/fideus-labs/ngff-zarr/commit/a2c1dd082f427bfcd72cd02725e1fcecda7620b0))
- **py**: improve cli support for additional protocols ([07cc062](https://github.com/fideus-labs/ngff-zarr/commit/07cc062a6d4ceb60a45373fb333a6a2eb2da4733))
- **py**: resolve relative paths to absolute in CLI for Windows compatibility ([101eb2a](https://github.com/fideus-labs/ngff-zarr/commit/101eb2a0d2d733d7ebe35b1cd15d9beb53aacc9e))

## py-v0.32.2 (2026-03-16)

### 🐛 Bug Fixes

- **py**: correctly calculate regions in to_ngff_zarr ([76772df](https://github.com/fideus-labs/ngff-zarr/commit/76772df11b7cfc860b38bc3c9ab5b872453e4481))

## py-v0.32.1 (2026-03-12)

### 🐛 Bug Fixes

- **py**: always use RGB colors for S-axis images, override OME-XML ([b34c700](https://github.com/fideus-labs/ngff-zarr/commit/b34c700606e3b08809aed3f89244109b72840ffb))

## py-v0.32.0 (2026-03-11)

### ✨ Features

- **py,ts**: add bidirectional anatomical orientation ↔ ITK direction matrix conversion ([e427ca4](https://github.com/fideus-labs/ngff-zarr/commit/e427ca4e5e131a97b9b566a5fd79ac6020110dd9))

### 🐛 Bug Fixes

- **py**: make level-selection test deterministic and CI-safe ([71e916d](https://github.com/fideus-labs/ngff-zarr/commit/71e916da84a6bbbc98ad0a8e51b118dc3d73a3ad))
- **py**: fix wrong OZX display for brightfield RGB images ([6f4f66c](https://github.com/fideus-labs/ngff-zarr/commit/6f4f66c34c67630a4db583d370513defa301b24e))

## py-v0.31.1 (2026-03-10)

### ♻️ Refactoring

- **py**: deterministic ozx content order ([597cd4f](https://github.com/fideus-labs/ngff-zarr/commit/597cd4f7be54dbb47bd8b53418f7fec5029ee2e5))

### ⚡ Performance

- **py**: add map_blocks fast path for bin-shrink downsampling ([54dd014](https://github.com/fideus-labs/ngff-zarr/commit/54dd014dbb96bbd71e58ec9c27369b18f30f4aef))
- **py**: avoid loading full file into memory for ozx ([04e72af](https://github.com/fideus-labs/ngff-zarr/commit/04e72af0d83cdf1c4174a312b67c874fe3195f68))
- **py**: stream ozx zip creation to avoid loading entire dataset into memory ([bef151e](https://github.com/fideus-labs/ngff-zarr/commit/bef151e7968c4c290c7b58594feca141491b0be4))

### 🐛 Bug Fixes

- **py**: fix _itkwasm_chunk_bin_shrink_nd dimensionality math and transposed_dims initialization ([5a711b8](https://github.com/fideus-labs/ngff-zarr/commit/5a711b85037388b701971ff83c958968c1e3ec17))

## py-v0.31.0 (2026-03-04)

### ✨ Features

- **py,ts**: relax well image path constraints per ome/ngff-spec#71 ([32b7723](https://github.com/fideus-labs/ngff-zarr/commit/32b7723a79698272dea694348f24505ff8efa6d9))
- **py**: add support for bioformats2raw container layout ([07b1e4f](https://github.com/fideus-labs/ngff-zarr/commit/07b1e4fb082256ca9a64db0392244aad3e7603f1))

### 🐛 Bug Fixes

- **ci**: avoid duplicate pixi binary install in release workflow ([07dbd36](https://github.com/fideus-labs/ngff-zarr/commit/07dbd3642a3dc44fb06f48523fcf0221d1d00607))

## py-v0.30.0 (2026-02-24)

### ✨ Features

- **py**: add HCS well path auto-detection in CLI ([c4822cf](https://github.com/fideus-labs/ngff-zarr/commit/c4822cff43979a410220f5f4ae246698d2dafc6a))
- **py**: add HCS well path support to from_ngff_zarr ([c21eaa7](https://github.com/fideus-labs/ngff-zarr/commit/c21eaa744c54925ae318c3eca46a98c77c98355c))

### 🐛 Bug Fixes

- **py**: apply PR review feedback for HCS path handling ([99061e1](https://github.com/fideus-labs/ngff-zarr/commit/99061e1f16f20a9d2550740f00f2c7d459b6f148))
- **py**: pass multithreading error in to_multiscales ([f807c6c](https://github.com/fideus-labs/ngff-zarr/commit/f807c6cc9a42138781f55ddc250e33b22fdbcc08))

## py-v0.29.0 (2026-02-20)

### ✨ Features

- **py**: improve NGFF version detection error messages ([7b03514](https://github.com/fideus-labs/ngff-zarr/commit/7b035144f117dfb8d07338121041ac63b5d5bf0b))
- **py**: integrate channel names with OMERO metadata computation ([c4d5601](https://github.com/fideus-labs/ngff-zarr/commit/c4d5601b4d8e7d88415ba163c3dd37fc776aff5d))
- **py**: add channel name extraction from OME-XML metadata ([9efb0b5](https://github.com/fideus-labs/ngff-zarr/commit/9efb0b561c053a42255778dc0d73b2510c269938))

## py-v0.28.1 (2026-02-18)

### 🐛 Bug Fixes

- **py**: apply channel reshaping to all pyramid levels in TIFF conversion ([f49155a](https://github.com/fideus-labs/ngff-zarr/commit/f49155ad7e66045c66db92aa937c07fe0b10b9fd))

## py-v0.28.0 (2026-02-18)

### ♻️ Refactoring

- **py**: consolidate test assertions in test_version_detection ([6cbacc9](https://github.com/fideus-labs/ngff-zarr/commit/6cbacc93c693a5a06844c33dd26ac443625ff399))

### 🐛 Bug Fixes

- **py**: remove unreachable HCS plate detection code ([789dfb1](https://github.com/fideus-labs/ngff-zarr/commit/789dfb127cf4bf3a2fe74d199426b869d1868a43))
- **py**: improve error message for OME-Zarr files with ome key but missing multiscales ([76b7ed7](https://github.com/fideus-labs/ngff-zarr/commit/76b7ed7e7dda159dd26b097dff8c604157a0e0ea))
- **ci**: correct new contributor detection and filter bots ([836992d](https://github.com/fideus-labs/ngff-zarr/commit/836992dd0094aaf0c8036200e7de606a628094da))
- **ci**: guard printf against names starting with dash ([7ef5d40](https://github.com/fideus-labs/ngff-zarr/commit/7ef5d406a6c6c8495e7b0652cedb8ec2656d82dc))

## py-v0.27.0 (2026-02-12)

### ✨ Features

- **py**: add dense histogram-based sampling for accurate OMERO quantiles ([8be5091](https://github.com/fideus-labs/ngff-zarr/commit/8be50912d753367b606d8d5baec826e060216152))

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
