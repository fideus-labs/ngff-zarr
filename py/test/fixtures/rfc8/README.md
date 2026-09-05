<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->

# RFC-8 collection example fixtures

Copies of the example documents in
<https://github.com/normanrz/ngff-rfc8-collection-examples> at commit
`54250e21f79cbe5d68f20b07dcb00079ef781b4a`, with two normalizations: every
`ome.version` string is set to `0.9.dev3` (the sources carry the draft tags
`0.8` and `1.0dev891342`), which is the version ngff-zarr gates the RFC-8
node model on, and the `node_type` of `webknossos_inline_multiscale.json`
is set to `"group"` (the source carries `"zarr"`, which Zarr v3 does not
define). Everything else is byte-identical to upstream, including the
extension attributes (`ngio:source`, `webknossos:*`, `mobie:*`) and the
`mobie_image_labels_table.json` extension path type (`google-sheet`, an
unprefixed extension identifier that fails `path-type-known` on purpose).

`webknossos_inline_multiscale.json` is a full `zarr.json` document (the
Zarr-group storage medium); the others are standalone JSON documents with a
root `ome` key. The TypeScript test suite reads these files from the Python
tree, like the RFC-5 transform cases.
