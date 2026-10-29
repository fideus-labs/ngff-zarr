<!-- SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC -->
<!-- SPDX-License-Identifier: MIT -->
# ngff-zarr

[![PyPI - Version](https://img.shields.io/pypi/v/ngff-zarr.svg)](https://pypi.org/project/ngff-zarr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ngff-zarr.svg)](https://pypi.org/project/ngff-zarr)
[![Python CI Testing](https://github.com/fideus-labs/ngff-zarr/actions/workflows/python.yml/badge.svg)](https://github.com/fideus-labs/ngff-zarr/actions/workflows/python.yml)
[![npm - Version](https://img.shields.io/npm/v/@fideus-labs/ngff-zarr.svg)](https://www.npmjs.com/package/@fideus-labs/ngff-zarr)
[![Typescript CI Testing](https://github.com/fideus-labs/ngff-zarr/actions/workflows/typescript.yml/badge.svg)](https://github.com/fideus-labs/ngff-zarr/actions/workflows/typescript.yml)
[![MCP CI Testing](https://github.com/fideus-labs/ngff-zarr/actions/workflows/mcp-ci.yml/badge.svg)](https://github.com/fideus-labs/ngff-zarr/actions/workflows/mcp-ci.yml)
[![DOI](https://zenodo.org/badge/541840158.svg)](https://zenodo.org/badge/latestdoi/541840158)
[![Documentation Status](https://readthedocs.org/projects/ngff-zarr/badge/?version=latest)](https://ngff-zarr.readthedocs.io/en/latest/?badge=latest)

---

A multi-language implementation of the
[Open Microscopy Environment (OME) Next Generation File Format (NGFF) Zarr](https://ngff.openmicroscopy.org)
specification.

## Repository Structure

This repository contains multiple packages implementing NGFF-Zarr support:

- **[`py/`](./py/)** - Python package (ngff-zarr) - A lean and kind NGFF-Zarr
  implementation
- **[`ts/`](./ts/)** - TypeScript/JavaScript package (@fideus-labs/ngff-zarr) -
  Universal OME-Zarr for Node.js and browsers
- **[`mcp/`](./mcp/)** - Model Context Protocol (MCP) server (ngff-zarr-mcp) for
  AI integration

## Python Package (`py/`)

The main Python package provides:

✨ **Features**

- Minimal dependencies
- Work with arbitrary Zarr store types
- Lazy, parallel, and web ready -- no local filesystem required
- Process extremely large datasets
- Conversion of most bioimaging file formats
- Multiple downscaling methods
- Supports Python>=3.11
- Reads OME-Zarr v0.1 to v0.6 into simple Python data classes with Dask arrays
- Optional OME-Zarr data model validation during reading
- Writes OME-Zarr v0.4 to v0.6
- v0.6 adds RFC-5 coordinate systems and transformations
- Reads and writes the opt-in development version `0.9.dev1`, which adopts RFC-3
  (expanded axis count, names, types and order)
- [Sharded Zarr] stores
- Zarr I/O backed by [zarrista] / Rust [zarrs] (Python) or zarrita
  (TypeScript) -- no zarr-python dependency; OME-Zarr 0.4 (Zarr v2) outputs
  stay readable by zarr-python 2 and 3, and 0.5+ (Zarr v3) by zarr-python 3
- [Anatomical orientation metadata](./docs/rfc4.md) (RFC-4)
- **OME-Zarr Zip (.ozx) file support** for single-file OME-Zarr datasets (RFC-9)
- **High Content Screening (HCS) support** for plate and well data
- **Model Context Protocol (MCP) server** for AI agent integration

📖 **Documentation**

More information about command line usage, the Python API, and library features
can be found in [our documentation](https://ngff-zarr.readthedocs.io/).

## TypeScript/JavaScript Package (`ts/`)

The TypeScript package provides universal OME-Zarr support for modern JavaScript environments:

✨ **Features**

- 📘 **TypeScript-first** with full type definitions
- 📦 **Universal compatibility** - Works in Node.js and browsers
- 🔍 **Type-safe** with Zod schema validation
- 🗂️ **OME-Zarr v0.4, v0.5, and v0.6** support: reads all three and writes v0.5 and v0.6 (v0.6 adds RFC-5 coordinate systems and transformations), plus the opt-in development version `0.9.dev1` which adopts RFC-3 (expanded axis count, names, types and order)
- 🌐 **Web ready** - No filesystem dependencies, works with remote stores
- 🏗️ **Mirrors Python API** - Familiar interfaces for Python users
- 📚 **Lazy loading** - Efficient handling of large datasets

📦 **Installation**

```bash
# Node.js / npm
npm install @fideus-labs/ngff-zarr

# Browser (CDN)
import * as ngffZarr from "https://esm.sh/@fideus-labs/ngff-zarr";
```

📖 **Documentation**

See the [TypeScript documentation](https://ngff-zarr.readthedocs.io/en/latest/typescript.html)
for detailed usage, API reference, and examples.

## MCP Server (`mcp/`)

The Model Context Protocol server enables AI assistants like Claude to interact
with NGFF-Zarr files. See the [MCP documentation](./mcp/README.md) for setup and
usage instructions.

## Contributing

Contributions are welcome and appreciated! 🎉 The [development guide] covers
environment setup, the pull request workflow, commit message conventions, and
the development commands for each package. Please also read our
[Code of Conduct](./CODE_OF_CONDUCT.md).

Thanks to everyone who has contributed code, documentation, bug reports, and
ideas ([emoji key](https://allcontributors.org/en/reference/emoji-key/)). To add
someone, comment `@all-contributors please add @<username> for <contributions>`
on an issue or pull request.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://fideus.io"><img src="https://avatars.githubusercontent.com/u/25432?v=4?s=100" width="100px;" alt="Matt McCormick"/><br /><sub><b>Matt McCormick</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=thewtex" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=thewtex" title="Documentation">📖</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=thewtex" title="Tests">⚠️</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=thewtex" title="Maintenance">🚧</a> <a href="https://github.com/fideus-labs/ngff-zarr/pulls?q=is%3Apr+reviewed-by%3Athewtex" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=thewtex" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Athewtex" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Athewtex" title="Project Management">📆</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vboussot"><img src="https://avatars.githubusercontent.com/u/17681131?v=4?s=100" width="100px;" alt="Valentin Boussot"/><br /><sub><b>Valentin Boussot</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=vboussot" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=vboussot" title="Documentation">📖</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=vboussot" title="Tests">⚠️</a> <a href="https://github.com/fideus-labs/ngff-zarr/pulls?q=is%3Apr+reviewed-by%3Avboussot" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://gerbi-gmb.de"><img src="https://avatars.githubusercontent.com/u/38459088?v=4?s=100" width="100px;" alt="Johannes Soltwedel"/><br /><sub><b>Johannes Soltwedel</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=jo-mueller" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=jo-mueller" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://bioimaging.fi"><img src="https://avatars.githubusercontent.com/u/11444749?v=4?s=100" width="100px;" alt="Junel Solis"/><br /><sub><b>Junel Solis</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=junelsolis" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=junelsolis" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/folterj"><img src="https://avatars.githubusercontent.com/u/7360564?v=4?s=100" width="100px;" alt="Joost de Folter"/><br /><sub><b>Joost de Folter</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=folterj" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=folterj" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.embl.org/people/person/michiel-vierdag/"><img src="https://avatars.githubusercontent.com/u/22346363?v=4?s=100" width="100px;" alt="Wouter-Michiel Vierdag"/><br /><sub><b>Wouter-Michiel Vierdag</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=melonora" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://craig.im/"><img src="https://avatars.githubusercontent.com/u/9630402?v=4?s=100" width="100px;" alt="Craig Myles"/><br /><sub><b>Craig Myles</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=CraigMyles" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=CraigMyles" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://ianhuntisaak.com"><img src="https://avatars.githubusercontent.com/u/10111092?v=4?s=100" width="100px;" alt="Ian Hunt-Isaak"/><br /><sub><b>Ian Hunt-Isaak</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=ianhi" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=ianhi" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/odinsbane"><img src="https://avatars.githubusercontent.com/u/3462801?v=4?s=100" width="100px;" alt="odinsbane"/><br /><sub><b>odinsbane</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=odinsbane" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=odinsbane" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/btbest"><img src="https://avatars.githubusercontent.com/u/63287233?v=4?s=100" width="100px;" alt="Benedikt Best"/><br /><sub><b>Benedikt Best</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=btbest" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=btbest" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ClementCaporal"><img src="https://avatars.githubusercontent.com/u/94049435?v=4?s=100" width="100px;" alt="Clement Caporal"/><br /><sub><b>Clement Caporal</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=ClementCaporal" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=ClementCaporal" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GFleishman"><img src="https://avatars.githubusercontent.com/u/8507206?v=4?s=100" width="100px;" alt="Greg M. Fleishman"/><br /><sub><b>Greg M. Fleishman</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=GFleishman" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DiloCorp"><img src="https://avatars.githubusercontent.com/u/199592093?v=4?s=100" width="100px;" alt="Guillaume Brunet"/><br /><sub><b>Guillaume Brunet</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=DiloCorp" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tbirdso"><img src="https://avatars.githubusercontent.com/u/40648863?v=4?s=100" width="100px;" alt="Tom Birdsong"/><br /><sub><b>Tom Birdsong</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=tbirdso" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.davidstansby.com"><img src="https://avatars.githubusercontent.com/u/6197628?v=4?s=100" width="100px;" alt="David Stansby"/><br /><sub><b>David Stansby</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=dstansby" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://markk.co"><img src="https://avatars.githubusercontent.com/u/7525285?v=4?s=100" width="100px;" alt="Mark Keller"/><br /><sub><b>Mark Keller</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=keller-mark" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/VKarlaftis"><img src="https://avatars.githubusercontent.com/u/215151228?v=4?s=100" width="100px;" alt="Vasilis M. Karlaftis"/><br /><sub><b>Vasilis M. Karlaftis</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=VKarlaftis" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://scigantic.com"><img src="https://avatars.githubusercontent.com/u/36093535?v=4?s=100" width="100px;" alt="aaronkanzer"/><br /><sub><b>aaronkanzer</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=aaronkanzer" title="Code">💻</a> <a href="https://github.com/fideus-labs/ngff-zarr/commits?author=aaronkanzer" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rtmalikian"><img src="https://avatars.githubusercontent.com/u/230301546?v=4?s=100" width="100px;" alt="Raphael Malikian"/><br /><sub><b>Raphael Malikian</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/commits?author=rtmalikian" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/scottcali"><img src="https://avatars.githubusercontent.com/u/49628600?v=4?s=100" width="100px;" alt="scottcali"/><br /><sub><b>scottcali</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Ascottcali" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://will-moore.github.io/"><img src="https://avatars.githubusercontent.com/u/900055?v=4?s=100" width="100px;" alt="Will Moore"/><br /><sub><b>Will Moore</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Awill-moore" title="Bug reports">🐛</a> <a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Awill-moore" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ahnsws"><img src="https://avatars.githubusercontent.com/u/17461258?v=4?s=100" width="100px;" alt="Sebastian Ahn"/><br /><sub><b>Sebastian Ahn</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Aahnsws" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/akhanf"><img src="https://avatars.githubusercontent.com/u/11492701?v=4?s=100" width="100px;" alt="Ali Khan"/><br /><sub><b>Ali Khan</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Aakhanf" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://arthurmasson.xyz"><img src="https://avatars.githubusercontent.com/u/1814799?v=4?s=100" width="100px;" alt="Arthur Masson"/><br /><sub><b>Arthur Masson</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Aarthursw" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.kitware.com"><img src="https://avatars.githubusercontent.com/u/51238406?v=4?s=100" width="100px;" alt="Brianna Major"/><br /><sub><b>Brianna Major</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Abnmajor" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BrianWhitneyAI"><img src="https://avatars.githubusercontent.com/u/94479316?v=4?s=100" width="100px;" alt="Brian Whitney"/><br /><sub><b>Brian Whitney</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3ABrianWhitneyAI" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/d-v-b"><img src="https://avatars.githubusercontent.com/u/3805136?v=4?s=100" width="100px;" alt="Davis Bennett"/><br /><sub><b>Davis Bennett</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Ad-v-b" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fredricj"><img src="https://avatars.githubusercontent.com/u/573445?v=4?s=100" width="100px;" alt="Fredric Johansson"/><br /><sub><b>Fredric Johansson</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Afredricj" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/IgorTatarnikov"><img src="https://avatars.githubusercontent.com/u/61896994?v=4?s=100" width="100px;" alt="Igor Tatarnikov"/><br /><sub><b>Igor Tatarnikov</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3AIgorTatarnikov" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ivirshup"><img src="https://avatars.githubusercontent.com/u/8238804?v=4?s=100" width="100px;" alt="Isaac Virshup"/><br /><sub><b>Isaac Virshup</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Aivirshup" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://ilovesymposia.com"><img src="https://avatars.githubusercontent.com/u/492549?v=4?s=100" width="100px;" alt="Juan Nunez-Iglesias"/><br /><sub><b>Juan Nunez-Iglesias</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Ajni" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://kevinyamauchi.github.io/"><img src="https://avatars.githubusercontent.com/u/1120672?v=4?s=100" width="100px;" alt="Kevin Yamauchi"/><br /><sub><b>Kevin Yamauchi</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Akevinyamauchi" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lorenzocerrone"><img src="https://avatars.githubusercontent.com/u/23013839?v=4?s=100" width="100px;" alt="Lorenzo"/><br /><sub><b>Lorenzo</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Alorenzocerrone" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LucaMarconato"><img src="https://avatars.githubusercontent.com/u/2664412?v=4?s=100" width="100px;" alt="LucaMarconato"/><br /><sub><b>LucaMarconato</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3ALucaMarconato" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/m-albert"><img src="https://avatars.githubusercontent.com/u/12528388?v=4?s=100" width="100px;" alt="Marvin Albert"/><br /><sub><b>Marvin Albert</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Am-albert" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/maoyuanpeng"><img src="https://avatars.githubusercontent.com/u/54882489?v=4?s=100" width="100px;" alt="maoyuanpeng"/><br /><sub><b>maoyuanpeng</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Amaoyuanpeng" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MrMupfel"><img src="https://avatars.githubusercontent.com/u/121063302?v=4?s=100" width="100px;" alt="Flo"/><br /><sub><b>Flo</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3AMrMupfel" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mtcaton"><img src="https://avatars.githubusercontent.com/u/62513969?v=4?s=100" width="100px;" alt="Mikala Caton"/><br /><sub><b>Mikala Caton</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Amtcaton" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://normanrz.com/"><img src="https://avatars.githubusercontent.com/u/335438?v=4?s=100" width="100px;" alt="Norman Rzepka"/><br /><sub><b>Norman Rzepka</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Anormanrz" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ralfox"><img src="https://avatars.githubusercontent.com/u/200069334?v=4?s=100" width="100px;" alt="ralfox"/><br /><sub><b>ralfox</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Aralfox" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://satra.cogitatum.org"><img src="https://avatars.githubusercontent.com/u/184063?v=4?s=100" width="100px;" alt="Satrajit Ghosh"/><br /><sub><b>Satrajit Ghosh</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Asatra" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.linkedin.com/in/sebisabs"><img src="https://avatars.githubusercontent.com/u/3833249?v=4?s=100" width="100px;" alt="Sebastian Rhode"/><br /><sub><b>Sebastian Rhode</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Asebi06" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://srivarra.dev"><img src="https://avatars.githubusercontent.com/u/8909315?v=4?s=100" width="100px;" alt="Sri Varra"/><br /><sub><b>Sri Varra</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Asrivarra" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/swelborn"><img src="https://avatars.githubusercontent.com/u/55868530?v=4?s=100" width="100px;" alt="Sam Welborn"/><br /><sub><b>Sam Welborn</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Aswelborn" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://napari.org"><img src="https://avatars.githubusercontent.com/u/47310455?v=4?s=100" width="100px;" alt="Tim Monko"/><br /><sub><b>Tim Monko</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3ATimMonko" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tischi"><img src="https://avatars.githubusercontent.com/u/2157566?v=4?s=100" width="100px;" alt="Christian Tischer"/><br /><sub><b>Christian Tischer</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3Atischi" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Tomaz-Vieira"><img src="https://avatars.githubusercontent.com/u/4031460?v=4?s=100" width="100px;" alt="Tomaz-Vieira"/><br /><sub><b>Tomaz-Vieira</b></sub></a><br /><a href="https://github.com/fideus-labs/ngff-zarr/issues?q=author%3ATomaz-Vieira" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## See also

- [ome-zarr-py](https://github.com/ome/ome-zarr-py)
- [multiscale-spatial-image](https://github.com/spatial-image/multiscale-spatial-image)
- [itk-ioomezarrngff](https://github.com/InsightSoftwareConsortium/ITKIOOMEZarrNGFF)
- [iohub](https://czbiohub-sf.github.io/iohub/)
- [pydantic-ome-ngff](https://janeliascicomp.github.io/pydantic-ome-ngff/)
- [aicsimageio](https://allencellmodeling.github.io/aicsimageio/)
- [bfio](https://bfio.readthedocs.io/)
- [zod-ome-ngff](https://github.com/hms-dbmi/zod-ome-ngff)

## License

`ngff-zarr` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.

[Sharded Zarr]: https://zarr.dev/zeps/accepted/ZEP0002.html
[development guide]: https://ngff-zarr.readthedocs.io/en/latest/development.html
[zarrista]: https://github.com/developmentseed/zarrista
[zarrs]: https://zarrs.dev/
