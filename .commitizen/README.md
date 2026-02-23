# cz-ngff-zarr

Custom Commitizen plugin for the ngff-zarr monorepo.

## Features

- **Scope-based filtering**: Filters changelog commits by package scope (py, mcp, ts)
- **File-based filtering**: Analyzes file changes for commits without explicit scopes
- **GitHub commit links**: Adds clickable GitHub links with short hashes to changelog entries
- **Multi-package commits**: Handles commits affecting multiple packages (CI, docs) correctly

## Usage

This plugin is automatically installed in the `py/.pixi/envs/lint` environment
as an editable package:

```toml
[tool.pixi.feature.lint.pypi-dependencies]
cz-ngff-zarr = { path = "../.commitizen", editable = true }
```

## Filtering Logic

### Per-Package Rules

- **py**: Includes commits with scope `py` or files in `py/`
- **mcp**: Includes commits with scope `mcp` or files in `mcp/`
- **ts**: Includes commits with scope `ts` or files in `ts/`

### Ambiguous Commits

Commits are considered "ambiguous" and included in all changelogs if they:
- Have no files (e.g., merge commits)
- Modify root-level files (README, LICENSE, CONTRIBUTING, etc.)
- Modify CI/CD configuration (.github/, .gitlab-ci)
- Modify files in multiple packages

## Customization

To modify filtering logic, edit the `_should_include_for_*` methods in
`cz_ngff_zarr.py`.

## Templates

The plugin uses custom Jinja2 templates located in `templates/`:
- `py_changelog.md.j2` - Python package changelog
- `mcp_changelog.md.j2` - MCP package changelog
- `ts_changelog.md.j2` - TypeScript package changelog

All templates include GitHub commit links in the format:
```markdown
- **scope**: message ([abc1234](https://github.com/fideus-labs/ngff-zarr/commit/abc1234...))
```

## Build System

This plugin uses `hatchling` as the build backend, configured via `pyproject.toml`.
