# GitHub Release Automation

This document describes the automated release workflow for the ngff-zarr monorepo.

## Overview

The release workflow automatically creates GitHub Releases with changelog content and build artifacts when version tags are pushed for any of the three sub-projects:

- **Python Package** (`py/`): Tags like `py-v0.22.0`
- **TypeScript Package** (`ts/`): Tags like `ts-v0.4.0`
- **MCP Package** (`mcp/`): Tags like `mcp-v0.5.0`

## How It Works

### 1. Tag-Based Triggering

The workflow triggers automatically when you push a tag matching the pattern:
- `py-v*` for Python packages
- `ts-v*` for TypeScript packages
- `mcp-v*` for MCP packages

### 2. Automated Steps

For each tag push, the workflow:

1. **Identifies the project** from the tag prefix
2. **Extracts the version** from the tag
3. **Parses the changelog** to get version-specific release notes
4. **Builds artifacts** for the specific project:
   - **Python**: Wheel (`.whl`) and source distribution (`.tar.gz`)
   - **TypeScript**: NPM tarball (`.tgz`)
   - **MCP**: Wheel (`.whl`) and source distribution (`.tar.gz`)
5. **Creates a GitHub Release** with:
   - Release title (e.g., "Python Package v0.22.0")
   - Changelog content for that version
   - Build artifacts attached
   - Pre-release flag (if applicable)

### 3. Pre-release Detection

Tags containing any of these keywords are automatically marked as pre-releases:
- `alpha`
- `beta`
- `rc` (release candidate)
- `dev`
- `pre`

Examples:
- `py-v0.22.0-beta.1` → Pre-release
- `ts-v0.4.0-rc.2` → Pre-release
- `mcp-v0.5.0` → Stable release

## Usage

### Creating a Release

#### Option 1: Using Commitizen (Recommended)

Each sub-project has Commitizen configured. Use the bump command to automatically:
- Increment version
- Update changelog
- Create a git tag
- Commit changes

```bash
# Python package
cd py
pixi run -e lint bump

# TypeScript package
cd ts
# Use commitizen from py's pixi environment
../py/.pixi/envs/lint/bin/cz bump

# MCP package
cd mcp
pixi run -e dev bump
```

After Commitizen creates the tag, push it:

```bash
git push origin <tag-name>
```

#### Option 2: Manual Tag Creation

If you prefer to create tags manually:

```bash
# Create an annotated tag
git tag -a py-v0.22.0 -m "Release Python package v0.22.0"

# Push the tag
git push origin py-v0.22.0
```

**Important**: Make sure the version exists in the corresponding `CHANGELOG.md` file before pushing the tag.

### Monitoring the Release

1. After pushing a tag, go to the **Actions** tab in GitHub
2. Find the **Release** workflow run
3. Monitor the build and release creation process
4. Once complete, check the **Releases** page

### Release Artifacts

Each project type includes specific artifacts:

#### Python Package (`py-v*`)
- `ngff-zarr-{version}-py3-none-any.whl` - Python wheel
- `ngff-zarr-{version}.tar.gz` - Source distribution

#### TypeScript Package (`ts-v*`)
- `fideus-labs-ngff-zarr-{version}.tgz` - NPM package tarball

#### MCP Package (`mcp-v*`)
- `ngff_zarr_mcp-{version}-py3-none-any.whl` - Python wheel
- `ngff-zarr-mcp-{version}.tar.gz` - Source distribution

## Publishing to Package Registries

The workflow includes **commented-out** sections for publishing to package registries (PyPI, NPM, JSR). These are disabled by default for safety.

### Enabling Publishing

#### For Python Packages (PyPI)

1. **Create a PyPI API token**:
   - Go to https://pypi.org/manage/account/token/
   - Create a token with appropriate scope

2. **Add the token to GitHub Secrets**:
   - Go to repository Settings → Secrets and variables → Actions
   - Add `PYPI_API_TOKEN` (for py package)
   - Optionally add `PYPI_MCP_TOKEN` (for mcp package, or reuse the same token)

3. **Uncomment the publishing step** in `.github/workflows/release.yml`:
   ```yaml
   # Find these sections and remove the comment markers (#)
   - name: Publish Python package to PyPI
   - name: Publish MCP package to PyPI
   ```

#### For TypeScript Package (NPM)

1. **Create an NPM access token**:
   - Go to https://www.npmjs.com/settings/{username}/tokens
   - Create an "Automation" token

2. **Add the token to GitHub Secrets**:
   - Add `NPM_TOKEN`

3. **Uncomment the publishing step** in `.github/workflows/release.yml`:
   ```yaml
   - name: Publish to NPM
   ```

#### For TypeScript Package (JSR - Optional)

1. **Create a Deno Deploy token** (if using JSR)
2. **Add `DENO_DEPLOY_TOKEN` to GitHub Secrets**
3. **Uncomment the JSR publishing step**

### Testing Publishing

Before enabling automatic publishing:

1. **Test the release creation** with a few tags
2. **Verify the artifacts** are correctly built
3. **Download and test artifacts** manually
4. **Manually publish** one release to confirm everything works
5. **Then enable** automatic publishing

## Changelog Format

The workflow expects changelogs to follow the [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [VERSION] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing features

### Fixed
- Bug fixes

### Removed
- Removed features
```

Or with tag prefixes (higher priority):

```markdown
## tag-vVERSION (YYYY-MM-DD)

### Features
- New features

### Bug Fixes
- Bug fixes
```

**Note**: The workflow prioritizes the tag format (e.g., `## py-v0.22.0`) over the bracket format (e.g., `## [0.22.0]`) when both exist.

## Troubleshooting

### Tag doesn't trigger workflow

**Possible causes**:
- Tag format doesn't match `{py|ts|mcp}-v*` pattern
- Tag was created but not pushed to GitHub
- Workflow file has syntax errors

**Solution**:
```bash
# Verify tag format
git tag -l

# Push tag explicitly
git push origin <tag-name>

# Check workflow syntax
# Use GitHub Actions tab or a YAML validator
```

### Changelog extraction fails

**Possible causes**:
- Version not found in changelog
- Changelog format doesn't match expected pattern
- Wrong tag prefix used

**Solution**:
```bash
# Test changelog extraction locally
.github/scripts/extract-changelog.sh py/CHANGELOG.md 0.22.0 py

# Ensure version exists in changelog
grep "0.22.0" py/CHANGELOG.md
```

### Build artifacts not found

**Possible causes**:
- Build process failed
- Artifacts in unexpected location
- Build dependencies missing

**Solution**:
- Check the workflow logs in GitHub Actions
- Verify build commands work locally
- Check artifact paths in release.yml

### Release created but artifacts missing

**Possible causes**:
- File glob pattern doesn't match artifacts
- Artifacts were not created successfully

**Solution**:
- Check the "Verify artifacts" step in workflow logs
- Ensure build step completed successfully
- Verify artifact paths in release.yml match actual output

## Workflow Files

- **`.github/workflows/release.yml`** - Main workflow file
- **`.github/scripts/extract-changelog.sh`** - Changelog extraction script
- **`.github/RELEASE.md`** - This documentation file

## Examples

### Creating a Python package release

```bash
# 1. Update version and changelog using commitizen
cd py
pixi run -e lint bump

# 2. Commitizen will create a commit and tag, push them
git push origin main
git push origin py-v0.23.0

# 3. Wait for GitHub Actions to complete
# 4. Check the Releases page for your new release
```

### Creating a pre-release

```bash
# Create a beta tag manually
git tag -a ts-v0.5.0-beta.1 -m "Beta release"
git push origin ts-v0.5.0-beta.1

# The workflow will automatically mark it as a pre-release
```

### Testing the workflow

```bash
# Create a test tag on a test branch
git checkout -b test-release
git tag -a py-v0.22.1-test -m "Test release"
git push origin py-v0.22.1-test

# Check the Actions tab
# If successful, delete the test release and tag
# If failed, check the logs and fix issues
```

## Best Practices

1. **Always test** with a test tag first before creating official releases
2. **Keep changelogs updated** before creating version tags
3. **Use semantic versioning** for version numbers
4. **Review build artifacts** in the first few releases before enabling auto-publishing
5. **Monitor Actions logs** for any warnings or issues
6. **Create annotated tags** with meaningful messages
7. **Test publishing manually** before enabling automated publishing

## Security Considerations

- **API tokens** should have minimal required permissions
- **Never commit tokens** to the repository
- **Use GitHub Secrets** for all sensitive values
- **Enable branch protection** to prevent accidental tag deletion
- **Review workflow changes** carefully before merging
- **Test on a fork** if making significant workflow changes

## Support

If you encounter issues:

1. Check the workflow logs in GitHub Actions
2. Test changelog extraction locally
3. Verify tag format matches expectations
4. Review this documentation
5. Check GitHub Actions documentation: https://docs.github.com/en/actions

## Version History

- **2026-02-03**: Initial release automation implementation
  - Support for py, ts, and mcp packages
  - Automated changelog extraction
  - Build artifact creation
  - Pre-release detection
  - Publishing infrastructure (commented out)
