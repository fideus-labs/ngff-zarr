"""Custom Commitizen plugin for ngff-zarr monorepo with scope filtering."""

from commitizen.cz.conventional_commits import ConventionalCommitsCz
from commitizen.git import GitCommit


class NgffZarrCz(ConventionalCommitsCz):
    """Custom Commitizen plugin for ngff-zarr monorepo.

    Filters commits based on:
    1. Conventional commit scope (py, mcp, ts, cli, hcs, etc.)
    2. Files changed in the commit
    3. Ambiguous commits (affecting all packages)

    Also adds GitHub commit links to changelog entries.
    """

    def __init__(self, config):
        super().__init__(config)
        self.package = self._detect_package(config)

    def _detect_package(self, config):
        """Detect package from tag_format or working directory."""
        tag_format = config.settings.get("tag_format", "")
        if "py-v" in tag_format:
            return "py"
        if "mcp-v" in tag_format:
            return "mcp"
        if "ts-v" in tag_format:
            return "ts"
        return None

    def changelog_message_builder_hook(
        self, parsed_message: dict, commit: GitCommit
    ) -> dict | None:
        """Filter commits based on scope and file changes.

        Returns:
            dict: Enhanced message dict with file info and commit hash
            None: Exclude commit from changelog
        """
        # Get files changed in this commit
        files = self._get_commit_files(commit.rev)
        parsed_message["files"] = files

        # Add commit hash info for GitHub links
        parsed_message["commit_hash"] = commit.rev
        parsed_message["short_hash"] = commit.rev[:7]

        # Get commit scope (can be None)
        scope = parsed_message.get("scope")

        # Apply filtering based on package
        if self._should_include_commit(scope, files):
            return parsed_message

        # Exclude this commit
        return None

    def _get_commit_files(self, commit_hash: str) -> list[str]:
        """Get list of files changed in a commit."""
        try:
            from commitizen.git import get_filenames_in_commit

            return get_filenames_in_commit(commit_hash)
        except Exception:
            # If we can't get files, return empty list
            # This will be treated as ambiguous
            return []

    def _should_include_commit(self, scope: str | None, files: list[str]) -> bool:
        """Determine if commit should be included in this package's changelog.

        Args:
            scope: Conventional commit scope (e.g., 'py', 'ts', 'mcp')
            files: List of file paths changed in commit

        Returns:
            True if commit should be included, False otherwise
        """
        # Handle package-specific scope filtering
        if self.package == "py":
            return self._should_include_for_py(scope, files)
        if self.package == "mcp":
            return self._should_include_for_mcp(scope, files)
        if self.package == "ts":
            return self._should_include_for_ts(scope, files)

        # If package unknown, include by default
        return True

    def _should_include_for_py(self, scope: str | None, files: list[str]) -> bool:
        """Filter logic for py package."""
        # Explicit py-related scopes
        if scope in ["py", "cli", "hcs"]:
            return True

        # Other scopes (not mcp/ts): check files
        if scope and scope not in ["mcp", "ts"]:
            return self._affects_package(files, "py") or self._is_ambiguous(files)

        # No scope: check files or include if ambiguous
        if not scope:
            return (
                not files
                or self._affects_package(files, "py")
                or self._is_ambiguous(files)
            )

        return False

    def _should_include_for_mcp(self, scope: str | None, files: list[str]) -> bool:
        """Filter logic for mcp package."""
        # Explicit mcp scope
        if scope == "mcp":
            return True

        # Other scopes (not py/ts): check files
        if scope and scope not in ["py", "ts", "cli", "hcs"]:
            return self._affects_package(files, "mcp") or self._is_ambiguous(files)

        # No scope: check files or include if ambiguous
        if not scope:
            return (
                not files
                or self._affects_package(files, "mcp")
                or self._is_ambiguous(files)
            )

        return False

    def _should_include_for_ts(self, scope: str | None, files: list[str]) -> bool:
        """Filter logic for ts package."""
        # Explicit ts scope
        if scope == "ts":
            return True

        # Other scopes (not py/mcp): check files
        if scope and scope not in ["py", "mcp", "cli", "hcs"]:
            return self._affects_package(files, "ts") or self._is_ambiguous(files)

        # No scope: check files or include if ambiguous
        if not scope:
            return (
                not files
                or self._affects_package(files, "ts")
                or self._is_ambiguous(files)
            )

        return False

    def _affects_package(self, files: list[str], package: str) -> bool:
        """Check if any file changes affect the given package."""
        package_prefix = f"{package}/"
        return any(f.startswith(package_prefix) for f in files)

    def _is_ambiguous(self, files: list[str]) -> bool:
        """Check if files are ambiguous (affect all packages).

        Examples of ambiguous changes:
        - Root-level documentation
        - CI/CD configuration
        - Repository-wide tools/configs
        - Changes affecting multiple packages
        """
        # If no files, consider ambiguous (merge commits, etc.)
        if not files:
            return True

        # Root-level files that affect all packages
        ambiguous_patterns = [
            ".github/",
            ".gitlab-ci",
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "AGENTS.md",
            ".pre-commit-config.yaml",
            ".gitignore",
            ".commitizen/",
            "docs/",
        ]

        # Check if any file matches ambiguous patterns
        for f in files:
            for pattern in ambiguous_patterns:
                if f.startswith(pattern) or f == pattern:
                    return True

        # If files affect multiple packages, it's ambiguous
        affects_py = any(f.startswith("py/") for f in files)
        affects_mcp = any(f.startswith("mcp/") for f in files)
        affects_ts = any(f.startswith("ts/") for f in files)

        if sum([affects_py, affects_mcp, affects_ts]) >= 2:
            return True

        return False
