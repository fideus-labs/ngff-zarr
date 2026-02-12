#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
#
# Extract changelog content for a specific version from a CHANGELOG.md file
#
# Usage: extract-changelog.sh <changelog_file> <version> <tag_prefix>
#
# Arguments:
#   changelog_file: Path to CHANGELOG.md
#   version: Version number (e.g., 0.22.0)
#   tag_prefix: Tag prefix (e.g., py, ts, mcp)
#
# Example: extract-changelog.sh py/CHANGELOG.md 0.22.0 py

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <changelog_file> <version> <tag_prefix>"
    exit 1
fi

CHANGELOG_FILE="$1"
VERSION="$2"
TAG_PREFIX="$3"

if [ ! -f "$CHANGELOG_FILE" ]; then
    echo "Error: Changelog file not found: $CHANGELOG_FILE"
    exit 1
fi

# Function to extract changelog content between two version headers
# Priority order:
# 1. ## tag-vVERSION (e.g., ## py-v0.22.0)
# 2. ## [VERSION] - DATE (e.g., ## [0.22.0] - 2026-02-02)
extract_changelog() {
    local file="$1"
    local version="$2"
    local prefix="$3"

    # Try tag-vVERSION format first (higher priority)
    local tag_pattern="^## ${prefix}-v${version}"
    local bracket_pattern="^## \[${version}\]"

    # Check which pattern exists in the file
    local has_tag_format
    has_tag_format=$(grep -c "$tag_pattern" "$file" || true)
    local has_bracket_format
    has_bracket_format=$(grep -c "$bracket_pattern" "$file" || true)

    if [ "$has_tag_format" -gt 0 ]; then
        # Use tag-vVERSION format (higher priority)
        pattern="$tag_pattern"
    elif [ "$has_bracket_format" -gt 0 ]; then
        # Use [VERSION] format
        pattern="$bracket_pattern"
    else
        echo "Error: Version $version not found in $file"
        echo "Looked for patterns:"
        echo "  1. ## ${prefix}-v${version}"
        echo "  2. ## [${version}]"
        exit 1
    fi

    # Extract content from matched header until next ## header (or end of file)
    # This preserves all markdown formatting, subsections, dates, etc.
    awk -v pattern="$pattern" '
        BEGIN { found=0; print_content=0 }

        # When we find our version header
        $0 ~ pattern {
            found=1
            print_content=1
            print $0  # Print the header itself
            next
        }

        # When we hit the next version header, stop
        print_content && /^## / {
            print_content=0
        }

        # Print lines between our version and the next version
        print_content {
            print $0
        }

        END {
            if (!found) {
                exit 1
            }
        }
    ' "$file"
}

get_release_tags() {
    local prefix="$1"
    local version="$2"
    local current_tag="${prefix}-v${version}"
    local previous_tag=""

    mapfile -t tags < <(git tag --list "${prefix}-v*" --sort=-version:refname)

    if [ "${#tags[@]}" -eq 0 ]; then
        echo "${current_tag}|${previous_tag}"
        return
    fi

    for ((i=0; i<${#tags[@]}; i++)); do
        if [ "${tags[$i]}" = "${current_tag}" ]; then
            if [ $((i+1)) -lt ${#tags[@]} ]; then
                previous_tag="${tags[$((i+1))]}"
            fi
            break
        fi
    done

    echo "${current_tag}|${previous_tag}"
}

append_contributors() {
    local prefix="$1"
    local version="$2"
    local tag_info
    local current_tag
    local previous_tag
    local range

    tag_info=$(get_release_tags "$prefix" "$version")
    current_tag="${tag_info%%|*}"
    previous_tag="${tag_info##*|}"

    if [ -n "$previous_tag" ]; then
        range="${previous_tag}..${current_tag}"
    else
        range="${current_tag}"
    fi

    mapfile -t contributor_lines < <(git shortlog -s -n "$range" | sed 's/^[[:space:]]*[0-9]\+[[:space:]]\+//')

    if [ "${#contributor_lines[@]}" -eq 0 ]; then
        return
    fi

    printf "\n## 🤝 Contributors\n\n"
    printf "We appreciate the contributions from:\n\n"
    for name in "${contributor_lines[@]}"; do
        if [ -n "$name" ]; then
            printf "- %s\n" "$name"
        fi
    done

    local new_contributors=()

    if [ -z "$previous_tag" ]; then
        for name in "${contributor_lines[@]}"; do
            if [ -n "$name" ]; then
                new_contributors+=("$name")
            fi
        done
    else
        for name in "${contributor_lines[@]}"; do
            if [ -n "$name" ]; then
                if ! git log --format="%an" "$previous_tag" | grep -Fxq "$name"; then
                    new_contributors+=("$name")
                fi
            fi
        done
    fi

    if [ "${#new_contributors[@]}" -gt 0 ]; then
        printf "\n## 🌟 Congratulations\n\n"
        printf "Congratulations to our new contributors:\n\n"
        for name in "${new_contributors[@]}"; do
            printf "- %s\n" "$name"
        done
    fi
}

# Extract and output the changelog
extract_changelog "$CHANGELOG_FILE" "$VERSION" "$TAG_PREFIX"
append_contributors "$TAG_PREFIX" "$VERSION"
