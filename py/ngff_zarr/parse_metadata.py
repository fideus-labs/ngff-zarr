# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Optional

from .methods import Methods
from ._supported_versions import NgffVersion
from .v04.zarr_metadata import Omero, OmeroChannel, OmeroWindow, MethodMetadata


def _extract_method_metadata(
    metadata_dict: dict,
) -> tuple[Optional[Methods], Optional[str], Optional[MethodMetadata]]:
    method = None
    method_type = None
    method_metadata = None
    if isinstance(metadata_dict, dict):
        if "type" in metadata_dict and metadata_dict["type"] is not None:
            method_type = metadata_dict["type"]
            # Find the corresponding Methods enum
            for method_enum in Methods:
                if method_enum.value == method_type:
                    method = method_enum
                    break

        # Extract method metadata if present
        if "metadata" in metadata_dict and metadata_dict["metadata"] is not None:
            from .v04.zarr_metadata import MethodMetadata

            metadata_dict = metadata_dict["metadata"]
            if isinstance(metadata_dict, dict):
                method_metadata = MethodMetadata(
                    description=str(metadata_dict.get("description", "")),
                    method=str(metadata_dict.get("method", "")),
                    version=str(metadata_dict.get("version", "")),
                )
    return method, method_type, method_metadata


def _parse_omero(omero_data: dict) -> Optional[Omero]:
    """Parse OMERO metadata dictionary into Omero dataclass."""
    omero = None
    if isinstance(omero_data, dict) and "channels" in omero_data:
        channels_data = omero_data["channels"]
        if isinstance(channels_data, list):
            channels = []
            for channel in channels_data:
                if not isinstance(channel, dict) or "window" not in channel:
                    continue

                window_data = channel["window"]
                if not isinstance(window_data, dict):
                    continue

                # Handle backward compatibility for OMERO window metadata
                # Some stores use min/max, others use start/end, some have both
                if "start" in window_data and "end" in window_data:
                    # New format with start/end
                    start = float(window_data["start"])  # type: ignore
                    end = float(window_data["end"])  # type: ignore
                    if "min" in window_data and "max" in window_data:
                        # Both formats present
                        min_val = float(window_data["min"])  # type: ignore
                        max_val = float(window_data["max"])  # type: ignore
                    else:
                        # Only start/end, use them as min/max
                        min_val = start
                        max_val = end
                elif "min" in window_data and "max" in window_data:
                    # Old format with min/max only
                    min_val = float(window_data["min"])  # type: ignore
                    max_val = float(window_data["max"])  # type: ignore
                    # Use min/max as start/end for backward compatibility
                    start = min_val
                    end = max_val
                else:
                    # Invalid window data, skip this channel
                    continue

                channels.append(
                    OmeroChannel(
                        color=str(channel["color"]),  # type: ignore
                        label=str(channel.get("label", None))
                        if channel.get("label") is not None
                        else None,  # type: ignore
                        window=OmeroWindow(
                            min=min_val,
                            max=max_val,
                            start=start,
                            end=end,
                        ),
                    )
                )

            if channels:
                omero = Omero(channels=channels)

    return omero


def _parse_hcs_path(store_path: str) -> tuple[str, Optional[str]]:
    """Parse a potential HCS path into store and sub-path components.
    
    Parameters
    ----------
    store_path : str
        Path that may contain a zarr store with an optional sub-path
        (e.g., 'plate.zarr/A/1/0')
    
    Returns
    -------
    store : str
        The path to the zarr store
    subpath : str or None
        The sub-path within the store, or None if no sub-path
        
    Examples
    --------
    >>> _parse_hcs_path('plate.zarr')
    ('plate.zarr', None)
    >>> _parse_hcs_path('plate.zarr/A/1/0')
    ('plate.zarr', 'A/1/0')
    >>> _parse_hcs_path('/path/to/plate.ome.zarr/B/2')
    ('/path/to/plate.ome.zarr', 'B/2')
    """
    from pathlib import Path
    
    path = Path(store_path)
    parts = path.parts
    
    # Find the zarr store in the path
    # Check longer extensions first to avoid matching '.zarr' in '.ome.zarr'
    for i, part in enumerate(parts):
        if part.endswith('.ome.zarr') or part.endswith('.ozx') or part.endswith('.zarr'):
            # Found the store, check if there's a sub-path
            store = str(Path(*parts[:i+1]))
            if i < len(parts) - 1:
                # There's a sub-path after the store
                subpath = '/'.join(parts[i+1:])
                return store, subpath
            else:
                # No sub-path, just the store
                return store, None
    
    # No zarr extension found, return as-is
    return store_path, None


def _is_hcs_plate(root_attrs: dict) -> bool:
    """Check if root attributes indicate an HCS plate structure.

    Returns True if this is an HCS plate (not a regular image group).
    """
    # v0.5+ plate: has "ome" key with "plate" subkey but no "multiscales"
    if (
        "ome" in root_attrs
        and isinstance(root_attrs["ome"], dict)
        and "plate" in root_attrs["ome"]
        and "multiscales" not in root_attrs["ome"]
    ):
        return True

    # v0.4 plate: has "plate" key at root but no "multiscales"
    if (
        "plate" in root_attrs
        and isinstance(root_attrs["plate"], dict)
        and "multiscales" not in root_attrs
    ):
        return True

    return False


def _detect_version(root_attrs: dict) -> NgffVersion:
    """Detect NGFF version from root attributes.

    Handles both regular image groups and HCS plate structures.
    """
    version_str: Optional[str] = None
    if "ome" in root_attrs:
        # v0.5+ format has metadata under "ome" key
        version_str = root_attrs["ome"].get("version")
        # If version is not specified in ome dict, default to 0.5
        # since the presence of "ome" key indicates v0.5+ format
        if version_str is None:
            version_str = "0.5"
    else:
        # v0.4 format has "multiscales" at root level
        multiscales = root_attrs.get("multiscales", [])
        if multiscales and isinstance(multiscales, list):
            version_str = multiscales[0].get("version", "0.4")
        # Handle HCS plate structures that don't have multiscales at root
        # but have "plate" metadata instead (and no "multiscales" key)
        elif (
            "plate" in root_attrs
            and isinstance(root_attrs["plate"], dict)
            and "multiscales" not in root_attrs
        ):
            # HCS plates in v0.4 format have "plate" key at root level
            # Default to 0.4 since this is the v0.4 plate structure
            version_str = "0.4"

    if version_str is None:
        raise ValueError("Could not detect NGFF version from root attributes.")

    return NgffVersion(version_str)
