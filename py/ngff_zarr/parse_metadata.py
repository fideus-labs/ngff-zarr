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


def _detect_version(root_attrs: dict) -> NgffVersion:
    """Detect NGFF version from root attributes."""
    version_str: Optional[str] = None
    if "ome" in root_attrs:
        version_str = root_attrs["ome"].get("version")
    else:
        multiscales = root_attrs.get("multiscales", [])
        if multiscales and isinstance(multiscales, list):
            version_str = multiscales[0].get("version", "0.4")

    if version_str is None:
        raise ValueError("Could not detect NGFF version from root attributes.")

    return NgffVersion(version_str)
