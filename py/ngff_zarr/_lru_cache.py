# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""A small least-recently-used cache shared by the lazy readers."""

from collections import OrderedDict


class LRUCache:
    """
    Least Recently Used (LRU) cache with size limit for HCS image caching.

    This prevents unbounded memory growth when accessing many images
    from large plates.
    """

    def __init__(self, max_size: int = 100):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            # Update existing item
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove least recently used item before adding
                self.cache.popitem(last=False)
            self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        """Support dict-like access."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        """Support dict-like assignment."""
        self.set(key, value)

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
