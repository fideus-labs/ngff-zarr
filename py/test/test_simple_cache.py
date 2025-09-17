#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Simple test of LRU cache.
"""


def test_lru_cache():
    from collections import OrderedDict

    class LRUCache:
        def __init__(self, max_size: int = 100):
            self.max_size = max_size
            self.cache = OrderedDict()

        def get(self, key):
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

        def set(self, key, value):
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                if len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)

        def __contains__(self, key):
            return key in self.cache

        def __getitem__(self, key):
            value = self.get(key)
            if value is None:
                raise KeyError(key)
            return value

        def __setitem__(self, key, value):
            self.set(key, value)

    # Test with small cache size
    cache = LRUCache(max_size=3)

    # Add items
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3

    # All should be present
    assert "a" in cache
    assert "b" in cache
    assert "c" in cache
    assert cache["a"] == 1
    assert cache["b"] == 2
    assert cache["c"] == 3

    # Add another item - should evict "a" (least recently used)
    cache["d"] = 4

    # "a" should be evicted, others should remain
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache
    assert "d" in cache

    # Access "b" to make it most recently used
    _ = cache["b"]

    # Add another item - should evict "c"
    cache["e"] = 5

    assert "b" in cache  # recently accessed
    assert "c" not in cache  # evicted
    assert "d" in cache
    assert "e" in cache

    print("✓ LRU cache working correctly")


if __name__ == "__main__":
    test_lru_cache()
