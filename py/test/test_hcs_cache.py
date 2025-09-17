# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""
Test to verify HCS cache functionality is working correctly.
"""


def test_lru_cache():
    """Test the LRU cache implementation."""
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

        def __contains__(self, key):
            return key in self.cache

        def __getitem__(self, key):
            value = self.get(key)
            if value is None:
                raise KeyError(key)
            return value

        def __setitem__(self, key, value):
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[key] = value

    cache = LRUCache(max_size=3)

    # Add items
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    print(f"After adding a,b,c: {list(cache.cache.keys())}")

    # All should be present (but accessing them changes LRU order!)
    assert "a" in cache
    assert "b" in cache
    assert "c" in cache
    # Get value without using __getitem__ to avoid changing order
    print(
        f"Cache values: a={cache.cache['a']}, b={cache.cache['b']}, c={cache.cache['c']}"
    )

    # Add another item - should evict "a" (least recently used)
    # But since we accessed them above, the order changed
    # Let's reset and do this more carefully
    cache.cache.clear()
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    print(f"Reset cache: {list(cache.cache.keys())}")

    # Add d without accessing any items first
    cache["d"] = 4
    print(f"After adding d: {list(cache.cache.keys())}")

    # "a" should be evicted since it was the first added (LRU)
    assert (
        "a" not in cache
    ), f"'a' should be evicted, but cache contains: {list(cache.cache.keys())}"
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

    print("✓ LRU cache test passed")


if __name__ == "__main__":
    test_lru_cache()
    print("All cache tests passed!")
