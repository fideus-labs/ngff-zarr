# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Call ITK-Wasm pipeline functions safely from dask's worker threads.

Each ``itkwasm_downsample`` function builds its ``Pipeline`` on first use,
unguarded (``if _pipeline is None``), and ``Pipeline.__init__`` memory-maps the
compiled-module cache file with ``Module.deserialize_file`` or, when the cache
is missing or stale, compiles and rewrites that same file with
``open("wb")``. Two threads reaching a function's first call together can
truncate the file one of them has mapped: the process dies of SIGBUS. Once a
function's pipeline exists, its runs are independent (one wasmtime store
each), so only the first call of each function is serialized.
"""

import threading
from collections.abc import Callable
from typing import Any

_lock = threading.Lock()
_ready: set[Callable] = set()


def call_itkwasm(function: Callable, *args: Any, **kwargs: Any) -> Any:
    """``function(*args, **kwargs)``, its first call in the process under a lock."""
    if function in _ready:
        return function(*args, **kwargs)
    with _lock:
        if function not in _ready:  # the threads that waited here run in parallel
            result = function(*args, **kwargs)
            _ready.add(function)
            return result
    return function(*args, **kwargs)
