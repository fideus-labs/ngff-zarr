# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Pyodide only: compute the itkwasm tasks of a dask graph on a pyodide-pool.

A dask array graph is mostly plumbing: slicing, halo concatenation, casts.
Shipping all of it to pool workers would move every block several times, so
only the ngff-zarr block functions that run an itkwasm pipeline travel; the
driver evaluates the rest and writes the finished blocks itself. Workers never
see a zarr store: store-backed leaves are read on the driver.
"""

import asyncio

import dask
from dask._task_spec import Task, convert_legacy_graph
from dask.array.core import slices_from_chunks
from dask.base import collections_to_expr
from dask.core import flatten

from .config import config

#: Tasks calling into this module run on the pool; everything else is cheap.
_POOL_TASK_MODULE = "ngff_zarr.methods._itkwasm"


def pool():
    """The registered ``pyodide_pool`` pool, or ``None`` to compute serially."""
    if not config.pyodide_pool:
        return None
    try:
        import pyodide_pool

        return pyodide_pool.default_pool()
    except (ImportError, RuntimeError):  # not installed, or no JS pool registered
        return None


def _is_pool_task(node) -> bool:
    return isinstance(node, Task) and (
        getattr(node.func, "__module__", None) == _POOL_TASK_MODULE
    )


async def _compute(pool, dsk, keys, consume):
    """Evaluate *dsk*; call ``consume(key, value)`` for each of *keys*.

    Pool tasks run concurrently on the workers, at most two in flight per
    worker, so the driver reads input only as fast as the workers consume it;
    it runs the rest as soon as their inputs exist and drops every intermediate
    value once its last dependent has consumed it.
    """
    dependents = {key: set() for key in dsk}
    for key, node in dsk.items():
        for dep in node.dependencies:
            dependents[dep].add(key)
    waiting = {key: len(node.dependencies) for key, node in dsk.items()}
    uses = {key: len(deps) for key, deps in dependents.items()}
    requested = set(keys)
    cache, pending = {}, {}
    ready = [key for key, count in waiting.items() if count == 0]

    def finish(key, value):
        if key in requested:
            consume(key, value)
        if uses[key]:
            cache[key] = value
        for dep in dsk[key].dependencies:
            uses[dep] -= 1
            if uses[dep] == 0:
                cache.pop(dep, None)
        for dependent in dependents[key]:
            waiting[dependent] -= 1
            if waiting[dependent] == 0:
                ready.append(dependent)

    try:
        while ready or pending:
            while ready and len(pending) < 2 * pool.pool_size:
                key = ready.pop()
                node = dsk[key]
                values = {dep: cache[dep] for dep in node.dependencies}
                if _is_pool_task(node):
                    pending[key] = asyncio.ensure_future(pool.submit(node, values))
                else:
                    finish(key, node(values))
            if pending:
                done, _ = await asyncio.wait(
                    pending.values(), return_when=asyncio.FIRST_COMPLETED
                )
                for key in [k for k, future in pending.items() if future in done]:
                    finish(key, pending.pop(key).result())
    except BaseException:
        for future in pending.values():
            future.cancel()
        raise


def store_on_pool(pool, darr, target, region=None) -> None:
    """Write dask array *darr* into *target* (at *region*), computing on *pool*."""
    from ._js_zarr import _block_on

    # Fused tasks would bundle an itkwasm call with the halo slices of all its
    # neighbours, or with the store read feeding it; unfused, each pool task
    # takes exactly one finished block.
    with dask.config.set({"optimization.fuse.active": False}):
        expr = collections_to_expr([darr], optimize_graph=True).optimize()
        dsk = convert_legacy_graph(dict(expr.__dask_graph__()))
        keys = list(flatten(expr.__dask_keys__()))
    region = tuple(region) if region is not None else ()
    region += (slice(None),) * (darr.ndim - len(region))
    offsets = [s.start or 0 for s in region]
    slices = dict(zip(keys, slices_from_chunks(darr.chunks)))

    def consume(key, block):
        target[
            tuple(
                # The step survives so a stepped region fails in the adapter,
                # which rejects steps, instead of landing contiguously.
                slice(s.start + offset, s.stop + offset, r.step)
                for s, offset, r in zip(slices[key], offsets, region)
            )
        ] = block

    # shortcut: pool reads go through the driver, upgrade in G3
    _block_on(_compute(pool, dsk, keys, consume))
