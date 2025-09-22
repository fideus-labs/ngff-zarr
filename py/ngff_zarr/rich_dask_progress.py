# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
from typing import Dict, Optional, Set

from dask.callbacks import Callback
from rich.progress import TaskID


class NgffProgress:
    def __init__(self, rich_progress):
        self.rich = rich_progress

    def add_multiscales_task(self, description: str, scales: int):
        self.multiscales_task = self.rich.add_task(description, total=scales)

    def update_multiscales_task_completed(self, completed: int):
        self.rich.update(self.multiscales_task, completed=completed, refresh=True)

    def add_cache_task(self, description: str, total: int):
        self.cache_task = self.rich.add_task(description, total=total)

    def update_cache_task_completed(self, completed: int):
        self.rich.update(self.cache_task, completed=completed, refresh=True)


class NgffProgressCallback(Callback, NgffProgress):
    def __init__(self, rich_progress):
        self.rich = rich_progress
        self.tasks: Dict[str, Optional[TaskID]] = {}
        self.hide_after_finished: Set[str] = set()
        self.next_task = None

    def add_callback_task(self, description: str):
        self.next_task = description
        self.tasks[self.next_task] = self.rich.add_task(self.next_task)
        self.hide_after_finished.add(self.next_task)

    def _start(self, dsk):
        if self.next_task:
            description = self.next_task
            dsk["ngff_zarr_task"] = description

    def _start_state(self, dsk, state):
        pass

    def _pretask(self, key, dsk, state):  # noqa: ARG002
        if "ngff_zarr_task" in dsk:
            description = dsk["ngff_zarr_task"]
            task = self.tasks[description]
            ndone = len(state["finished"])
            ntasks = sum(len(state[k]) for k in ["ready", "waiting", "running"]) + ndone
            self.rich.update(task, total=ntasks, completed=ndone)

    def _posttask(self, key, result, dsk, state, worker_id):
        pass

    def _finish(self, dsk, state, errored):
        if "ngff_zarr_task" in dsk:
            description = dsk["ngff_zarr_task"]
            task = self.tasks[description]
            if not errored:
                ndone = len(state["finished"])
                ntasks = (
                    sum(len(state[k]) for k in ["ready", "waiting", "running"]) + ndone
                )
                self.rich.update(task, total=ntasks, completed=ndone)
                if description in self.hide_after_finished:
                    self.rich.update(task, visible=False)

            # self.rich.update(task, total=1.0, completed=1.0)
