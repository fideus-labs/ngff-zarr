# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""The oztest validate_zarr cases, answered by the oztest dingus and scored the
way the oztest runner scores them. The cases not passed yet are listed in
oztest_known_failures.txt; any other result fails, including a listed case
that passes."""

from pathlib import Path

from ngff_zarr.oztest_dingus import validate_zarr

from ._data import oztest_cases_dir


def _status(store: Path, validity: str) -> str:
    try:
        answer = validate_zarr(store)
    except Exception:
        return "error"
    if answer["validity"] == validity:
        return "pass"
    return "xfail" if answer.get("xfail") else "fail"


def test_oztest_cases():
    known = {}
    for line in (
        Path(__file__).with_name("oztest_known_failures.txt").read_text().splitlines()
    ):
        if entry := line.partition("#")[0].split():
            known[entry[1]] = entry[0]
    cases = oztest_cases_dir()
    got = {}
    for store in (cases / "validate_zarr").rglob("*.zarr"):
        rel = store.relative_to(cases)
        slug = (rel.parent / rel.name.split(".")[0]).as_posix()
        got[slug] = _status(store, slug.split("/")[3])
    unexpected = [
        f"{slug}: expected {known.get(slug, 'pass')}, got {status}"
        for slug, status in sorted(got.items())
        if status != known.get(slug, "pass")
    ]
    unexpected += [
        f"{slug}: listed, but not a case" for slug in known.keys() - got.keys()
    ]
    assert not unexpected, "\n".join(unexpected)
