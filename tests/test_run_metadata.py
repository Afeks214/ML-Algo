from __future__ import annotations

from ml_algo.utils.run_metadata import collect_metadata


def test_collect_metadata_has_core_fields() -> None:
    meta = collect_metadata()
    assert meta.python_version
    assert meta.platform
    assert meta.hostname
