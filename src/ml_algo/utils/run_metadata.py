from __future__ import annotations

import dataclasses
import platform
import socket
import time
from pathlib import Path
from typing import Any, Dict

import subprocess


@dataclasses.dataclass(frozen=True)
class RunMetadata:
    """Container for reproducibility info (SPEC Appendix B)."""

    start_ts: float
    python_version: str
    platform: str
    hostname: str
    git_sha: str | None = None
    env_hash: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _detect_git_sha() -> str | None:
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            .stdout.strip()
        )
        return sha or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_metadata(git_sha: str | None = None, env_hash: str | None = None) -> RunMetadata:
    """Gather baseline metadata; extend once run registry implemented."""
    sha = git_sha if git_sha is not None else _detect_git_sha()
    return RunMetadata(
        start_ts=time.time(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        hostname=socket.getfqdn(),
        git_sha=sha,
        env_hash=env_hash,
    )


def persist_metadata(metadata: RunMetadata, out_path: Path) -> None:
    """Persist metadata JSON for audits."""
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata.as_dict(), fh, indent=2)
