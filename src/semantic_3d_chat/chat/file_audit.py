from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, ClassVar, Self


class FileAccessAudit:
    """Record path-based open events emitted by Python's process-wide audit hook."""

    _installed: ClassVar[bool] = False
    _instances: ClassVar[list[FileAccessAudit]] = []
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, forbidden_roots: list[str | Path] | None = None) -> None:
        self.active = False
        self.paths: list[str] = []
        self.forbidden_roots = [Path(path).resolve() for path in (forbidden_roots or [])]
        with self._lock:
            self._instances.append(self)
            if not self.__class__._installed:
                sys.addaudithook(self.__class__._hook)
                self.__class__._installed = True

    @classmethod
    def _hook(cls, event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 else None
        if isinstance(mode, str) and "r" not in mode and "+" not in mode:
            return
        try:
            candidate = Path(raw).expanduser().resolve()
            # CPython emits the hook before opening.  Missing paths are failed
            # probes, not loaded files, and output-only opens are filtered above.
            if not candidate.exists():
                return
            path = str(candidate)
        except (OSError, TypeError, ValueError):
            path = os.fsdecode(raw)
        with cls._lock:
            for instance in cls._instances:
                if instance.active:
                    instance.paths.append(path)

    def __enter__(self) -> Self:
        self.paths.clear()
        self.active = True
        return self

    def __exit__(self, *_: object) -> None:
        self.active = False

    @property
    def unique_paths(self) -> list[str]:
        return sorted(set(self.paths))

    def record(self, path: str | Path) -> None:
        """Explicitly record native-extension reads that may bypass Python's hook."""

        resolved = str(Path(path).expanduser().resolve())
        with self._lock:
            if self.active:
                self.paths.append(resolved)

    def forbidden_accesses(self) -> list[str]:
        violations = []
        for raw_path in self.unique_paths:
            path = Path(raw_path)
            for root in self.forbidden_roots:
                try:
                    path.relative_to(root)
                except ValueError:
                    continue
                violations.append(raw_path)
                break
        return violations

    def assert_clean(self) -> None:
        if violations := self.forbidden_accesses():
            raise RuntimeError(f"Forbidden runtime file access detected: {violations}")

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "loaded_files": self.unique_paths,
            "forbidden_roots": [str(path) for path in self.forbidden_roots],
            "forbidden_accesses": self.forbidden_accesses(),
            "passed": not self.forbidden_accesses(),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
