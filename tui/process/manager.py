"""Two-slot subprocess manager for the TUI.

Slots:
- ``training``: one of {train, train-team-*, resume, resume-team}. Mutex among
  themselves; running while ``aux`` runs is OK.
- ``aux``: one of {demos, eval, eval-team, tensorboard, promote, list-runs,
  lineage, repro}. Mutex among themselves; running while ``training`` runs is OK.

Output lines are captured into a ring buffer per slot for the log overlay.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Iterable, Literal

SlotName = Literal["training", "aux"]
_RING = 1000


class ManagedProcess:
    def __init__(self, argv: list[str], env: dict[str, str] | None,
                 run_dir: Path | None = None) -> None:
        self.argv = argv
        self.run_dir = run_dir
        self._lines: deque[str] = deque(maxlen=_RING)
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env={**os.environ, **(env or {})},
        )
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.append(line.rstrip("\n"))

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def returncode(self) -> int | None:
        return self._proc.returncode

    def tail(self, n: int = 200) -> list[str]:
        return list(self._lines)[-n:]

    def stop(self, *, escalate_after_sec: float = 5.0) -> None:
        if not self.is_alive():
            return
        self._proc.send_signal(signal.SIGINT)
        try:
            self._proc.wait(timeout=escalate_after_sec)
        except subprocess.TimeoutExpired:
            self._proc.terminate()


class ProcessManager:
    def __init__(self) -> None:
        self._slots: dict[SlotName, ManagedProcess | None] = {"training": None, "aux": None}

    def start(self, slot: SlotName, argv: list[str], *,
              env: dict[str, str] | None = None,
              run_dir: Path | None = None) -> ManagedProcess:
        current = self._slots[slot]
        if current is not None and current.is_alive():
            raise RuntimeError(f"slot {slot!r} is already running; stop it first")
        # Reap a dead process so a new one can start.
        proc = ManagedProcess(argv, env=env, run_dir=run_dir)
        self._slots[slot] = proc
        return proc

    def stop(self, slot: SlotName) -> None:
        p = self._slots[slot]
        if p is not None:
            p.stop()

    def is_running(self, slot: SlotName) -> bool:
        p = self._slots[slot]
        return p is not None and p.is_alive()

    def current(self, slot: SlotName) -> ManagedProcess | None:
        p = self._slots[slot]
        if p is None:
            return None
        return p if p.is_alive() else None

    def stop_all(self) -> None:
        for slot in self._slots:
            self.stop(slot)
