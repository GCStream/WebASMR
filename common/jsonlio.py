"""Append-only JSONL with resume, tolerant of a truncated last line.

A transcription pass over 131 h of audio is long enough that it will be
interrupted; every stage therefore re-reads its own output, skips completed
keys, and appends as it goes.
"""

import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def done_keys(path: str | Path, key: str = "key") -> set:
    keys = set()
    for row in read_jsonl(path):
        k = row.get(key)
        if k is not None:
            keys.add(k)
    return keys


class JsonlAppender:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None

    def __enter__(self):
        self._f = open(self.path, "a", encoding="utf-8")
        return self

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        return False


def write_jsonl(path: str | Path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")