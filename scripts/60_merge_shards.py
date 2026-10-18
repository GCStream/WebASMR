#!/usr/bin/env python3
"""Merge corpus shard files into one transcript set and report coverage.

    python scripts/60_merge_shards.py --shard-dir out/corpus
    python scripts/60_merge_shards.py --shard-dir out/corpus --out out/segments.jsonl

Reports anything that a silent partial pass would otherwise hide: files with
no segments, files that fell back to sub-windows, and total audio covered.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import probe_duration_s  # noqa: E402
from common.jsonlio import read_jsonl, write_jsonl  # noqa: E402
from common.keys import iter_audio_files, source_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=Path("out/corpus"))
    ap.add_argument("--audio-root", type=Path, default=Path("/root/data_raw/audio"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    rows, seen = [], {}
    dupes = 0
    for path in sorted(args.shard_dir.glob("segments.shard*.jsonl")):
        for row in read_jsonl(path):
            key = row["key"]
            if key in seen:
                dupes += 1
                continue
            seen[key] = True
            rows.append(row)
    rows.sort(key=lambda r: r["key"])

    corpus = {source_key(p, args.audio_root): probe_duration_s(p)
              for p in iter_audio_files(args.audio_root)}
    missing = sorted(set(corpus) - set(seen))
    audio_h = sum(corpus.get(r["key"], 0.0) for r in rows) / 3600
    total_h = sum(corpus.values()) / 3600

    empty = [r["key"] for r in rows if not r.get("segments")]
    fallback = [r["key"] for r in rows
                if any(w.get("fallback") for w in r.get("windows", []))]
    partial = [r["key"] for r in rows if r.get("status") in ("partial", "decode_failed")]
    speakers = sum(r.get("stats", {}).get("n_speakers", 0) for r in rows)
    segs = sum(len(r.get("segments", [])) for r in rows)

    print(f"rows            {len(rows)}")
    print(f"duplicate keys  {dupes}")
    print(f"audio covered   {audio_h:.1f} h of {total_h:.1f} h "
          f"({100 * audio_h / max(total_h, 1e-9):.1f}%)")
    print(f"segments        {segs}")
    print(f"speaker turns   {speakers}")
    print(f"no segments     {len(empty)}")
    print(f"sub-window fbk  {len(fallback)}")
    print(f"partial/failed  {len(partial)}")
    print(f"missing keys    {len(missing)}")
    for label, items in [("no segments", empty), ("partial", partial),
                         ("missing", missing), ("fallback", fallback)]:
        for key in items[:10]:
            print(f"  {label}: {key}")
        if len(items) > 10:
            print(f"  ... {len(items) - 10} more {label}")

    if args.out and not args.report_only:
        write_jsonl(args.out, rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()