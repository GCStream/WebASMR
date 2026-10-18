#!/usr/bin/env python3
"""Launch the full corpus pass as N concurrent shard workers.

Neither default is a guess, and the two were measured against each other on an
idle GPU (`42_sweep_bench.py`, `docs/concurrency.md`):

  2 workers x batch 1   18.1 audio-h/h   <- best
  2 workers x batch 2   14.5
  1 worker  x batch 1   11.9
  1 worker  x batch 4    9.7

Worker processes win because batching pads every clip's audio to the longest
member, which costs ~20% in redundant prefill when the GPU is not already
time-sliced with another job. Batching does pay off when the card is shared
(1.97x in `41_batch_bench.py`), so pass `--batch-size 4` if the pass has to
share the GPU with a training run.

    python scripts/50_run_corpus.py --workers 2
    python scripts/50_run_corpus.py --workers 2 --dry-run
    python scripts/50_run_corpus.py --workers 2 --min-duration 1800   # long files only

Each worker writes its own shard file, so a killed run resumes from the last
completed file per shard. Merge with `60_merge_shards.py`.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.audio import probe_duration_s  # noqa: E402
from common.jsonlio import done_keys  # noqa: E402
from common.keys import iter_audio_files, source_key  # noqa: E402

SCRIPT = HERE / "30_transcribe.py"
MEASURED_WORKERS = 3
MEASURED_BATCH = 1


def plan(audio_root: Path, args) -> list[list[str]]:
    rows = [(probe_duration_s(p), source_key(p, audio_root))
            for p in iter_audio_files(audio_root)]
    rows = [(d, k) for d, k in rows
            if d >= args.min_duration and (not args.max_duration or d <= args.max_duration)]
    rows.sort(key=lambda r: r[0])
    shards = [rows[i::args.workers] for i in range(args.workers)]
    return [[k for _, k in shard] for shard in shards]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", type=Path, default=Path("/root/data_raw/audio"))
    ap.add_argument("--workers", type=int, default=MEASURED_WORKERS)
    ap.add_argument("--out-dir", type=Path, default=Path("out/corpus"))
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--max-duration", type=float, default=0.0)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--batch-size", type=int, default=MEASURED_BATCH,
                    help="clips per generate() call, measured sweet spot on this GPU")
    ap.add_argument("--batch-max-audio-s", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = plan(args.audio_root, args)
    todo_total, done_total = 0, 0
    commands = []
    for i, keys in enumerate(shards):
        out_path = args.out_dir / f"segments.shard{i:02d}.jsonl"
        done = done_keys(out_path)
        pending = [k for k in keys if k not in done]
        todo_total += len(pending)
        done_total += len(done)
        hours = sum(probe_duration_s(args.audio_root / k) for k in pending) / 3600
        print(f"shard {i}: {len(keys)} files, {len(done)} done, {len(pending)} pending "
              f"({hours:.1f} audio-h)")
        cmd = [sys.executable, str(SCRIPT),
               "--out", str(out_path),
               "--num-shards", str(args.workers), "--shard", str(i),
               "--batch-size", str(args.batch_size),
               "--batch-max-audio-s", str(args.batch_max_audio_s),
               "--audio-root", str(args.audio_root)]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        commands.append(cmd)

    print(f"\ntotal: {done_total} done, {todo_total} pending across {args.workers} workers")
    if args.dry_run:
        for cmd in commands:
            print("  " + " ".join(cmd))
        return

    procs = []
    for i, cmd in enumerate(commands):
        log = (args.out_dir / f"shard{i:02d}.log").open("a")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"}), log))
    print(f"launched {len(procs)} workers; logs in {args.out_dir}", flush=True)

    failed = 0
    while any(p.poll() is None for p, _ in procs):
        time.sleep(30)
    for p, log in procs:
        log.close()
        if p.returncode != 0:
            failed += 1
    print(f"all workers exited, {failed} failed; merge with scripts/60_merge_shards.py",
          flush=True)


if __name__ == "__main__":
    main()