#!/usr/bin/env python3
"""Corpus transcription pass: every raw file -> absolute-time segments.

Window strategy, longest first:

  1. one pass per 60 min window (the model's documented single-pass limit)
  2. a pass that degenerates into a repetition loop, returns nothing, or
     produces unparseable output is retried on smaller sub-windows, because a
     file the model loops on is usually still transcribable a chunk at a time
  3. sub-window results are merged and overlap-duplicated

Batching: `--batch-size N` puts N clips in one `generate()` call. Decode is
bandwidth-bound at batch 1 (every step streams the full 16.7 GB of weights), so
batching raises throughput where extra worker processes cannot. Files are
sorted by duration, so consecutive files batch without padding waste, and
`--batch-max-audio-s` caps the longest member to bound memory.

Resumable: one JSONL row per source file, appended as each file completes.

    python scripts/30_transcribe.py --limit 20
    python scripts/30_transcribe.py --batch-size 4
    python scripts/30_transcribe.py --limit 0 --batch-size 4   # whole corpus
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.audio import load_24k_mono, pass_windows, probe_duration_s  # noqa: E402
from common.jsonlio import JsonlAppender, done_keys  # noqa: E402
from common.keys import iter_audio_files, source_key  # noqa: E402
from common.segments import (dedupe_overlap, file_stats, normalize_segments,  # noqa: E402
                             transcript_of)
from importlib import import_module  # noqa: E402

BAD_STATUSES = {"unparseable", "empty", "truncated"}
MAX_AUDIO = 24000


def resume_keys(out_path: Path) -> set:
    """Completed keys from this worker's output *and* its sibling shards.

    Reading only its own file makes a worker re-transcribe everything another
    shard already finished, which is what happens when the shard count changes
    between runs: switching a 1-shard pass to 3 shards redid 59 files that
    already had transcripts.
    """
    keys = done_keys(out_path)
    for sibling in sorted(out_path.parent.glob("segments.shard*.jsonl")):
        if sibling != out_path:
            keys |= done_keys(sibling)
    return keys


def build_row(key, duration, segments, windows, status=None):
    segments = dedupe_overlap(segments)
    if status is None:
        status = "ok" if segments else "no_segments"
        if any(w["status"] == "unparseable" for w in windows):
            status = "partial"
    return {
        "key": key,
        "duration_s": round(duration, 2),
        "windows": windows,
        "status": status,
        "segments": segments,
        "stats": file_stats(segments, duration),
        "transcript": transcript_of(segments),
    }


def sub_window_pass(probe, processor, model, audio, start_s, end_s, prompt,
                    args) -> tuple[list[dict], dict]:
    """Re-run a failed window as smaller sub-windows and merge."""
    step = args.fallback_s[0]
    merged, infos = [], []
    t = start_s
    while t < end_s - 0.01:
        sub_end = min(t + step, end_s)
        lo, hi = int((t - start_s) * MAX_AUDIO), int((sub_end - start_s) * MAX_AUDIO)
        rows, _ = probe.transcribe_batch(processor, model, [audio[lo:hi]], prompt,
                                         args.max_new_tokens, args.retries)
        raw, status, _ = rows[0]
        segs = normalize_segments(raw, t, sub_end)
        infos.append({"start": round(t, 2), "end": round(sub_end, 2),
                      "status": status, "segments": len(segs), "fallback": True})
        merged += segs
        if sub_end >= end_s:
            break
        t = sub_end
    return merged, infos


def transcribe_file(probe, processor, model, path, args):
    """Full single-file path: 60 min windows, then sub-window fallback."""
    key = source_key(path, args.audio_root)
    duration = probe_duration_s(path)
    try:
        audio = load_24k_mono(path)
    except RuntimeError as exc:
        return {"key": key, "duration_s": round(duration, 2), "windows": [],
                "status": "decode_failed", "error": str(exc)[:200],
                "segments": [], "stats": {}, "transcript": ""}

    segments, infos = [], []
    for start_s, end_s in pass_windows(duration, args.max_pass_s):
        lo, hi = int(start_s * MAX_AUDIO), int(end_s * MAX_AUDIO)
        window = audio[lo:hi]
        rows, elapsed = probe.transcribe_batch(processor, model, [window], prompt=args.prompt,
                                               max_new_tokens=args.max_new_tokens,
                                               retries=args.retries)
        raw, status, _ = rows[0]
        segs = normalize_segments(raw, start_s, end_s)
        info = {"start": round(start_s, 2), "end": round(end_s, 2), "status": status,
                "elapsed_s": round(elapsed, 1), "fallback": False, "segments": len(segs)}
        if status in BAD_STATUSES and (end_s - start_s) > args.fallback_s[0]:
            sub_segs, sub_infos = sub_window_pass(probe, processor, model, window,
                                                  0.0, end_s - start_s, args.prompt, args)
            info["fallback"] = True
            info["fallback_from"] = status
            info["fallback_windows"] = sub_infos
            segs = [{**s, "start": s["start"] + start_s, "end": s["end"] + start_s}
                    for s in sub_segs]
            info["segments"] = len(segs)
        segments += segs
        infos.append(info)

    return build_row(key, duration, segments, infos)


def form_batches(todo, args) -> list[list]:
    """Group duration-sorted files so batches have similar lengths."""
    batches, current = [], []
    for duration, path in todo:
        too_long = duration > args.batch_max_audio_s
        if current and (len(current) >= args.batch_size or too_long):
            batches.append(current)
            current = []
        if too_long:
            if not batches or batches[-1]:
                batches.append([(duration, path)])
            continue
        current.append((duration, path))
        if len(current) == args.batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def transcribe_batch_group(probe, processor, model, group, args):
    """Transcribe a batch of similar-length files; fall back per file on failure."""
    paths = [p for _, p in group]
    durations = [d for d, _ in group]
    audios, keys, infos = [], [], []
    for duration, path in group:
        key = source_key(path, args.audio_root)
        try:
            audio = load_24k_mono(path)
        except RuntimeError as exc:
            return [transcribe_file(probe, processor, model, path, args)]
        audios.append(audio)
        keys.append(key)

    rows, elapsed = probe.transcribe_batch(processor, model, audios, args.prompt,
                                           args.max_new_tokens, args.retries)
    out = []
    for (raw, status, _), audio, key, duration in zip(rows, audios, keys, durations):
        if status in BAD_STATUSES:
            out.append(transcribe_file(probe, processor, model,
                                       Path(args.audio_root / key), args))
            continue
        segs = normalize_segments(raw, 0.0, duration)
        infos = [{"start": 0.0, "end": round(duration, 2), "status": status,
                  "elapsed_s": round(elapsed / len(audios), 1), "fallback": False,
                  "segments": len(segs)}]
        out.append(build_row(key, duration, segs, infos))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", type=Path, default=Path("/root/data_raw/audio"))
    ap.add_argument("--out", type=Path, default=Path("out/segments.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--min-duration", type=float, default=0.0)
    ap.add_argument("--max-duration", type=float, default=0.0)
    ap.add_argument("--max-pass-s", type=float, default=3600.0)
    ap.add_argument("--fallback-s", type=float, nargs="*", default=[120.0],
                    help="sub-window sizes tried when a pass fails")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="0 = scale the budget to the window length")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="clips per generate() call; 1 disables batching")
    ap.add_argument("--batch-max-audio-s", type=float, default=900.0,
                    help="longest clip allowed in a batch, to bound padding waste")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the work across N processes; each takes every Nth file")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--keys-file", type=Path, default=None,
                    help="JSON list of source keys to process, ignoring shard/limit "
                         "selection; used by the concurrency benchmark")
    ap.add_argument("--model-id", default="microsoft/VibeVoice-ASR-HF")
    args = ap.parse_args()

    if not args.fallback_s:
        ap.error("--fallback-s needs at least one value")

    files = [(probe_duration_s(p), p) for p in iter_audio_files(args.audio_root)]
    files = [(d, p) for d, p in files
             if d >= args.min_duration and (not args.max_duration or d <= args.max_duration)]
    files.sort(key=lambda dp: dp[0])
    total_files = len(files)

    if args.keys_file:
        wanted = set(json.loads(Path(args.keys_file).read_text()))
        files = [(d, p) for d, p in files if source_key(p, args.audio_root) in wanted]
    elif args.num_shards > 1:
        if not 0 <= args.shard < args.num_shards:
            ap.error(f"--shard must be in [0, {args.num_shards})")
        files = [(d, p) for i, (d, p) in enumerate(files) if i % args.num_shards == args.shard]

    done = resume_keys(args.out)
    todo = [(d, p) for d, p in files if source_key(p, args.audio_root) not in done]
    if args.limit and not args.keys_file:
        todo = todo[:args.limit]

    total_h = sum(d for d, _ in files) / 3600
    batches = form_batches(todo, args)
    print(f"corpus {len(files)} files / {total_h:.1f} h; {len(todo)} to run "
          f"({len(done)} already done); shard {args.shard + 1}/{args.num_shards} "
          f"of {total_files} files; {len(batches)} batches of <= {args.batch_size}",
          flush=True)
    if not todo:
        return

    probe = import_module("10_probe_model")
    processor, model = probe.load_model(args.model_id)

    started = time.time()
    completed = 0
    with JsonlAppender(args.out) as sink:
        for group in batches:
            t0 = time.time()
            rows = (transcribe_batch_group(probe, processor, model, group, args)
                    if len(group) > 1 else
                    [transcribe_file(probe, processor, model, group[0][1], args)])
            batch_wall = time.time() - t0
            for row in rows:
                sink.write(row)
                completed += 1
                print(f"[{completed}/{len(todo)}] {row['key']}: "
                      f"{row['duration_s'] / 60:.1f} min segs={len(row.get('segments', []))} "
                      f"spk={row.get('stats', {}).get('n_speakers', 0)} "
                      f"chars={len(row.get('transcript', ''))} "
                      f"[{row['status']}]", flush=True)
            done_h = sum(d for d, _ in todo[:completed]) / 3600
            print(f"  batch of {len(group)} took {batch_wall:.0f}s; "
                  f"{done_h / max((time.time() - started) / 3600, 1e-9):.1f} audio-h/h",
                  flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()