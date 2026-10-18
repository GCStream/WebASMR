#!/usr/bin/env python3
"""Audit transcript quality before committing to a training/publishing format.

Not a WER check (there is no ground truth), but the failure modes that would
poison a finetuning set are measurable:

  text/timing sanity   characters per second of speech; Mandarin lands around
                       3-7, so a segment claiming 40 chars in 2 s or 2 chars in
                       15 s has a bad boundary or bad text
  boundary sanity      negative or heavily overlapping spans within one file
  leakage              the same line emitted twice in one file
  language mix         kana / latin / CJK ratios, for code-switch detection
  recall vs baseline   chars per audio hour against the Qwen3-ASR pass; a file
                       where VibeVoice produces far less text is a candidate
                       false negative, not a clean file
  sound-only           files the model labels entirely non-speech

    python tools/audit_quality.py
    python tools/audit_quality.py --samples 12
"""

import argparse
import collections
import glob
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonlio import read_jsonl  # noqa: E402
from common.segments import repetition_ratio, vocalization_ratio  # noqa: E402

BASELINE = Path("/root/LocalASMR/asmr-bilingual-webml/data/processed/"
                "zh_retranscribe/transcripts_qwen3asr.jsonl")
TAG_ANY = re.compile(r"\[([^\]]+)\]")
CJK = re.compile(r"[\u4e00-\u9fff]")
KANA = re.compile(r"[\u3040-\u30ff]")
LATIN = re.compile(r"[A-Za-z]")


def pct(values, p):
    if not values:
        return 0.0
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=Path("out/corpus"))
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--cps-low", type=float, default=1.5)
    ap.add_argument("--cps-high", type=float, default=9.0)
    args = ap.parse_args()

    rows, seen = [], set()
    for path in sorted(glob.glob(str(args.shard_dir / "segments.shard*.jsonl"))):
        for row in read_jsonl(path):
            if row["key"] in seen:
                continue
            seen.add(row["key"])
            rows.append(row)
    rows.sort(key=lambda r: r["key"])
    if not rows:
        print("no segments found")
        return

    total_audio = sum(r["duration_s"] for r in rows) / 3600
    all_segs = [(r["key"], s) for r in rows for s in r.get("segments", [])]
    print(f"files {len(rows)}   audio {total_audio:.1f} h   segments {len(all_segs)}")
    print(f"segments per file: median {statistics.median(len(r.get('segments', [])) for r in rows):.0f}"
          f"  p10 {pct([len(r.get('segments', [])) for r in rows], .1):.0f}"
          f"  p90 {pct([len(r.get('segments', [])) for r in rows], .9):.0f}")

    speech_s = sum(s["end"] - s["start"] for _, s in all_segs)
    spoken_chars = 0
    tag_only = 0
    durations, cps_values, overlaps, duplicates = [], [], 0, 0
    kana_chars = latin_chars = cjk_chars = 0
    tag_counts = collections.Counter()
    bad_cps = []

    for key, seg in all_segs:
        dur = seg["end"] - seg["start"]
        durations.append(dur)
        text = seg["text"]
        tags = TAG_ANY.findall(text)
        for t in tags:
            tag_counts[t] += 1
        spoken = TAG_ANY.sub("", text).strip()
        if not spoken:
            tag_only += 1
            continue
        spoken_chars += len(spoken)
        cjk_chars += len(CJK.findall(spoken))
        kana_chars += len(KANA.findall(spoken))
        latin_chars += len(LATIN.findall(spoken))
        if dur > 0.5:
            cps = len(spoken) / dur
            cps_values.append(cps)
            if cps < args.cps_low or cps > args.cps_high:
                bad_cps.append((round(cps, 2), round(dur, 1), spoken[:40], key[-40:]))

    for row in rows:
        segs = sorted(row.get("segments", []), key=lambda s: s["start"])
        texts = [TAG_ANY.sub("", s["text"]).strip() for s in segs]
        texts = [t for t in texts if t]
        duplicates += len(texts) - len(set(texts))
        for a, b in zip(segs, segs[1:]):
            if b["start"] < a["end"] - 0.5:
                overlaps += 1

    print(f"\ntext")
    print(f"  spoken characters           {spoken_chars}")
    print(f"  chars per audio hour        {spoken_chars / max(total_audio, 1e-9):.0f}")
    print(f"  tag-only segments           {tag_only} "
          f"({100 * tag_only / max(len(all_segs), 1):.1f}%)")
    print(f"  speech/sound ratio          {speech_s / max(sum(r['duration_s'] for r in rows), 1e-9):.3f}")
    print(f"  kana / latin / cjk chars    {kana_chars} / {latin_chars} / {cjk_chars}")
    print(f"  repeated lines in one file  {duplicates}")
    print(f"  overlapping spans           {overlaps}")

    print(f"\ntiming")
    print(f"  segment duration  median {statistics.median(durations):.1f}s"
          f"  p90 {pct(durations, .9):.1f}s  p99 {pct(durations, .99):.1f}s"
          f"  max {max(durations):.1f}s")
    print(f"  chars/second      median {statistics.median(cps_values):.2f}"
          f"  p10 {pct(cps_values, .1):.2f}  p90 {pct(cps_values, .9):.2f}")
    print(f"  outside [{args.cps_low}, {args.cps_high}] cps: {len(bad_cps)} "
          f"({100 * len(bad_cps) / max(len(cps_values), 1):.1f}%)")
    for cps, dur, text, key in sorted(bad_cps)[-3:]:
        print(f"    {cps:6.2f} cps  {dur:5.1f}s  {text:<40} {key}")
    for cps, dur, text, key in sorted(bad_cps)[:3]:
        print(f"    {cps:6.2f} cps  {dur:5.1f}s  {text:<40} {key}")

    print(f"\ncoverage per file")
    cov = [r.get("stats", {}).get("coverage", 0.0) for r in rows]
    print(f"  median {statistics.median(cov):.2f}  p10 {pct(cov, .1):.2f}"
          f"  files under 0.3: {sum(1 for c in cov if c < 0.3)}")
    spk = [r.get("stats", {}).get("n_speakers", 0) for r in rows]
    print(f"  speakers per file: {dict(collections.Counter(spk).most_common())}")
    print(f"  files with 0 speech segments: "
          f"{sum(1 for r in rows if not r.get('segments'))}")

    print(f"\nmost common tags: {tag_counts.most_common(10)}")

    if args.baseline.exists():
        base = collections.defaultdict(float)
        for row in read_jsonl(args.baseline):
            src = row["key"].rsplit("#", 1)[0]
            base[src] += len((row.get("transcript") or "").strip())
        ours = collections.defaultdict(int)
        for row in rows:
            ours[row["key"]] = len(TAG_ANY.sub("", row.get("transcript", "")).strip())
        ratios = []
        low = []
        for row in rows:
            b = base.get(row["key"])
            if not b:
                continue
            ratio = ours[row["key"]] / b
            ratios.append(ratio)
            if ratio < 0.5 and row["duration_s"] > 120:
                low.append((round(ratio, 2), row["key"], ours[row["key"]], int(b)))
        print(f"\nrecall vs qwen3asr baseline ({len(ratios)} files comparable)")
        print(f"  chars ratio  median {statistics.median(ratios):.2f}"
              f"  p10 {pct(ratios, .1):.2f}  p90 {pct(ratios, .9):.2f}")
        print(f"  files under 0.5 ratio: {len(low)}")
        for ratio, key, o, b in sorted(low)[:8]:
            print(f"    {ratio:5.2f}  ours={o:5d} base={b:5d}  {key[-52:]}")

    if args.samples:
        print(f"\nrandom segments ({args.samples})")
        import random
        rng = random.Random(11)
        pool = [(k, s) for k, s in all_segs if TAG_ANY.sub("", s["text"]).strip()]
        for key, seg in rng.sample(pool, min(args.samples, len(pool))):
            dur = seg["end"] - seg["start"]
            print(f"  [{seg['start']:7.1f}-{seg['end']:7.1f}] {dur:5.1f}s spk{seg['speaker']}  "
                  f"{seg['text'][:90]}")


if __name__ == "__main__":
    main()