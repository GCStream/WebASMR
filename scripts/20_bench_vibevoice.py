#!/usr/bin/env python3
"""A/B VibeVoice-ASR against the existing clip-level baselines.

The baselines (whisper-turbo, Qwen3-ASR) transcribed fixed 30 s slices, so
their rows carry no usable timing. This script reconstructs each baseline
clip's absolute time span from its cumulative `duration_s` (verified to sum
to the true file duration), restricts every system to the same window, and
compares them on metrics that matter for ASMR training data:

  repetition      runaway "啊啊啊啊..." hallucination, the known whisper failure
  vocalization    share of bare moans, i.e. sound-only stretches transcribed
  cut_rate        segments starting on a continuation particle, i.e. boundaries
                  that landed mid-word. Fixed 30 s cuts should score badly here.
  speakers        roleplay is two-person; a diarizing model should see 2
  coverage        transcribed seconds over window seconds
  agreement       char-level similarity to the baseline (corroboration, not truth)

    python scripts/20_bench_vibevoice.py --n-files 9 --window-s 300
"""

import argparse
import difflib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import load_24k_mono, probe_duration_s  # noqa: E402
from common.jsonlio import JsonlAppender, done_keys  # noqa: E402
from common.keys import clip_key, iter_audio_files, source_key  # noqa: E402
from common.segments import (CUT_STARTERS, file_stats, normalize_segments,  # noqa: E402
                             repetition_ratio, transcript_of, vocalization_ratio)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

BASELINE_DIR = Path("/root/LocalASMR/asmr-bilingual-webml/data/processed/zh_retranscribe")
AUDIO_ROOT = Path("/root/data_raw/audio")
MODEL_ID = "microsoft/VibeVoice-ASR-HF"


def clip_spans(transcript_path: Path) -> dict[str, list[tuple[int, float, float, str]]]:
    """Rebuild absolute [start, end) spans for every clip of every source file."""
    rows = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = r.get("key", "")
            src, _, idx = key.rpartition("#")
            if not idx:
                continue
            rows.setdefault(src, []).append((int(idx), r.get("duration_s") or 0.0,
                                            r.get("transcript") or ""))
    spans = {}
    for src, clips in rows.items():
        clips.sort()
        t = 0.0
        out = []
        for idx, dur, text in clips:
            out.append((idx, t, t + dur, text))
            t += dur
        spans[src] = out
    return spans


def baseline_window(spans, start_s: float, end_s: float) -> tuple[str, int, float]:
    """Concatenate baseline text for clips overlapping the window."""
    parts, n, covered = [], 0, 0.0
    for idx, s, e, text in spans:
        if e <= start_s or s >= end_s:
            continue
        parts.append(text.strip())
        n += 1
        covered += min(e, end_s) - max(s, start_s)
    return " ".join(p for p in parts if p), n, covered


def cut_rate(texts) -> float:
    texts = [t for t in texts if t]
    if not texts:
        return 0.0
    return round(sum(1 for t in texts if t[0] in CUT_STARTERS) / len(texts), 4)


def vitals(text: str, n_units: int, coverage: float) -> dict:
    return {
        "units": n_units,
        "chars": len(text),
        "repetition": round(repetition_ratio(text), 4),
        "vocalization": round(vocalization_ratio(text), 4),
        "coverage": round(coverage, 4),
    }


def sample_files(audio_root: Path, n_files: int, seed: int, max_dur_s: float) -> list[Path]:
    files = [(probe_duration_s(p), p) for p in iter_audio_files(audio_root)]
    files = [(d, p) for d, p in files if 60.0 <= d <= max_dur_s]
    rng = random.Random(seed)
    buckets = {"short": (60, 300), "mid": (300, 900), "long": (900, max_dur_s)}
    picked = []
    per = max(1, n_files // len(buckets))
    for (lo, hi) in buckets.values():
        pool = sorted(p for d, p in files if lo <= d < hi)
        if pool:
            picked += rng.sample(pool, min(per, len(pool)))
    rest = [p for _, p in files if p not in picked]
    rng.shuffle(rest)
    picked += rest[: max(0, n_files - len(picked))]
    return picked[:n_files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    ap.add_argument("--baselines", nargs="+", type=Path, default=[
        BASELINE_DIR / "transcripts_qwen3asr.jsonl",
        BASELINE_DIR / "transcripts.jsonl",
    ])
    ap.add_argument("--baseline-names", nargs="+", default=["qwen3asr", "whisper"])
    ap.add_argument("--n-files", type=int, default=9)
    ap.add_argument("--window-s", type=float, default=300.0)
    ap.add_argument("--max-dur-s", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=6144)
    ap.add_argument("--retries", type=int, default=1,
                    help="retry a truncated generation with 2x the token budget")
    ap.add_argument("--out", type=Path, default=Path("out/bench_vibevoice.jsonl"))
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    span_maps = {name: clip_spans(p) for name, p in zip(args.baseline_names, args.baselines)}
    for name, path in zip(args.baseline_names, args.baselines):
        if not path.exists():
            print(f"missing baseline {name}: {path}")
            return
    print("baseline source files: " +
          ", ".join(f"{n}={len(m)}" for n, m in span_maps.items()), flush=True)

    done = done_keys(args.out)
    todo = [p for p in sample_files(args.audio_root, args.n_files, args.seed, args.max_dur_s)
            if source_key(p, args.audio_root) not in done]
    print(f"{len(todo)} files to run (skipped {len(done)} already done)", flush=True)
    if not todo:
        return

    probe = import_module("10_probe_model")
    processor, model = probe.load_model(args.model_id)

    with JsonlAppender(args.out) as sink:
        for path in todo:
            key = source_key(path, args.audio_root)
            duration = probe_duration_s(path)
            window = min(args.window_s, duration)
            audio = load_24k_mono(path)[: int(window * 24000)]
            t0 = time.time()
            raw, status, elapsed, raw_text = probe.transcribe(
                processor, model, audio, args.prompt, args.max_new_tokens, args.retries)
            segments = normalize_segments(raw, 0.0, window)
            stats = file_stats(segments, window)
            vv_text = transcript_of(segments)
            row = {
                "key": key,
                "duration_s": round(duration, 2),
                "window_s": round(window, 2),
                "elapsed_s": round(elapsed, 1),
                "realtime_factor": round(window / max(elapsed, 1e-6), 2),
                "load_s": round(time.time() - t0, 1),
                "status": status,
                "raw": raw_text if status in ("unparseable", "truncated") else "",
                "vibevoice": {
                    **vitals(vv_text, len(segments), stats["coverage"]),
                    "speakers": stats["n_speakers"],
                    "cut_rate": cut_rate([s["text"] for s in segments]),
                    "tail_gap_s": stats["tail_gap_s"],
                    "monotonic": stats["monotonic"],
                    "segments": segments,
                    "text": vv_text,
                },
                "baselines": {},
            }
            for name, spans in span_maps.items():
                got = spans.get(key)
                if not got:
                    continue
                text, n_clips, covered = baseline_window(got, 0.0, window)
                texts = [t for _, _, _, t in got if t.strip()]
                row["baselines"][name] = {
                    **vitals(text, n_clips, covered / max(window, 1e-6)),
                    "cut_rate": cut_rate(texts[:n_clips] if texts else []),
                    "agreement_with_vibevoice": round(
                        difflib.SequenceMatcher(None, text, vv_text).ratio(), 4),
                    "text": text,
                }
            sink.write(row)
            print(f"{key}: [{status}] {window / 60:.1f} min in {elapsed:.1f}s "
                  f"({window / max(elapsed, 1e-6):.1f}x rt) "
                  f"segs={len(segments)} spk={stats['n_speakers']} "
                  f"chars={len(vv_text)}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()