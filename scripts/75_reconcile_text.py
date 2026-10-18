#!/usr/bin/env python3
"""Reconcile transcript text with the manual labels and the Qwen3-ASR pass.

Decision order, as specified:

  1. A manually labeled clip is ground truth. A non-empty correction replaces
     the text outright; "correct"/"minor" keep the VibeVoice text; "unusable"
     removes the clip from the speech set.
  2. Otherwise, compare the VibeVoice segment with the Qwen3-ASR text covering
     the same time window. If they agree, keep it. If they disagree, use the
     Qwen3-ASR text for that span.
  3. A Qwen3-ASR replacement still has to pass the same quality gates
     (characters/second, repetition, minimum length); if it does not, the
     VibeVoice text is kept and the row is marked, because a hallucinated loop
     from the fallback model is worse than a plausible VibeVoice line.

Agreement is measured by aligning the two strings character by character after
removing punctuation and whitespace. A matched-block coverage of at least
`--agree` means agreement. The Qwen3-ASR span for a segment is recovered from
the same alignment, so the replacement text is the part of the 30 s baseline
clip that actually corresponds to this segment, not the whole clip.

    python scripts/75_reconcile_text.py --report-only
    python scripts/75_reconcile_text.py --out-dir out/dataset
"""

import argparse
import collections
import csv
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import probe_duration_s  # noqa: E402
from common.jsonlio import read_jsonl, write_jsonl  # noqa: E402
from common.segments import repetition_ratio, vocalization_ratio  # noqa: E402

TAG_ANY = re.compile(r"\[([^\]]+)\]")
BASELINE = Path("/root/LocalASMR/asmr-bilingual-webml/data/processed/"
                "zh_retranscribe/transcripts_qwen3asr.jsonl")
LABELS = Path("out/review/labels.jsonl")
METADATA = Path("out/dataset/metadata.csv")
EDGE = " \t\r\n，。！？、；：,.!?;:…~\"'“”‘’()（）[]【】"

COLS = ["wav", "text", "text_vv", "text_qwen", "source", "manual_accuracy",
        "manual_corrected", "speaker", "source_file", "segment_index",
        "start_s", "end_s", "duration", "cps", "n_speakers_in_file", "tags",
        "repetition", "vocalization"]


def normalize_with_index(text: str) -> tuple[str, list[int]]:
    """Keep only alphanumeric characters, remembering where each came from."""
    chars, index = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.lower())
            index.append(i)
    return "".join(chars), index


def baseline_spans() -> dict[str, list[tuple[float, float, str]]]:
    rows = collections.defaultdict(list)
    for row in read_jsonl(BASELINE):
        src, _, idx = row.get("key", "").rpartition("#")
        if not idx:
            continue
        rows[src].append((int(idx), row.get("duration_s") or 0.0,
                          (row.get("transcript") or "").strip()))
    spans = {}
    for src, clips in rows.items():
        clips.sort()
        t, out = 0.0, []
        for _, dur, text in clips:
            out.append((t, t + dur, text))
            t += dur
        spans[src] = out
    return spans


def window_text(spans, start_s: float, end_s: float) -> str:
    parts = [text for lo, hi, text in spans
             if text and hi > start_s + 0.05 and lo < end_s - 0.05]
    return " ".join(parts)


def qwen_counterpart(vv_text: str, q_text: str) -> tuple[str, float]:
    """Return (Qwen3-ASR text aligned to this segment, agreement coverage)."""
    vv_norm, _ = normalize_with_index(vv_text)
    q_norm, q_map = normalize_with_index(q_text)
    if not vv_norm or not q_norm:
        return "", 0.0
    matcher = difflib.SequenceMatcher(None, vv_norm, q_norm, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return "", 0.0
    coverage = sum(b.size for b in blocks) / len(vv_norm)
    lo = min(b.b for b in blocks)
    hi = max(b.b + b.size for b in blocks)
    raw_lo = q_map[lo]
    raw_hi = q_map[min(hi - 1, len(q_map) - 1)] + 1
    return q_text[raw_lo:raw_hi].strip(EDGE), coverage


def gates_ok(text: str, duration: float, args) -> bool:
    if len(text) < args.min_chars:
        return False
    if duration <= 0:
        return False
    cps = len(text) / duration
    if not (args.min_cps <= cps <= args.max_cps):
        return False
    if repetition_ratio(text) > args.max_repetition:
        return False
    if vocalization_ratio(text) > args.max_vocalization:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=METADATA)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--out-dir", type=Path, default=Path("out/dataset"))
    ap.add_argument("--agree", type=float, default=0.80,
                    help="matched-character coverage treated as agreement")
    ap.add_argument("--min-chars", type=int, default=6)
    ap.add_argument("--min-cps", type=float, default=1.5)
    ap.add_argument("--max-cps", type=float, default=9.0)
    ap.add_argument("--max-repetition", type=float, default=0.6)
    ap.add_argument("--max-vocalization", type=float, default=0.75)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    spans = baseline_spans()
    labels = {row["id"]: row for row in read_jsonl(args.labels) if row.get("id")}
    meta = [row for row in csv.DictReader(open(args.metadata, encoding="utf-8"))
            if row.get("wav")]

    stats = collections.Counter()
    rows, sft = [], []
    for row in meta:
        key = f"{row['source_file']}#{row['segment_index']}"
        vv = row["text"].strip()
        duration = float(row["duration"])
        q_text = window_text(spans.get(row["source_file"], []),
                             float(row["start_s"]), float(row["end_s"]))
        q_part, coverage = qwen_counterpart(vv, q_text)
        label = labels.get(key)
        manual_accuracy = (label or {}).get("accuracy") or ""
        manual_corrected = ((label or {}).get("corrected") or "").strip()
        stats["segments"] += 1

        if label:
            stats["manually_labeled"] += 1
            if manual_accuracy == "unusable":
                stats["drop_manual_unusable"] += 1
                continue
            if manual_corrected:
                final, source = manual_corrected, "manual_correction"
            elif manual_accuracy in ("correct", "minor"):
                final, source = vv, "manual_verified"
            else:
                if q_part and gates_ok(q_part, duration, args):
                    final, source = q_part, "manual_major_qwen"
                else:
                    final, source = vv, "manual_major_vv"
        else:
            if coverage >= args.agree:
                final, source = vv, "agree"
            elif q_part and gates_ok(q_part, duration, args):
                final, source = q_part, "qwen3asr"
            elif q_part:
                final, source = vv, "qwen_rejected_gates"
            else:
                stats["drop_no_qwen"] += 1
                continue

        if source not in ("manual_correction", "manual_verified") and \
                not gates_ok(final, duration, args):
            stats["drop_final_gates"] += 1
            continue

        stats[f"source:{source}"] += 1
        cps = len(final) / duration if duration else 0.0
        rows.append({
            "wav": row["wav"], "text": final, "text_vv": vv,
            "text_qwen": q_part, "source": source,
            "manual_accuracy": manual_accuracy,
            "manual_corrected": manual_corrected,
            "speaker": row["speaker"], "source_file": row["source_file"],
            "segment_index": row["segment_index"], "start_s": row["start_s"],
            "end_s": row["end_s"], "duration": row["duration"],
            "cps": round(cps, 3),
            "n_speakers_in_file": row["n_speakers_in_file"], "tags": row["tags"],
            "repetition": row["repetition"], "vocalization": row["vocalization"],
        })
        tags = row.get("tags", "")
        sft.append({
            "prompt": (f"Scenario: {Path(row['source_file']).stem}\n"
                       f"Tone: neutral\nTags: {tags}\n"
                       "Write one soft ASMR line in Chinese:"),
            "completion": f"[tone:neutral] {final}",
        })

    print(f"segments in            {stats['segments']}")
    print(f"kept                   {len(rows)}")
    print(f"manually labeled       {stats['manually_labeled']}")
    for key in ("source:manual_correction", "source:manual_verified",
                "source:manual_major_qwen", "source:manual_major_vv",
                "source:agree", "source:qwen3asr", "source:qwen_rejected_gates"):
        print(f"  {key:28s} {stats[key]}")
    print("dropped:")
    for key in ("drop_manual_unusable", "drop_final_gates", "drop_no_qwen"):
        print(f"  {key:28s} {stats[key]}")
    changed = sum(1 for r in rows if r["text"] != r["text_vv"])
    print(f"text changed vs VibeVoice: {changed} "
          f"({100 * changed / max(len(rows), 1):.1f}%)")
    print(f"audio in kept clips    {sum(float(r['duration']) for r in rows) / 3600:.2f} h")
    if args.report_only:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "metadata_committed.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(args.out_dir / "instructions_committed.jsonl", sft)
    report = {"stats": dict(stats), "kept": len(rows),
              "changed_vs_vibevoice": changed, "agree_threshold": args.agree,
              "audio_hours": round(sum(float(r["duration"]) for r in rows) / 3600, 2)}
    (args.out_dir / "reconcile_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_csv}, instructions_committed.jsonl, reconcile_report.json")


if __name__ == "__main__":
    main()