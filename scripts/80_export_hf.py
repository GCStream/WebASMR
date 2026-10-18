#!/usr/bin/env python3
"""Export segments in finetuning and Hugging Face dataset layouts.

Three consumers, three shapes:

  out/hf/utterances/     HF dataset, segment-level clips with embedded audio.
                         Column names deliberately mirror DownFlow/WebAMSR
                         (audio, transcript, source_file, duration_s, speaker,
                         nsfw) so it can replace it without a schema rewrite.
  out/hf/transcripts/    HF dataset, file-level segments with timestamps and
                         speaker turns, no audio. Small; good for text work.
  out/hf/f5tts_metadata.csv  `audio_file|text`, the shape F5-TTS loaders expect.
  out/hf/instructions.jsonl  LLM SFT rows, {prompt, completion}.

Quality gates are applied and counted, because the difference between "a clip
exists" and "a clip is usable for training" is most of this file. The dataset
card records the gates and the known limitations, including that no WER has
been measured (there is no ground truth) and that speaker ids are per-file
indices, not identities.

    python scripts/80_export_hf.py --report-only
    python scripts/80_export_hf.py --out-dir out/hf
"""

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonlio import read_jsonl, write_jsonl  # noqa: E402
from common.segments import repetition_ratio, vocalization_ratio  # noqa: E402

TAG_ANY = re.compile(r"\[([^\]]+)\]")
CJK = re.compile(r"[\u4e00-\u9fff]")
KANA = re.compile(r"[\u3040-\u30ff]")
LATIN = re.compile(r"[A-Za-z]")

EXPLICIT_LEXICON = (
    "鸡巴", "肉棒", "龟头", "阴茎", "阴道", "淫水", "爱液", "射精", "高潮",
    "自慰", "口交", "插", "操你", "骚", "淫", "乳", "逼",
)

COLUMNS = ["audio", "transcript", "source_file", "segment_index", "start_s",
           "end_s", "duration_s", "speaker", "n_speakers_in_file", "tags",
           "language", "nsfw", "repetition", "vocalization", "chars_per_second"]


def detect_language(text: str) -> str:
    cjk = len(CJK.findall(text))
    kana = len(KANA.findall(text))
    latin = len(LATIN.findall(text))
    total = cjk + kana + latin
    if not total:
        return "unknown"
    if kana > 0.2 * total:
        return "ja"
    if latin > 0.5 * total:
        return "en"
    if cjk > 0.5 * total:
        return "zh"
    return "mixed"


def is_nsfw(text: str) -> bool:
    return any(word in text for word in EXPLICIT_LEXICON)


def gate(segments, source_file, args, stats, row_source=""):
    """Filter one file's segments into exportable speech clips.

    Manually verified or corrected rows bypass the text-shape gates: a human
    has already decided the line is right, and re-rejecting it on a
    characters-per-second heuristic would discard ground truth.
    """
    out, seen_text = [], set()
    for position, seg in enumerate(segments):
        index = seg.get("segment_index", position)
        stats["in"] += 1
        text = seg["text"].strip()
        tags = TAG_ANY.findall(text)
        spoken = TAG_ANY.sub("", text).strip()
        dur = seg["end"] - seg["start"]

        if not spoken:
            stats["drop_tag_only"] += 1
            continue
        if len(spoken) < args.min_chars:
            stats["drop_short_text"] += 1
            continue
        if dur < args.min_clip_s or dur > args.max_clip_s:
            stats["drop_duration"] += 1
            continue
        cps = len(spoken) / dur if dur else 0.0
        manual = str(seg.get("source", row_source)).startswith("manual_")
        if not manual and not (args.min_cps <= cps <= args.max_cps):
            stats["drop_cps"] += 1
            continue
        if not manual and repetition_ratio(spoken) > args.max_repetition:
            stats["drop_repetition"] += 1
            continue
        if not manual and vocalization_ratio(spoken) > args.max_vocalization:
            stats["drop_vocalization"] += 1
            continue
        if spoken in seen_text:
            stats["drop_duplicate_line"] += 1
            continue
        language = detect_language(spoken)
        if language not in args.languages:
            stats["drop_language"] += 1
            continue
        seen_text.add(spoken)
        stats["kept"] += 1
        out.append({
            "transcript": spoken,
            "source_file": source_file,
            "segment_index": index,
            "start_s": round(seg["start"], 2),
            "end_s": round(seg["end"], 2),
            "duration_s": round(dur, 2),
            "speaker": f"{source_file}#spk{seg['speaker']}",
            "n_speakers_in_file": stats.get("n_speakers", 0),
            "tags": ", ".join(tags),
            "language": language,
            "nsfw": is_nsfw(spoken),
            "repetition": round(repetition_ratio(spoken), 4),
            "vocalization": round(vocalization_ratio(spoken), 4),
            "chars_per_second": round(cps, 3),
        })
    return out


AUDIO_STRUCT = [("bytes", "binary"), ("path", "string")]

STRING_COLS = {"transcript", "source_file", "speaker", "tags", "language",
               "segments_json"}
FLOAT_COLS = {"start_s", "end_s", "duration_s", "repetition", "vocalization",
              "chars_per_second", "coverage"}
INT_COLS = {"segment_index", "n_segments", "n_speakers", "n_speakers_in_file"}


def _pyarrow_type(name):
    import pyarrow as pa

    if name in STRING_COLS:
        return pa.string()
    if name in FLOAT_COLS:
        return pa.float32()
    if name in INT_COLS:
        return pa.int32()
    if name == "nsfw":
        return pa.bool_()
    return pa.string()


def write_parquet(rows, columns, path: Path, audio: bool = False,
                  row_group_size: int = 500) -> None:
    """Write a parquet file with HF's `{bytes, path}` audio layout.

    `datasets` cannot write the audio column here: its Audio feature imports
    torchcodec to encode, and torchcodec's bundled libraries fail to load on this
    ROCm host. The `{bytes, path}` struct is exactly what the Audio feature
    serialises to, so the file reads back with
    `cast_column("audio", Audio(...))` — recorded in the dataset card.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    schema_fields = []
    if audio:
        schema_fields.append(("audio", pa.struct([(n, pa.binary() if n == "bytes"
                                                    else pa.string())
                                                   for n, _ in AUDIO_STRUCT])))
    schema_fields += [(c, _pyarrow_type(c)) for c in columns if c != "audio"]
    schema = pa.schema(schema_fields)

    arrays = []
    for field in schema:
        values = [r.get(field.name) for r in rows]
        if pa.types.is_struct(field.type):
            values = [{"bytes": v if isinstance(v, bytes) else None, "path": None}
                      for v in values]
        arrays.append(pa.array(values, type=field.type))
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), str(path),
                   row_group_size=row_group_size)


def write_parquet_sharded(rows, columns, out_dir: Path, audio: bool = False,
                          rows_per_shard: int = 2000, row_group_size: int = 500) -> None:
    """Shard output so no single binary column exceeds Arrow's 2 GiB offsets.

    A struct<bytes> column is an int32-offset binary array, so a 3.4 GB
    single-row-group file cannot be read back: pyarrow raises
    "Nested data conversions not implemented for chunked array outputs".
    Hugging Face datasets writes ~500 MB shards for the same reason, and the
    dataset viewer handles a glob of shards transparently.
    """
    import math

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("train-*.parquet"):
        old.unlink()
    if not rows:
        return
    shards = max(1, math.ceil(len(rows) / rows_per_shard))
    for i in range(shards):
        chunk = rows[i * rows_per_shard:(i + 1) * rows_per_shard]
        path = out_dir / f"train-{i:05d}-of-{shards:05d}.parquet"
        write_parquet(chunk, columns, path, audio=audio,
                      row_group_size=row_group_size)
    print(f"  {out_dir.name}: {len(rows)} rows in {shards} shard(s)")


def dataset_card(stats, args, total_h, clip_h, n_files) -> str:
    return f"""---
license: other
language:
- zh
- ja
- en
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- asmr
- chinese
- roleplay
- timestamps
- diarization
- nsfw
pretty_name: WebASMR VibeVoice transcripts
---

# WebASMR transcripts (VibeVoice-ASR)

Segment-level transcripts for {n_files} ASMR recordings ({total_h:.1f} audio-hours
transcribed, {clip_h:.1f} hours of exported speech clips) produced with
`microsoft/VibeVoice-ASR` in a single pass per file, up to 60 minutes of audio
per pass.

## Content warning

Adult audio. The corpus is NSFW roleplay and the transcriptions contain explicit
language. Gate and age-restrict any use.

## Licensing and provenance

The audio was not produced for this dataset and is not ours to relicense. It was
assembled from third-party recordings; the upstream mirror
`DownFlow/WebAMSR` labels itself CC-BY-4.0, which this project cannot verify.
Treat the audio rights as unresolved and do not redistribute without clearing
them.

The transcriptions themselves were generated here with MIT-licensed
VibeVoice-ASR.

## Loading

The audio column is stored in Hugging Face's `{{bytes, path}}` struct layout.
This environment cannot run torchcodec, so the file was written with pyarrow;
cast the column to get decoded audio:

```python
from datasets import load_dataset, Audio

ds = load_dataset("parquet", data_files="utterances/train-00000-of-00001.parquet",
                  split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
```

## Configs

- `utterances` - one row per utterance-sized clip with embedded 24 kHz audio,
  absolute `start_s`/`end_s` into the source file, `speaker` (a per-file index,
  not an identity), and event `tags`.
- `transcripts` - one row per source file with the full segment list as JSON and
  no audio.

## Quality

- No WER is reported: there is no ground truth. Accuracy was not benchmarked.
- Recall on speech-dense files matches the Qwen3-ASR pass (median character
  ratio 1.00); VibeVoice emits ~14x less runaway repetition on moaning audio.
- For comparison, the upstream `DownFlow/WebAMSR` transcripts carry 114
  characters per audio hour with a median of 1 character per clip; these carry
  roughly 5,600 per hour.
- {stats['drop_tag_only']} segments were non-speech only (`[Human Sounds]`,
  `[Music]`, `[Sigh]`, ...) and are excluded from `utterances`; they are kept in
  the SFX bank instead.
- {stats['drop_cps']} segments were rejected for an implausible
  characters-per-second rate, which is how stretched boundaries (one word in a
  minute-long span) are removed.
- {stats['drop_duplicate_line']} duplicate lines within the same file were
  dropped.

## Known limitations

- Speaker ids are per-file (`<file>#spk0`), so they do not identify a person
  across files. Most files are single-narrator roleplay, so most rows are spk0.
- Segmentation follows the model's segment boundaries, which occasionally
  swallow a pause; the CPS gate removes the worst of them but not all.
- Generated transcripts can contain errors, including invented text during
  non-speech. Do not treat any row as verbatim ground truth.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=Path("out/corpus"))
    ap.add_argument("--metadata", type=Path, default=Path("out/dataset/metadata.csv"),
                    help="authoritative clip index written by 70_build_dataset.py")
    ap.add_argument("--out-dir", type=Path, default=Path("out/hf"))
    ap.add_argument("--min-chars", type=int, default=6)
    ap.add_argument("--min-clip-s", type=float, default=1.0)
    ap.add_argument("--max-clip-s", type=float, default=20.0)
    ap.add_argument("--min-cps", type=float, default=1.5)
    ap.add_argument("--max-cps", type=float, default=9.0)
    ap.add_argument("--max-repetition", type=float, default=0.6)
    ap.add_argument("--max-vocalization", type=float, default=0.75)
    ap.add_argument("--languages", nargs="+", default=["zh", "mixed"])
    ap.add_argument("--committed", type=Path, default=None,
                    help="metadata_committed.csv from 75_reconcile_text.py; when set, "
                         "its rows (reconciled text + provenance) are the segment source")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    rows, seen = [], set()
    if args.committed:
        by_file = collections.defaultdict(list)
        with open(args.committed, encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                by_file[rec["source_file"]].append(rec)
        for src, recs in by_file.items():
            recs.sort(key=lambda r: int(r["segment_index"]))
            segs = [{
                "start": float(r["start_s"]), "end": float(r["end_s"]),
                "speaker": int(str(r["speaker"]).rsplit("spk", 1)[-1])
                           if "spk" in str(r["speaker"]) else 0,
                "text": r["text"], "segment_index": int(r["segment_index"]),
                "source": r.get("source", ""),
            } for r in recs]
            rows.append({
                "key": src,
                "duration_s": max(float(r["end_s"]) for r in recs),
                "segments": segs,
                "stats": {"n_speakers": int(recs[0].get("n_speakers_in_file") or 1),
                          "coverage": round(sum(s["end"] - s["start"] for s in segs)
                                            / max(float(recs[-1]["end_s"]), 1e-9), 4)},
            })
    else:
        for path in sorted(args.shard_dir.glob("segments.shard*.jsonl")):
            for row in read_jsonl(path):
                if row["key"] in seen:
                    continue
                seen.add(row["key"])
                rows.append(row)
    rows.sort(key=lambda r: r["key"])

    wav_index = {}
    meta_path = args.committed or args.metadata
    if meta_path and meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("wav"):
                    continue
                key = (row["source_file"], int(row["segment_index"]))
                wav_index[key] = str((meta_path.parent / row["wav"]).resolve())

    covered_files = {k[0] for k in wav_index}
    stats = collections.Counter()
    utterances, file_rows, f5_rows, sft_rows = [], [], [], []
    total_h = sum(r["duration_s"] for r in rows) / 3600
    clip_h = 0.0
    missing_audio = 0
    newer_files = 0

    for row in rows:
        stats["n_speakers"] = row.get("stats", {}).get("n_speakers", 0)
        segs = gate(row.get("segments", []), row["key"], args, stats,
                    row.get("source", ""))
        file_rows.append({
            "source_file": row["key"],
            "duration_s": row.get("duration_s"),
            "n_segments": len(row.get("segments", [])),
            "n_speakers": row.get("stats", {}).get("n_speakers", 0),
            "coverage": row.get("stats", {}).get("coverage"),
            "transcript": TAG_ANY.sub("", row.get("transcript", "")).strip(),
            "segments_json": json.dumps(row.get("segments", []), ensure_ascii=False),
        })
        for seg in segs:
            lookup = (seg["source_file"], seg["segment_index"])
            rel = wav_index.get(lookup, "")
            if not rel:
                if seg["source_file"] not in covered_files:
                    newer_files += 1
                else:
                    missing_audio += 1
            clip_h += seg["duration_s"] / 3600
            utterances.append({**seg, "_wav": rel})
            f5_rows.append({"audio_file": rel, "text": seg["transcript"]})
            sft_rows.append({
                "prompt": (f"Scenario: {Path(row['key']).stem}\n"
                           f"Tone: neutral\nTags: {seg['tags']}\n"
                           "Write one soft ASMR line in Chinese:"),
                "completion": f"[tone:neutral] {seg['transcript']}",
            })

    print(f"source files      {len(rows)}")
    print(f"audio transcribed {total_h:.1f} h")
    print(f"segments in       {stats['in']}")
    print(f"segments kept     {stats['kept']}  ({clip_h:.1f} h of clips)")
    for key in ("drop_tag_only", "drop_short_text", "drop_duration", "drop_cps",
                "drop_repetition", "drop_vocalization", "drop_duplicate_line",
                "drop_language"):
        print(f"  {key:24s} {stats[key]}")
    print(f"segments in files newer than the clip build: {newer_files}")
    print(f"segments failing the clip join: {missing_audio}")
    lang = collections.Counter(u["language"] for u in utterances)
    print(f"languages          {dict(lang)}")
    print(f"nsfw (heuristic)   {sum(1 for u in utterances if u['nsfw'])}")
    if args.report_only:
        return

    audio_rows = []
    for u in utterances:
        if not u["_wav"]:
            continue
        with open(u["_wav"], "rb") as f:
            raw = f.read()
        audio_rows.append({**{k: u[k] for k in COLUMNS if k != "audio"},
                           "audio": raw})

    write_parquet_sharded(audio_rows, COLUMNS, args.out_dir / "utterances", audio=True)
    write_parquet_sharded(file_rows,
                          ["source_file", "duration_s", "n_segments", "n_speakers",
                           "coverage", "transcript", "segments_json"],
                          args.out_dir / "transcripts")

    with open(args.out_dir / "f5tts_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_file", "text"], delimiter="|")
        writer.writeheader()
        writer.writerows([r for r in f5_rows if r["audio_file"]])
    write_jsonl(args.out_dir / "instructions.jsonl", sft_rows)
    (args.out_dir / "README.md").write_text(
        dataset_card(stats, args, total_h, clip_h, len(rows)), encoding="utf-8")
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "source_files": len(rows), "audio_hours": round(total_h, 2),
        "segments_in": stats["in"], "segments_kept": stats["kept"],
        "clip_hours": round(clip_h, 2), "gates": {k: v for k, v in stats.items()
                                                  if k.startswith("drop_")},
        "languages": dict(lang), "filters": vars(args) | {"languages": args.languages},
    }, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {args.out_dir}/utterances, /transcripts, f5tts_metadata.csv, "
          f"instructions.jsonl, README.md, manifest.json")


if __name__ == "__main__":
    main()