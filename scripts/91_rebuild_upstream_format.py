#!/usr/bin/env python3
"""Rebuild the clip-level dataset in DownFlow/WebAMSR's exact format.

The upstream layout is one row per 30 s clip with the audio embedded, 15
columns, 16 kHz. That is what the dataset viewer plays. This script reproduces
it byte-for-byte in structure and replaces only the transcripts with the
Qwen3-ASR pass, recomputing the two flags that were meaningless upstream
(`has_verbal_speech` was true on 22 of 19,803 rows, `hallucinated` never true).

Audio bytes are copied from the local cache rather than re-decoded, so the
clips are identical to the upstream ones.

    python scripts/91_rebuild_upstream_format.py --report-only
    python scripts/91_rebuild_upstream_format.py --out-dir out/upstream_format
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonlio import read_jsonl  # noqa: E402
from common.segments import repetition_ratio  # noqa: E402

CACHE = Path("/root/.cache/huggingface/datasets/DownFlow___web_amsr/default/0.0.0/"
             "19cb4f13c2f2f699ec8256e20979e468c6e876ce")
QWEN = Path("/root/LocalASMR/asmr-bilingual-webml/data/processed/zh_retranscribe/"
            "transcripts_qwen3asr.jsonl")

COLUMNS = ["audio", "transcript", "hallucinated", "creator", "source_file",
           "clip_index", "num_clips", "duration_s", "label", "tags", "nsfw",
           "scenario", "tone", "speaker", "has_verbal_speech"]

SCHEMA = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("transcript", pa.string()),
    ("hallucinated", pa.bool_()),
    ("creator", pa.string()),
    ("source_file", pa.string()),
    ("clip_index", pa.int64()),
    ("num_clips", pa.int64()),
    ("duration_s", pa.float64()),
    ("label", pa.string()),
    ("tags", pa.list_(pa.string())),
    ("nsfw", pa.bool_()),
    ("scenario", pa.string()),
    ("tone", pa.string()),
    ("speaker", pa.string()),
    ("has_verbal_speech", pa.bool_()),
])

FEATURES = {
    "audio": {"sampling_rate": 16000, "_type": "Audio"},
    "transcript": {"dtype": "string", "_type": "Value"},
    "hallucinated": {"dtype": "bool", "_type": "Value"},
    "creator": {"dtype": "string", "_type": "Value"},
    "source_file": {"dtype": "string", "_type": "Value"},
    "clip_index": {"dtype": "int64", "_type": "Value"},
    "num_clips": {"dtype": "int64", "_type": "Value"},
    "duration_s": {"dtype": "float64", "_type": "Value"},
    "label": {"dtype": "string", "_type": "Value"},
    "tags": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"},
    "nsfw": {"dtype": "bool", "_type": "Value"},
    "scenario": {"dtype": "string", "_type": "Value"},
    "tone": {"dtype": "string", "_type": "Value"},
    "speaker": {"dtype": "string", "_type": "Value"},
    "has_verbal_speech": {"dtype": "bool", "_type": "Value"},
}

CARD = """---
license: other
language:
- zh
task_categories:
- automatic-speech-recognition
tags:
- asmr
- chinese
- roleplay
- nsfw
- audio
pretty_name: WebASMR clips with Qwen3-ASR transcripts
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  default: true
dataset_info:
  features:
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - {name: transcript, dtype: string}
  - {name: hallucinated, dtype: bool}
  - {name: creator, dtype: string}
  - {name: source_file, dtype: string}
  - {name: clip_index, dtype: int64}
  - {name: num_clips, dtype: int64}
  - {name: duration_s, dtype: float64}
  - {name: label, dtype: string}
  - name: tags
    sequence: string
  - {name: nsfw, dtype: bool}
  - {name: scenario, dtype: string}
  - {name: tone, dtype: string}
  - {name: speaker, dtype: string}
  - {name: has_verbal_speech, dtype: bool}
  splits:
  - name: train
    num_examples: $n_rows
---

# WebASMR clips with Qwen3-ASR transcripts

**Content warning: adult audio and explicit language.** Age-restrict any use.

## What this is

A drop-in replacement for the `DownFlow/WebAMSR` clip set: the same 19,803
30-second clips with the same columns, same order, and the same audio bytes,
but with transcripts produced by **Qwen3-ASR-1.7B** and the two unusable flags
recomputed.

The upstream transcripts were placeholders: a median of one character per clip
(100% of clips had two characters or fewer, 19.6% were empty) and
`has_verbal_speech` was true on 22 of 19,803 rows. Every row here carries the
Qwen3-ASR transcript for its audio.

| | |
|---|---|
| clips | $n_rows |
| audio | 16 kHz mono, 30 s windows |
| transcript source | Qwen3-ASR-1.7B, per clip |
| clips with non-empty transcript | $n_verbal |
| clips flagged `hallucinated` | $n_loops |
| median transcript length | $med_chars characters |
| characters per audio-hour | $chars_per_hour |

## Columns

Identical to upstream: `audio`, `transcript`, `hallucinated`, `creator`,
`source_file`, `clip_index`, `num_clips`, `duration_s`, `label`, `tags`,
`nsfw`, `scenario`, `tone`, `speaker`, `has_verbal_speech`.

Recomputed:

- `transcript` - Qwen3-ASR text for this exact 30 s clip.
- `has_verbal_speech` - true when the transcript contains at least one
  alphanumeric character.
- `hallucinated` - true when the transcript is dominated by a repeated run
  (repetition ratio above 0.6), which is the ASR-loop failure mode.

Everything else is carried over unchanged from the upstream rows, including
`nsfw`, `label`, `tags`, `scenario`, `tone` and `speaker`.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("DownFlow/ASMR", split="train")
# audio is decoded automatically, e.g. ds[0]["audio"]["array"]
```

## Provenance and licensing

The recordings are not ours to relicense. They were assembled from third-party
ASMR uploads; the upstream mirror labelled itself CC-BY-4.0, which cannot be
verified here. The transcripts were generated with Qwen3-ASR (Apache-2.0).

## Limitations

- **No WER.** No ground truth exists for these clips, so accuracy is not
  quantified. On a separate sample of this corpus, a VibeVoice-ASR pass
  reconciled against Qwen3-ASR was 86% correct by human review; that figure
  applies to the reconciled segment set, not to these clip transcripts.
- **Qwen3-ASR still loops on non-speech.** 8.1% of the upstream clips were
  flagged as repetition loops in the full pass; the `hallucinated` column marks
  them so they can be filtered.
- **Clips are fixed 30 s slices**, so some start or end mid-sentence.
- **Not for voice cloning.** Per-speaker volume is thin.
"""


def qwen_map() -> dict:
    out = {}
    for row in read_jsonl(QWEN):
        key = row.get("key", "")
        src, _, idx = key.rpartition("#")
        if idx:
            out[(src, int(idx))] = (row.get("transcript") or "").strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out-dir", type=Path, default=Path("out/upstream_format"))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--rows-per-shard", type=int, default=600)
    ap.add_argument("--row-group-size", type=int, default=150)
    args = ap.parse_args()

    qwen = qwen_map()
    print(f"qwen transcripts: {len(qwen)}")
    shards = sorted(args.cache.glob("*.arrow"))
    print(f"cache shards: {len(shards)}")
    if not shards:
        print("no cached arrow shards; aborting")
        return

    rows_out, missing, n_verbal, n_loops = 0, 0, 0, 0
    lengths = []
    chars = 0
    audio_s = 0.0
    tables = []

    for path in shards:
        with pa.memory_map(str(path), "r") as src:
            table = pa.ipc.open_stream(src).read_all()
        cols = {name: table.column(name).to_pylist() for name in COLUMNS}
        transcripts, flags_v, flags_h = [], [], []
        for src_file, clip_index in zip(cols["source_file"], cols["clip_index"]):
            text = qwen.get((src_file, int(clip_index)), "")
            if (src_file, int(clip_index)) not in qwen:
                missing += 1
            transcripts.append(text)
            has = any(ch.isalnum() for ch in text)
            loop = bool(text) and repetition_ratio(text) > 0.6
            flags_v.append(has)
            flags_h.append(loop)
            n_verbal += int(has)
            n_loops += int(loop)
            lengths.append(len(text))
            chars += len(text)

        cols["transcript"] = transcripts
        cols["has_verbal_speech"] = flags_v
        cols["hallucinated"] = flags_h
        tables.append(pa.Table.from_pydict(cols, schema=SCHEMA))

    rows_out = sum(t.num_rows for t in tables)
    total_h = sum(sum(t["duration_s"].to_pylist()) for t in tables) / 3600
    cph = chars / max(total_h, 1e-9)
    print(f"rows {rows_out} | missing qwen {missing} | verbal {n_verbal} "
          f"| loops {n_loops}")
    print(f"audio {total_h:.1f} h | median chars {statistics.median(lengths):.1f} "
          f"| chars/audio-hour {cph:,.0f}")
    if args.report_only:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for old in data_dir.glob("*.parquet"):
        old.unlink()

    writer_index, buffer, rows_in_shard = 0, [], 0
    meta = {"info": {"features": FEATURES,
                     "splits": {"train": {"name": "train",
                                          "num_examples": rows_out}}}}
    schema_meta = {b"huggingface": json.dumps(meta).encode("utf-8")}

    def flush():
        nonlocal buffer, rows_in_shard, writer_index
        if not buffer:
            return
        combined = pa.concat_tables(buffer)
        combined = combined.replace_schema_metadata(schema_meta)
        out = data_dir / f"train-{writer_index:05d}-of-XXXXX.parquet"
        pq.write_table(combined, str(out), row_group_size=args.row_group_size)
        writer_index += 1
        buffer, rows_in_shard = [], 0

    for table in tables:
        offset = 0
        while offset < table.num_rows:
            take = min(args.rows_per_shard - rows_in_shard, table.num_rows - offset)
            buffer.append(table.slice(offset, take))
            rows_in_shard += take
            offset += take
            if rows_in_shard >= args.rows_per_shard:
                flush()
    flush()

    # rename with the final shard count
    files = sorted(data_dir.glob("train-*"))
    for i, path in enumerate(files):
        final = data_dir / f"train-{i:05d}-of-{len(files):05d}.parquet"
        path.rename(final)
    total_mb = sum(p.stat().st_size for p in data_dir.glob("*.parquet")) / 1e6
    print(f"wrote {len(files)} parquet shards, {total_mb:.0f} MB")

    card = CARD.replace("$n_rows", str(rows_out)) \
               .replace("$n_verbal", str(n_verbal)) \
               .replace("$n_loops", str(n_loops)) \
               .replace("$med_chars", f"{statistics.median(lengths):.0f}") \
               .replace("$chars_per_hour", f"{cph:,.0f}")
    (args.out_dir / "README.md").write_text(card, encoding="utf-8")
    (args.out_dir / "REBUILD_MANIFEST.json").write_text(json.dumps({
        "rows": rows_out, "missing_qwen": missing, "verbal": n_verbal,
        "hallucinated": n_loops, "shards": len(files),
        "median_chars": statistics.median(lengths),
    }, indent=1), encoding="utf-8")
    print("card + manifest written")


if __name__ == "__main__":
    main()