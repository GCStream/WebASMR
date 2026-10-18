#!/usr/bin/env python3
"""Assemble the publishable dataset and optionally push it to the Hub.

Builds four configs so the dataset viewer can show each without ambiguity:

  utterances     clips with embedded 24 kHz audio (the training set)
  transcripts    file-level segment lists as JSON, no audio
  events         labelled non-speech ranges (the SFX bank), no audio
  manual_labels  the human accuracy labels, so the accuracy claim is auditable

The card declares `dataset_info` explicitly. The parquet stores audio in HF's
`{bytes, path}` struct layout, which the viewer maps to Audio, but the sampling
rate is not in the parquet schema, so it has to be declared or the viewer will
guess wrong.

    python scripts/90_publish.py --out-dir out/hf_publish
    python scripts/90_publish.py --push --repo-id DownFlow/ASMR --private
"""

import argparse
import collections
import csv
import json
import shutil
import sys
from pathlib import Path
from string import Template

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonlio import read_jsonl  # noqa: E402

CARD = """---
license: other
language:
- zh
task_categories:
- automatic-speech-recognition
- text-to-speech
- audio-classification
tags:
- asmr
- chinese
- roleplay
- timestamps
- diarization
- nsfw
pretty_name: WebASMR segment transcripts and clips
size_categories:
- 10K<n<100K
extra_gated_prompt: >-
  This dataset contains adult (NSFW) audio and explicit language. Access is
  limited to adults who agree not to attempt to identify or contact any speaker.
extra_gated_fields:
  I am 18 years or older: checkbox
  I will not attempt to identify or contact any speaker: checkbox
  Country: country
configs:
- config_name: utterances
  data_files:
  - split: train
    path: utterances/train-*.parquet
  default: true
- config_name: transcripts
  data_files:
  - split: train
    path: transcripts/train-*.parquet
- config_name: events
  data_files:
  - split: train
    path: events/train-*.parquet
- config_name: manual_labels
  data_files:
  - split: train
    path: manual_labels/train-*.parquet
dataset_info:
- config_name: utterances
  features:
  - name: audio
    dtype:
      audio:
        sampling_rate: 24000
  - {name: transcript, dtype: string}
  - {name: source_file, dtype: string}
  - {name: segment_index, dtype: int32}
  - {name: start_s, dtype: float32}
  - {name: end_s, dtype: float32}
  - {name: duration_s, dtype: float32}
  - {name: speaker, dtype: string}
  - {name: n_speakers_in_file, dtype: int32}
  - {name: tags, dtype: string}
  - {name: language, dtype: string}
  - {name: nsfw, dtype: bool}
  - {name: repetition, dtype: float32}
  - {name: vocalization, dtype: float32}
  - {name: chars_per_second, dtype: float32}
  splits:
  - name: train
    num_examples: $n_utt
- config_name: transcripts
  features:
  - {name: source_file, dtype: string}
  - {name: duration_s, dtype: float32}
  - {name: n_segments, dtype: int32}
  - {name: n_speakers, dtype: int32}
  - {name: coverage, dtype: float32}
  - {name: transcript, dtype: string}
  - {name: segments_json, dtype: string}
  splits:
  - name: train
    num_examples: $n_files
- config_name: events
  features:
  - {name: source_file, dtype: string}
  - {name: start_s, dtype: float32}
  - {name: end_s, dtype: float32}
  - {name: duration_s, dtype: float32}
  - {name: kind, dtype: string}
  - {name: label, dtype: string}
  - {name: tags, dtype: string}
  splits:
  - name: train
    num_examples: $n_events
- config_name: manual_labels
  features:
  - {name: id, dtype: string}
  - {name: source_file, dtype: string}
  - {name: start_s, dtype: float32}
  - {name: end_s, dtype: float32}
  - {name: duration_s, dtype: float32}
  - {name: text, dtype: string}
  - {name: accuracy, dtype: string}
  - {name: errors, dtype: string}
  - {name: corrected, dtype: string}
  - {name: notes, dtype: string}
  splits:
  - name: train
    num_examples: $n_labels
---

# WebASMR: segment transcripts and clips

**Content warning: adult audio and explicit language.** This corpus is NSFW
roleplay. Age-restrict and gate any deployment.

## What this is

Speech clips cut on utterance boundaries from $n_files ASMR recordings, with
transcripts produced by `microsoft/VibeVoice-ASR` and reconciled against a
second ASR pass and $n_labels human accuracy labels.

| | |
|---|---|
| source recordings | $n_files |
| audio transcribed | $src_h h |
| clips (`utterances`) | $n_utt |
| clip audio | $clip_h h |
| clip text changed by reconciliation | $changed ($changed_pct%) |
| language | zh ($n_utt rows), one mixed row |

Clip length: median $med_s s, p10 $p10_s s, p90 $p90_s s, all inside the
0.3-30 s window F5-TTS trains on. 24 kHz mono PCM.

## How the text was produced

1. `VibeVoice-ASR` transcribed each full recording in a single pass of up to
   60 minutes, emitting `Start`, `End`, `Speaker` and `Content` per utterance.
   Those segments define the clip boundaries and speaker turns.
2. Each segment was compared with the Qwen3-ASR 30-second clip covering the same
   span. Agreement (at least 80% aligned characters after removing punctuation)
   keeps the VibeVoice text; disagreement takes the Qwen3-ASR text for that span
   if it passes quality gates. $n_qwen segments ($qwen_pct%) were replaced.
3. Human labels override both: $n_manual clips reviewed, $n_corrected
   corrections substituted, $n_unusable unusable clips removed.
4. Non-speech stretches are excluded from `utterances` and kept in `events`
   with the model's labels (`Human Sounds`, `Silence`, `Music`, `Sigh`,
   `Breathing`, `Sobbing`, `Crying`, ...).

## Measured quality

$accuracy_blurb

For comparison, the upstream `DownFlow/WebAMSR` transcripts carry about 114
characters per audio-hour with a median of one character per clip; this dataset
carries about 5,600 per hour.

## Configs

- **utterances** - one row per clip. `audio` is the clip, `transcript` the
  reconciled text, `start_s`/`end_s` absolute offsets into `source_file`.
  `speaker` is a per-file index, not an identity. `nsfw` is a lexicon
  heuristic, not a classifier.
- **transcripts** - one row per source file with the full transcript and the
  model segment list as JSON.
- **events** - $n_events labelled non-speech ranges. `kind=event` are short
  (<= 30 s) effect candidates, `kind=bed` longer ambience stretches. Ranges and
  labels only, no audio.
- **manual_labels** - the human review sheet behind the accuracy figure.

## Loading

```python
from datasets import load_dataset, Audio

ds = load_dataset("DownFlow/ASMR", "utterances", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=24000))

meta = load_dataset("DownFlow/ASMR", "transcripts", split="train")
events = load_dataset("DownFlow/ASMR", "events", split="train")
```

## Provenance and licensing

The recordings were not produced for this dataset and are **not ours to
relicense**. They were assembled from third-party ASMR uploads; the upstream
mirror labelled itself CC-BY-4.0, which this project cannot verify and which is
probably not the uploaders' to grant. Treat the audio rights as unresolved. The
transcriptions were generated here with MIT-licensed VibeVoice-ASR and
Qwen3-ASR.

## Limitations

- **No WER.** Beyond the $n_labels human labels there is no ground truth, so
  accuracy is estimated, not measured corpus-wide, and the labelled sample
  over-represents shorter recordings because transcription ran shortest-first.
- **Boundaries are the model's.** A segment occasionally swallows a pause; the
  characters-per-second gate removes the worst cases but not all.
- **Speaker ids do not identify people across files.** $single_spk_pct% of
  files resolve to a single speaker; performers often voice both roleplay sides.
- **Generated text can be wrong.** Do not treat any row as verbatim ground
  truth, especially over moaning or screaming.
- **Not suitable for voice cloning.** Per-speaker volume is thin (median 4.6
  minutes per performer).

## Ethics

Adult content made by consenting performers for adult audiences. Do not attempt
to identify speakers and do not use the corpus for impersonation or harassment.
"""


def write_parquet(rows, columns, types, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [pa.array([r.get(c) for r in rows], type=types[c]) for c in columns]
    pq.write_table(pa.Table.from_arrays(arrays, names=columns), str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=Path("out/hf_final"))
    ap.add_argument("--dataset-dir", type=Path, default=Path("out/dataset"))
    ap.add_argument("--out-dir", type=Path, default=Path("out/hf_publish"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default="DownFlow/ASMR")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    args = ap.parse_args()

    out = args.out_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for config in ("utterances", "transcripts"):
        shutil.copytree(args.export_dir / config, out / config)
        print(f"{config}: copied")

    events = []
    for row in read_jsonl(args.dataset_dir / "sfx_bank.jsonl"):
        events.append({
            "source_file": row["source_file"], "start_s": float(row["start"]),
            "end_s": float(row["end"]), "duration_s": float(row["duration"]),
            "kind": row.get("kind", ""), "label": row.get("label", ""),
            "tags": ", ".join(row.get("tags") or []),
        })
    write_parquet(events, ["source_file", "start_s", "end_s", "duration_s",
                           "kind", "label", "tags"],
                  {"source_file": pa.string(), "start_s": pa.float32(),
                   "end_s": pa.float32(), "duration_s": pa.float32(),
                   "kind": pa.string(), "label": pa.string(), "tags": pa.string()},
                  out / "events" / "train-00000-of-00001.parquet")
    print(f"events: {len(events)} rows")

    committed = {}
    with open(args.dataset_dir / "metadata_committed.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            committed[f"{row['source_file']}#{row['segment_index']}"] = row
    labels = {}
    for row in read_jsonl("out/review/labels.jsonl"):
        if row.get("id"):
            labels[row["id"]] = row
    manual = []
    for key, label in labels.items():
        src = committed.get(key, {})
        manual.append({
            "id": key, "source_file": src.get("source_file", key.rsplit("#", 1)[0]),
            "start_s": float(src.get("start_s") or 0), "end_s": float(src.get("end_s") or 0),
            "duration_s": float(src.get("duration") or 0),
            "text": src.get("text", ""), "accuracy": label.get("accuracy", ""),
            "errors": ";".join(label.get("errors") or []),
            "corrected": label.get("corrected", ""), "notes": label.get("notes", ""),
        })
    write_parquet(manual, ["id", "source_file", "start_s", "end_s", "duration_s",
                           "text", "accuracy", "errors", "corrected", "notes"],
                  {"id": pa.string(), "source_file": pa.string(),
                   "start_s": pa.float32(), "end_s": pa.float32(),
                   "duration_s": pa.float32(), "text": pa.string(),
                   "accuracy": pa.string(), "errors": pa.string(),
                   "corrected": pa.string(), "notes": pa.string()},
                  out / "manual_labels" / "train-00000-of-00001.parquet")
    print(f"manual_labels: {len(manual)} rows")

    manifest = json.loads((args.export_dir / "manifest.json").read_text())
    report = json.loads((args.dataset_dir / "reconcile_report.json").read_text())
    utt_durations = sorted(float(r["duration"]) for r in committed.values())
    counts = collections.Counter(r["accuracy"] for r in manual)
    n_labels = len(manual)
    correct = counts.get("correct", 0)
    n_corrected = sum(1 for r in manual if r["corrected"])
    n_unusable = counts.get("unusable", 0)
    p = correct / n_labels if n_labels else 0.0
    if n_labels:
        z = 1.96
        den = 1 + z * z / n_labels
        centre = (p + z * z / (2 * n_labels)) / den
        half = z * ((p * (1 - p) / n_labels + z * z / (4 * n_labels ** 2)) ** 0.5) / den
        ci = (max(0.0, centre - half), min(1.0, centre + half))
    else:
        ci = (0.0, 0.0)
    acceptable = (correct + counts.get("minor", 0)) / max(n_labels, 1)
    accuracy_blurb = (
        f"On {n_labels} human-labeled clips: **{100 * p:.1f}% correct** "
        f"(95% Wilson CI {100 * ci[0]:.1f}-{100 * ci[1]:.1f}%), "
        f"{100 * counts.get('minor', 0) / max(n_labels, 1):.1f}% minor, "
        f"{100 * counts.get('major', 0) / max(n_labels, 1):.1f}% major, "
        f"{100 * counts.get('unusable', 0) / max(n_labels, 1):.1f}% unusable. "
        f"Acceptable (correct + minor) is {100 * acceptable:.1f}%."
    )

    file_speakers = {}
    with open(args.dataset_dir / "metadata.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            file_speakers[row["source_file"]] = int(row.get("n_speakers_in_file") or 1)
    single_spk = sum(1 for v in file_speakers.values() if v <= 1)
    single_pct = 100 * single_spk / max(len(file_speakers), 1)

    changed = report.get("changed_vs_vibevoice", 0)
    kept = manifest["segments_kept"]
    values = {
        "n_files": manifest["source_files"],
        "src_h": manifest["audio_hours"],
        "n_utt": kept,
        "clip_h": manifest["clip_hours"],
        "changed": changed,
        "changed_pct": 100 * changed / max(kept, 1),
        "n_qwen": report["stats"].get("source:qwen3asr", 0),
        "qwen_pct": 100 * report["stats"].get("source:qwen3asr", 0) / max(kept, 1),
        "n_manual": n_labels,
        "n_corrected": n_corrected,
        "n_unusable": n_unusable,
        "accuracy_blurb": accuracy_blurb,
        "n_events": len(events),
        "n_labels": n_labels,
        "single_spk_pct": single_pct,
        "med_s": utt_durations[len(utt_durations) // 2] if utt_durations else 0.0,
        "p10_s": utt_durations[int(len(utt_durations) * 0.1)] if utt_durations else 0.0,
        "p90_s": utt_durations[int(len(utt_durations) * 0.9)] if utt_durations else 0.0,
    }
    card = Template(CARD).substitute(values)
    (out / "README.md").write_text(card, encoding="utf-8")
    (out / "PUBLISH_MANIFEST.json").write_text(json.dumps({
        "configs": {"utterances": manifest["segments_kept"],
                    "transcripts": manifest["source_files"],
                    "events": len(events), "manual_labels": len(manual)},
        "audio_hours": manifest["clip_hours"],
        "accuracy": {"labeled": n_labels, "correct": correct,
                     "ci95": [round(ci[0], 4), round(ci[1], 4)]},
    }, indent=1), encoding="utf-8")
    print(f"card + manifest written to {out}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=args.private,
                        exist_ok=True)
        print(f"pushing {out} -> {args.repo_id} (private={args.private})")
        api.upload_folder(folder_path=str(out), repo_id=args.repo_id,
                          repo_type="dataset", commit_message="WebASMR segments + transcripts")
        print("done")


if __name__ == "__main__":
    main()