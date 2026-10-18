#!/usr/bin/env python3
"""Build training artifacts from transcribed segments. CPU only.

This is the step the segment data exists for, and it is deliberately kept off
the GPU so it can run while transcription is in progress or paused.

Artifacts:

  metadata.csv        one row per utterance-sized clip for the acoustic model:
                      wav, text, speaker, emotion, duration, plus provenance
  instructions.jsonl  one row per speaker turn for the LLM: prompt built from
                      scenario/tone, completion the line with its `[tone:X]` tag
  sfx_bank.jsonl      non-speech stretches with their event tags, which is the
                      sample bank the upstream plan wanted and previously threw
                      away. 37% of all segments are non-speech, and they are
                      labelled: Human Sounds, Silence, Music, Lyric, Sigh,
                      Breathing, Sobbing, Crying, Environmental Sounds
  vocalizations.jsonl short spoken interjections ("嗯", "不要") that are too
                      thin to train text on but are the raw material for the
                      vocalisation SFX the browser engine schedules
  sound_only.jsonl    files with no speech at all, so they can be quarantined
                      from the text corpus without a second transcription pass

Naming: the upstream pipeline writes `clip_{clip_index}.wav`, but
`clip_index` restarts at 1 for every source file, so 6,683 metadata rows shared
186 wav files and every text was paired with someone else's audio. Names here
are derived from a hash of the source file, so they cannot collide.

Clips are cut on segment boundaries rather than fixed 30 s offsets, which is the
other thing the segment data buys: an utterance starts where the utterance
starts. `--no-audio` writes metadata and instructions without touching the audio,
which is fast and is the right mode while the corpus pass is still running.

    python scripts/70_build_dataset.py --report-only
    python scripts/70_build_dataset.py --no-audio --out-dir out/dataset
    python scripts/70_build_dataset.py --min-chars 6 --max-clip-s 20
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import TARGET_SR, load_24k_mono  # noqa: E402
from common.jsonlio import read_jsonl, write_jsonl  # noqa: E402
from common.segments import repetition_ratio, vocalization_ratio  # noqa: E402

METADATA_COLUMNS = [
    "wav", "text", "speaker", "source_file", "segment_index", "start_s", "end_s", "duration",
    "emotion", "n_speakers_in_file", "repetition", "vocalization", "tags",
]

TONES = ("neutral", "soft", "calm", "gentle", "sweet", "teasing", "breathy",
         "whisper", "playful", "seductive", "innocent", "shy", "sad", "serious")

TAG_ONLY = re.compile(r"^\[[^\]]+\]$")
TAG_ANY = re.compile(r"\[([^\]]+)\]")


def clip_name(source_file: str, index: int, start: float) -> str:
    slug = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{index:05d}_{int(start):06d}.wav"


def scene_tags(source_file: str) -> list[str]:
    stem = Path(source_file).stem
    parts = [p for p in re.split(r"[\s_\-]+", stem) if p]
    return parts[:4]


def partition(segments, args) -> tuple[list, list, list]:
    """Split one file's segments into speech, non-speech and vocalizations.

    The model marks non-speech explicitly (`[Human Sounds]`, `[Music]`,
    `[Sigh]`, ...), so sound-only stretches are identifiable rather than
    needing to be inferred from an empty or looping transcript.
    """
    speech, nonspeech, vocal = [], [], []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        tags = TAG_ANY.findall(text)
        spoken = TAG_ANY.sub("", text).strip()
        dur = seg["end"] - seg["start"]
        common = {"source_file": seg["source_file"], "start": round(seg["start"], 2),
                  "end": round(seg["end"], 2), "duration": round(dur, 2),
                  "speaker": seg["speaker"], "tags": tags}
        if not spoken:
            kind = "bed" if dur > args.sfx_max_s else "event"
            nonspeech.append({**common, "kind": kind,
                              "label": "+".join(tags) if tags else "unlabelled"})
            continue
        if len(spoken) < args.min_chars:
            vocal.append({**common, "text": spoken})
            continue
        if dur < args.min_clip_s or dur > args.max_clip_s:
            continue
        if dur > 0 and not (args.min_cps <= len(spoken) / dur <= args.max_cps):
            continue
        if repetition_ratio(spoken) > args.max_repetition:
            continue
        if vocalization_ratio(spoken) > args.max_vocalization:
            continue
        speech.append({**seg, "text": spoken, "tags": tags, "duration": dur})
    return speech, nonspeech, vocal


def write_audio(wavs_dir: Path, source_file: str, audio_root: Path,
                segments, args) -> dict[int, str]:
    """Cut one wav per segment. Returns {segment index: filename}.

    The result is keyed by index rather than being a list, because a skipped
    clip must not shift the names of the clips after it: a positional list
    silently pairs every later row's text with the previous clip's audio, which
    is the same failure mode as the upstream `clip_{clip_index}.wav` collision.
    """
    wavs_dir.mkdir(parents=True, exist_ok=True)
    src = Path(audio_root) / source_file
    written: dict[int, str] = {}
    try:
        audio = load_24k_mono(src)
    except RuntimeError:
        return written
    for i, seg in enumerate(segments):
        lo = max(0, int(seg["start"] * TARGET_SR))
        hi = min(len(audio), int(seg["end"] * TARGET_SR))
        if hi - lo < int(0.2 * TARGET_SR):
            continue
        name = clip_name(source_file, i, seg["start"])
        sf.write(str(wavs_dir / name), audio[lo:hi], TARGET_SR, subtype="PCM_16")
        written[i] = name
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, default=Path("out/corpus"))
    ap.add_argument("--audio-root", type=Path, default=Path("/root/data_raw/audio"))
    ap.add_argument("--out-dir", type=Path, default=Path("out/dataset"))
    ap.add_argument("--no-audio", action="store_true",
                    help="write metadata and instructions without cutting wavs")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N usable files (audio cutting)")
    ap.add_argument("--min-chars", type=int, default=6)
    ap.add_argument("--min-clip-s", type=float, default=1.0)
    ap.add_argument("--max-clip-s", type=float, default=20.0)
    ap.add_argument("--max-repetition", type=float, default=0.6)
    ap.add_argument("--max-vocalization", type=float, default=0.75)
    ap.add_argument("--min-cps", type=float, default=1.5,
                    help="reject clips whose text/timing is implausible")
    ap.add_argument("--max-cps", type=float, default=9.0)
    ap.add_argument("--sfx-max-s", type=float, default=30.0,
                    help="longer non-speech stretches are filed as beds, not events")
    args = ap.parse_args()

    rows, seen = [], set()
    for path in sorted(args.shard_dir.glob("segments.shard*.jsonl")):
        for row in read_jsonl(path):
            key = row["key"]
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda r: r["key"])

    usable_files, dropped_segments = 0, 0
    meta_rows, sft_rows = [], []
    sfx_rows, vocal_rows, sound_only = [], [], []
    total_audio = 0.0
    processed = 0
    for row in rows:
        if args.limit and processed >= args.limit:
            break
        tagged = [{**seg, "source_file": row["key"], "segment_index": i}
                  for i, seg in enumerate(row.get("segments", []))]
        segments, nonspeech, vocal = partition(tagged, args)
        sfx_rows.extend(nonspeech)
        vocal_rows.extend(vocal)
        if not segments and not vocal and nonspeech:
            sound_only.append({"key": row["key"],
                               "duration_s": row.get("duration_s"),
                               "labels": sorted({n["label"] for n in nonspeech})})
        if not segments:
            continue
        usable_files += 1
        processed += 1
        dropped_segments += len(tagged) - len(segments) - len(nonspeech) - len(vocal)
        n_speakers = row.get("stats", {}).get("n_speakers", 0)
        tags = scene_tags(row["key"])
        tag_str = ", ".join(tags)
        total_audio += sum(s["duration"] for s in segments)

        names = {} if args.no_audio else write_audio(
            args.out_dir / "wavs", row["key"], args.audio_root, segments, args)

        for i, seg in enumerate(segments):
            if not args.no_audio and i not in names:
                continue
            wav = f"wavs/{names[i]}" if i in names else ""
            sfx = "".join(f"[{t}]" for t in seg["tags"])
            text = f"{sfx}{seg['text']}" if sfx else seg["text"]
            meta_rows.append({
                "wav": wav,
                "text": seg["text"],
                "speaker": f"{row['key']}#spk{seg['speaker']}",
                "source_file": row["key"],
                "segment_index": seg["segment_index"],
                "start_s": round(seg["start"], 2),
                "end_s": round(seg["end"], 2),
                "duration": round(seg["duration"], 2),
                "emotion": "neutral",
                "n_speakers_in_file": n_speakers,
                "repetition": round(repetition_ratio(seg["text"]), 4),
                "vocalization": round(vocalization_ratio(seg["text"]), 4),
                "tags": ", ".join(seg["tags"]),
            })
            prompt = (f"Scenario: {tags[0] if tags else ''}\n"
                      f"Tone: neutral\n"
                      f"Tags: {tag_str}\n"
                      "Write one soft ASMR line in Chinese:")
            sft_rows.append({"prompt": prompt,
                             "completion": f"[tone:neutral] {seg['text']}"})

    names = [m["wav"] for m in meta_rows if m["wav"]]
    dupes = len(names) - len(set(names))
    speakers = {m["speaker"] for m in meta_rows}
    multi = sum(1 for m in meta_rows if m["n_speakers_in_file"] > 1)
    from_sfx = sum(1 for m in meta_rows if m["tags"])

    print(f"source files        {len(rows)}")
    print(f"files with segments {usable_files}")
    print(f"segments exported   {len(meta_rows)} "
          f"(dropped {dropped_segments} by quality gates)")
    print(f"distinct speakers   {len(speakers)}")
    print(f"clips from a 2+ spk file {multi} "
          f"({100 * multi / max(len(meta_rows), 1):.0f}%)")
    print(f"segments with sfx/non-speech tags {from_sfx}")
    print(f"audio in exported clips {total_audio / 3600:.2f} h")
    print(f"wav names written   {len(names)} (duplicate names: {dupes})")
    events = sum(1 for r in sfx_rows if r["kind"] == "event")
    print(f"\nnon-speech bank    {len(sfx_rows)} stretches "
          f"({events} events, {len(sfx_rows) - events} beds)")
    labels = {}
    for r in sfx_rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    for label, n in sorted(labels.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {label:24s} {n:5d}")
    print(f"vocalization bank   {len(vocal_rows)} short interjections")
    print(f"sound-only files    {len(sound_only)}")
    if args.report_only:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out_dir / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(meta_rows)
    write_jsonl(args.out_dir / "instructions.jsonl", sft_rows)
    write_jsonl(args.out_dir / "sfx_bank.jsonl", sfx_rows)
    write_jsonl(args.out_dir / "vocalizations.jsonl", vocal_rows)
    write_jsonl(args.out_dir / "sound_only.jsonl", sound_only)
    print(f"\nwrote {meta_path}, instructions.jsonl, sfx_bank.jsonl, "
          f"vocalizations.jsonl, sound_only.jsonl in {args.out_dir}")


if __name__ == "__main__":
    main()