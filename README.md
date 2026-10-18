# WebASMR — VibeVoice-ASR data prep

Long-form transcription pass over the raw ASMR corpus with
[microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR), built to
replace the fixed 30 s clip transcription that the current dataset is built on.

## Why this exists

The existing pipeline transcribes `DownFlow/WebAMSR`, whose audio is pre-cut
into 30 s FLAC slices. Two passes exist for it (whisper-large-v3-turbo, then
Qwen3-ASR-1.7B), both writing `{key: "<source_file>#<clip_index>",
transcript, duration_s}`. That format has three structural limits:

- **Boundaries are arbitrary.** Clip 7 ends mid-word because the cut is at
  30.0 s, so a training clip can start with `不，不可以这样`.
- **No speaker identity.** Roleplay is two-person; both sides land in one
  transcript row, so neither the LLM SFT data nor F5-TTS speaker
  conditioning can tell them apart.
- **No alignment.** There is no mapping from a transcript back to a time span
  in the original file, so utterances cannot be re-cut on sentence
  boundaries.

The raw corpus in `/root/data_raw` has none of these limits: 528 files, 131 h,
median 9.6 min, p95 46 min. VibeVoice-ASR transcribes up to 60 minutes in one
pass and returns `{"Start", "End", "Speaker", "Content"}` per utterance, so the
same pass that produces the text also produces the boundaries and the speaker
turns.

## Model

Use `microsoft/VibeVoice-ASR-HF`, not `microsoft/VibeVoice-ASR`. Both are the
same weights; the first is the transformers-native layout. The plain repo
declares `architectures: ["VibeVoiceForASRTraining"]`, a class that only exists
in the VibeVoice GitHub package, which pins `transformers==4.57.6` and needs
flash-attn. The `-HF` repo declares `VibeVoiceAsrForConditionalGeneration`,
which the installed `transformers 5.20.0` already implements.

Two runtime requirements that are not obvious from the model card:

- `attn_implementation="eager"` is mandatory. The acoustic tokenizer encoder
  is conv-only and rejects sdpa.
- The feature extractor emits float32 and the weights are bf16, so
  `input_values` must be cast to the model dtype or the first conv1d raises
  a dtype mismatch.

`VibeVoice-ASR-HF` is 16.7 GB bf16, which fits the MI300X (205 GB) beside a
running LoRA job.

## Layout

    common/keys.py      source-file and clip-key conventions (matches upstream)
    common/audio.py     24 kHz mono decode, >60 min window splitting with overlap
    common/segments.py  timestamp normalization, dedup across windows, quality metrics
    common/generation.py degenerate-loop stopping, token budgets
    common/jsonlio.py   resumable append-only JSONL
    scripts/10_probe_model.py     transcribe one file, print segments and stats
    scripts/11_raw_output.py      raw generation, unparsed, for diagnosing failures
    scripts/20_bench_vibevoice.py A/B against the whisper / Qwen3-ASR baselines
    scripts/30_transcribe.py     corpus pass, longest window first with fallback
    scripts/40_concurrency_bench.py  find the worker count that maximizes throughput
    scripts/41_batch_bench.py        find the clips-per-generate() sweet spot
    scripts/50_run_corpus.py     launch the corpus pass as N concurrent shards
    scripts/60_merge_shards.py   merge shards and report coverage gaps
    scripts/70_build_dataset.py  build training artifacts from segments (CPU only)
    scripts/80_export_hf.py      export HF parquet + F5-TTS + SFT formats
    review/                      accuracy-labeling console (served at /review/)
    tools/summarize_bench.py     comparison table from a benchmark file
    tools/audit_quality.py       transcript quality audit (CPS, recall, leaks)
    docs/eval.md          A/B results against the baselines
    docs/concurrency.md   worker-count measurement and the bandwidth ceiling
    docs/upstream_findings.md  bugs found in the existing pipeline
    out/                  transcripts and benchmark rows (git ignored)

## Results (2026-10-17)

- **Dataset**: 19,803 clips, Qwen3-ASR transcripts, upstream `DownFlow/WebAMSR`
  schema -> `DownFlow/ASMR` (16.3 GB). Viewer disabled by HF: the `nsfw` tag
  disables it platform-wide.
- **F5-TTS**: 21 epochs / 91,750 updates on 33.9 h -> `DownFlow/ASMR-f5tts`,
  ONNX fp16/int4 in `DownFlow/ASMR-onnx`.
- **LLM**: Qwen3.5-0.8B LoRA, 10 epochs -> `DownFlow/ASMR-llm-lora`. ONNX export
  of this hybrid model is not implemented upstream (sibling TODO-23).
- **Few-step**: 16 steps at CFG 2 matches 32-step quality exactly (CER 0.0972,
  MFCC-DTW 2.68) at half the forwards. A 4-step reflow student improved 4:0 CER
  from 0.775 to 0.597 but is not yet shippable. Details in
  `docs/tts_few_step.md`.

## Usage

    python scripts/10_probe_model.py --list-shortest 5
    python scripts/10_probe_model.py --audio "/root/data_raw/audio/安柔/器材室.mp3"
    python scripts/10_probe_model.py --audio <file> --max-minutes 2 --prompt "ASMR 角色扮演"

    python scripts/20_bench_vibevoice.py --n-files 9 --window-s 300
    python tools/summarize_bench.py out/bench_vibevoice.jsonl --samples

    python scripts/50_run_corpus.py --workers 1            # full corpus, batch 4, resumable
    python scripts/50_run_corpus.py --workers 1 --dry-run
    python scripts/60_merge_shards.py --shard-dir out/corpus --out out/segments.jsonl

    python3 tests/test_segments.py && python3 tests/test_generation.py

## Training artifacts (CPU only)

`70_build_dataset.py` turns segments into the dataset the training side needs.
It does not touch the GPU, so it can run while transcription is paused or in
flight.

    python scripts/70_build_dataset.py --report-only     # counts, no writes
    python scripts/70_build_dataset.py --no-audio         # metadata only, fast
    python scripts/70_build_dataset.py --limit 6          # smoke test
    python scripts/70_build_dataset.py                    # cut all clips

Outputs in `out/dataset/`:

- `metadata.csv` — one row per utterance-sized clip: `wav, text, speaker,
  source_file, start_s, end_s, duration, emotion, n_speakers_in_file,
  repetition, vocalization, tags`
- `instructions.jsonl` — one LLM SFT row per segment, `{prompt, completion}`
  with the `[tone:X]` prefix
- `sfx_bank.jsonl` — non-speech stretches with their event labels
  (`Human Sounds`, `Silence`, `Music`, `Lyric`, `Sigh`, `Breathing`, `Sobbing`,
  `Crying`, ...), split into `event` (<=30 s) and `bed` (longer ambience)
- `vocalizations.jsonl` — short interjections too thin for text training but
  usable as vocalisation SFX
- `sound_only.jsonl` — files with no speech at all, quarantined from the corpus

Clip names are `sha1(source_file)[:10]_index_start.wav`. This matters: the
upstream pipeline used `clip_{clip_index}.wav` while `clip_index` restarts per
source file, which made 6,683 metadata rows point at 186 wav files and paired
every text with another recording's audio. `tests/test_dataset_export.py`
locks the fix in.

Clips are cut on segment boundaries, not fixed 30 s offsets, so an utterance
starts where the utterance starts.

## Quality gates and export

`80_export_hf.py` applies the gates that separate "a segment exists" from "a
segment is trainable", and counts every rejection:

| gate | reason |
|---|---|
| `drop_tag_only` | non-speech only (`[Human Sounds]`, `[Music]`, `[Sigh]`) - goes to the SFX bank instead |
| `drop_cps` | characters/second outside 1.5-9.0: a segment with one word in a minute has a stretched boundary |
| `drop_short_text` | under 6 characters |
| `drop_duration` | outside 1-20 s |
| `drop_duplicate_line` | the same line twice in one file |
| `drop_language` | not Chinese (kana/latin share) |
| `drop_repetition` / `drop_vocalization` | degenerate or moan-dominated text |

Outputs in `out/hf/`:

- `utterances/` - parquet, one row per clip, 24 kHz audio embedded in HF's
  `{bytes, path}` struct, columns mirroring `DownFlow/WebAMSR`
- `transcripts/` - parquet, one row per source file with segments as JSON
- `f5tts_metadata.csv` - `audio_file|text`
- `instructions.jsonl` - LLM SFT rows
- `README.md` - dataset card, including the content warning, the unresolved
  audio licensing, and the fact that no WER has been measured
- `manifest.json` - counts, gates applied, exact filter values

`datasets` cannot write the audio column on this host (its Audio feature needs
torchcodec, whose bundled libraries fail to load under ROCm), so the parquet is
written with pyarrow. Read it back with:

```python
ds = load_dataset("parquet", data_files="utterances/train-*.parquet", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=24000))
```

## Accuracy review console

Served at `/review/` alongside the existing transcripts console (Caddy strips the
prefix and proxies to `127.0.0.1:8141`).

    ./review/run.sh

Listen to a clip, read the transcript, press 1-4 to label accuracy; labels
auto-save and the queue auto-advances. `context ±1s` plays the clip with padding
so boundary errors are judgeable. Set `sample` to 200-400 to review a random
sample instead of all 4,595 clips — the side panel then reports the correct share
with a 95% Wilson interval. Labels land in `out/review/labels.jsonl`
(append-only) and export as CSV. See `review/README.md`.

## Output format

One row per source file, timestamps in absolute file time:

```json
{
  "key": "<performer>/<file>.mp3",
  "duration_s": 974.26,
  "stats": {"n_segments": 21, "n_speakers": 2, "coverage": 0.93, "...": "..."},
  "segments": [
    {"start": 0.0, "end": 11.5, "speaker": 0, "text": "弟弟，你回来啦。"}
  ]
}
```

`clip_key()` in `common/keys.py` re-attaches the upstream `<file>#<clip_index>`
form so these rows stay joinable against the existing baseline transcripts.