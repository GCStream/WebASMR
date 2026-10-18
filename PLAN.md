# WebASMR plan

Status as of 2026-10-17. This folder is the VibeVoice-ASR data-prep work; the
training pipeline and console live in `../LocalASMR/asmr-bilingual-webml` and are
not touched from here.

## 1. Evaluation (running)

- DONE: model bring-up. `microsoft/VibeVoice-ASR-HF`, bf16, `eager` attention,
  float32 feature output cast to bf16. 16.7 GB, fits beside a running LoRA job.
- DONE: `20_bench_vibevoice.py` reconstructs each baseline clip's absolute
  time span from its cumulative `duration_s` (sums match ffprobe to 0.04 s),
  so all three systems are compared on identical audio.
- DONE: parser hardened against the model's real failure modes
  (malformed JSON objects, unterminated arrays, degenerate repetition loops).
- DONE: 9-file benchmark, 5 min windows, 3 duration buckets (`docs/eval.md`).
- DONE: long-form validation, 59.7 min in one pass, 4.4x realtime (`docs/eval.md`).
- DONE: throughput, processes vs batching compared head to head on an idle GPU
  (`docs/concurrency.md`). Worker processes win: 2-3 workers at batch 1 give
  18.1 audio-h/h against 9.7 for 1 worker at batch 4. Batching pads every clip
  to the longest in the batch, a ~20% loss when the GPU is not already
  time-sliced; it pays 1.97x when the card is shared with another job.

## 2. Corpus pass (not started)

- TODO: `30_transcribe.py` over all 528 files / 131 h, longest window first
  with a 120 s sub-window fallback for files that loop or return nothing.
- TODO: measure clean throughput once the GPU is free, then decide chunk
  size. A 131 h pass is 10-30 GPU-hours depending on window size and
  contention, so this is a scheduling decision, not a detail.
- TODO: quarantine rules from the metrics (loop rate, vocalization share,
  coverage) so sound-only files are dropped rather than trained on.

## 3. Downstream

- DONE: `70_build_dataset.py` (CPU only). Exports segment-cut clips with
  collision-free names, LLM SFT rows per speaker turn, a labelled non-speech
  SFX bank, a vocalization bank and a sound-only quarantine list. Verified at
  scale: 3,571 clips / 8.12 h / 0 duplicate names / 0 dangling refs.
- TODO: re-run it over the completed corpus once transcription finishes
  (idempotent; clip names are deterministic).
- DONE: `80_export_hf.py` - HF parquet (utterances + transcripts), F5-TTS CSV,
  SFT JSONL, dataset card, manifest. Verified: 4,500 rows with embedded 24 kHz
  audio, 0 join failures against the clip index.
- DONE: `tools/audit_quality.py` - CPS, overlap, duplicate, language, and
  recall-vs-baseline audit. Result: 0 overlaps, median coverage 0.97, and the
  low-recall files are sound-only files the baseline hallucinated over.
- DONE: accuracy review console (`review/`), mounted at `/review/` behind Caddy
  without touching the existing console's routes. Single-clip listen + label,
  keyboard-driven, context audio with padding, random sampling, live accuracy
  estimate with a 95% Wilson interval, append-only labels, CSV export.
- TODO: review a 200-400 clip sample and record the resulting interval in
  `docs/eval.md`; that is the dataset's accuracy number.
- TODO before any public release: decide the audio licensing question and add
  NSFW gating.
- TODO: carry `flags.jsonl` QC decisions and `vad_ratios.jsonl` over to the new
  segment keys, or the existing review work is lost.
- TODO: fix the wav filename collision upstream (`docs/upstream_findings.md`).
  Unowned code; not touched from here.