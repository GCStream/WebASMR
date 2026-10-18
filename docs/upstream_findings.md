# Findings in the upstream pipeline

Observed while building the VibeVoice pass. Not fixed here — the code lives in
`../LocalASMR/asmr-bilingual-webml`, which another agent owns. Recorded so the
findings are not lost and so the fix can be applied deliberately.

## 1. Every wav collides with 200 others (data corruption)

`training/scripts/01_prep_dataset.py:186`

```python
name = f"clip_{row.get('clip_index', i):06d}.wav"
```

`clip_index` is an index **within a source file**, not a global index. The Hub
dataset has 578 source files with `clip_index` restarting at 1 in each, so
`clip_000001.wav` is written once per source file that has a first clip.

Measured on the current output:

    metadata.csv rows            6683
    distinct wav names in rows    186
    wav files actually on disk    205
    rows pointing at clip_000001  241
    rows pointing at clip_000002  220

So 97% of the written audio is overwritten, and every surviving row's `text_zh`
belongs to a different recording than the `wav` beside it. `03_train_f5tts.py`
consumes exactly `wav` + `text_zh`, so acoustic fine-tuning on this metadata
trains on shuffled audio-text pairs. The LLM side is unaffected: it reads
`data/instructions/train.jsonl`, which does not reference wavs.

Fix: derive the name from the source file as well, e.g.

```python
slug = hashlib.sha1(row["source_file"].encode()).hexdigest()[:8]
name = f"{slug}_{row['clip_index']:06d}.wav"
```

`clip_key()` in `common/keys.py` here already provides the collision-free form.

## 2. Fixed 30 s clip boundaries cut mid-word

Not a bug so much as a structural limit of the upstream dataset. Examples from
`transcripts_qwen3asr.jsonl`:

    #5  30.0s  你觉得身边的声音好听吗？他好听话，那我给你再仔细一点。
    #6  30.0s  啊！我把我的人生放放下来，这样啊，这样啊，我我为了不说我的，
                不，不可以，不可以这样。

Clip 6 starts mid-sentence and ends mid-phrase. Any training clip built from a
whole 30 s block therefore carries text that does not match its audio at the
edges. This is what motivates re-cutting on VibeVoice segment boundaries.

## 3. Both baseline passes loop on non-speech

Measured over all 19,803 clips:

| pass | rows with a repeated-run ratio > 0.6 | rows <= 2 CJK chars |
|------|--------------------------------------|--------------------|
| whisper-turbo | 2,209 (11.2%) | 1,587 (8.0%) |
| Qwen3-ASR-1.7B | 1,595 (8.1%) | 1,241 (6.3%) |

`sanitize_zh.compress_runs` cleans the text afterwards, but the information
that a stretch was sound-only is gone by then. VibeVoice tags it
(`[Human Sounds]`, `[Sigh]`, `[Environmental Sounds]`) instead, which is what
makes the non-speech clips usable as an SFX bank rather than discardable.

## 4. Case-sensitive globs silently drop 50 of 578 files

`data_raw/audio` holds 578 audio files: 528 `.mp3`, 42 `.m4a`, and 8 `.MP3`.
The 8 uppercase ones plus the 42 `.m4a` are invisible to `find -name "*.mp3"`
or `Path.rglob("*.mp3")`, which is exactly how the corpus was first counted
here (528) before the discrepancy with the upstream `source_file` list
(578) showed up.

Reconciled case-insensitively, the two sets match exactly: 578 upstream source
files, 578 local files, none missing, none local-only. A corpus pass must walk
with a case-insensitive suffix check or it will skip 8.6% of the data while
looking finished.

`iter_audio_files()` in `common/keys.py` does the case-insensitive walk, and
`tests/test_segments.py` covers it.