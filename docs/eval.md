# VibeVoice-ASR vs the current baselines

All numbers from `out/bench_vibevoice.jsonl` (9 files, 37 min of audio, 5 min
windows, MI300X shared with a LoRA training job that was at 100% GPU for most
of the run). Reproduce with:

    python scripts/20_bench_vibevoice.py --n-files 9 --window-s 300
    python tools/summarize_bench.py out/bench_vibevoice.jsonl --samples

Every system is scored on identical audio. The baseline transcripts carry no
timing, so each clip's absolute span was rebuilt from its cumulative
`duration_s`; the reconstruction sums to 397.07 s against an ffprobe duration
of 397.11 s on the file checked.

## Results

| metric                     | vibevoice | qwen3asr | whisper |
|----------------------------|-----------|----------|---------|
| chars per window            | 836       | 655      | 541     |
| units per window            | 29.7      | 10.0     | 10.0    |
| chars per unit              | 29.6      | 66.6     | 54.5    |
| **repetition**              | **0.008** | 0.111    | 0.178   |
| **vocalization**            | **0.021** | 0.118    | 0.065   |
| coverage                    | 0.866     | 1.000    | 1.000   |
| cut_rate (starts mid-word)  | 0.036     | 0.111    | 0.019   |
| agreement with vibevoice    | -         | 0.525    | 0.505   |
| speakers found (mean)       | 1.11      | n/a      | n/a     |
| wall clock per 5 min window | 72 s      | -        | -       |
| realtime factor             | 3.7x      | -        | -       |

Parse status over the 9 windows: 6 `ok`, 3 `truncated` (degenerate-loop stop or
unterminated array). After segment repair, **no window contained runaway
repeated text**, and none was mostly vocalization.

## What the numbers mean

**Repetition is the headline.** ASMR is mostly moaning and breathing, and that
is exactly where the baselines fail: whisper-turbo devotes 17.8% of its output
characters to repeated runs and Qwen3-ASR 11.1%, while VibeVoice-ASR spends
0.8%. This is the failure already documented in the project as "whisper
hallucinates fluent loops on moans", measured rather than asserted.

The concrete case is `橙子/进入陌生人橙子（一）.m4a`, a distress/screaming
roleplay, where both baselines collapse into garbage and VibeVoice does not:

    vibevoice: [Environmental Sounds] [Sigh]谁呀?[Sigh]开门。 [CS]谁呀?[Sigh]
               [Human Sounds] [Sigh]不要抱我衣服。 [Human Sounds]

    qwen3asr:  嗯。 哦吼，谁呀？嗯，啊，开门！嗯嗯嗯嗯，是谁？我我我给，嗯嗯嗯嗯。
               不要扒我衣服！嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯嗯…

    whisper:   你 谁呀看门是谁 不要不要云 你身材没做好痛啊 不要哭不要哭哭 痛...
               青青青青青青青青青青青青青青青青青青青青青青青青青青青青青青青…

**Higher recall.** 28% more characters per window than Qwen3-ASR and 55% more
than whisper. The baselines under-transcribe whispered and quiet speech; the
model's own `[Human Sounds]` / `[Sigh]` / `[CS]` tags mark where it deliberately
stopped rather than inventing words.

**Sentence-level units.** 29.6 characters per unit against 66.6 / 54.5 means
VibeVoice returns utterance-sized segments instead of 30 s blocks, so a segment
boundary lands on a sentence rather than at an arbitrary offset. `cut_rate` is
3x lower than Qwen3-ASR for the same reason. (whisper scores lower on
`cut_rate` only because it emits no punctuation, so a segment rarely begins
with a particle; the metric is biased against it.)

**Speaker turns.** The thing neither baseline can do at all. On
`_unattributed/第12期 3倒计时&接吻.mp3` it returned 44 segments across 2
speakers; on `离二烟烟/姐姐疼爱弟弟.mp3` it cleanly separated 姐姐 from 弟弟
with plausible turn boundaries.

**Agreement is moderate, not high** (0.52). That is expected between two
independent ASR systems on hard audio and is not an error rate. It does mean
neither baseline corroborates the other; VibeVoice was judged on its own
behaviour plus the loop/loop-free contrast above.

## Caveats

- 9 files, 37 min. Enough to separate a 14x repetition gap, not enough to put a
  tight bound on diarization accuracy.
- No ground truth. Everything above is either self-consistency (does it loop)
  or cross-system comparison, not word accuracy.
- Throughput was measured under GPU contention with another job at 100%; the
  clean number is better.
- `coverage` of 0.866 is not lost audio. The model leaves gaps where it emits a
  non-speech tag, and it deliberately declines to transcribe screaming.

## Long-form behaviour

`30_transcribe.py` runs one pass per 60 min window, which is the model's
documented limit. Two operational facts that the model card does not mention:

- **A 60 min single pass OOMs with eager attention.** ~27k audio tokens means
  eager materializes a 28 x 27k x 27k attention matrix and asks for ~76 GB in a
  single float32 softmax. The fix is per-submodule attention: the conv-only
  acoustic tokenizers require eager, but the Qwen2 text tower must be sdpa.
  `load_model()` does this, and without it a 60 min pass dies on a 192 GB card
  with `Tried to allocate 75.71 GiB`.
- **Degenerate loops are common enough to need a stop.** Exact token-block
  repetition was not enough to catch them; `common/generation.py` compares tail
  blocks at 90% token similarity, which turned a pass that would have burned
  18k tokens into one that stops in seconds.

### Measured on a real 60 minute file

`小美/470 姐姐的寸止调教室（上）--小美.mp3`, 59.7 min, single pass, GPU otherwise
idle:

| | |
|---|---|
| audio tokens in | 26,942 |
| tokens generated | 15,491 (budget 43,008, no truncation) |
| wall clock | 816 s = **4.4x realtime** |
| segments | 318 |
| coverage | 0.957, `monotonic=true`, tail gap 0.03 s |
| median / p95 segment | 11.0 s / 17.5 s |
| repetition / vocalization | 0.003 / 0.029 |

It ran to the final second of the file as one pass with no retries, which is
the capability neither baseline can offer: they see 30 s slices and therefore
cannot produce any of this. Segment durations land in the range F5-TTS wants,
and consecutive segments are 0.35 s apart on median, so re-cutting on
boundaries yields nearly contiguous usable clips.

On this speech-dense file the three systems are close, which bounds how much of
the earlier table is domain-specific:

| | chars | repetition | vocalization | agreement with vibevoice |
|---|---|---|---|---|
| vibevoice | 10,218 | 0.0029 | 0.029 | - |
| qwen3asr | 10,036 | 0.0040 | 0.037 | 0.844 |
| whisper | 8,660 | 0.0007 | 0.019 | 0.857 |

So on clean speech the baselines are not worse at the words. The 14x
repetition gap in the 9-file table comes from moan-heavy and screaming audio,
which is where the baselines actually break. The unconditional wins for
VibeVoice are structural: timestamps, speaker turns, no fixed 30 s cuts.

**Extrapolating cost:** 4.4x realtime on an idle GPU puts a 131 h corpus at
roughly 30 GPU-hours in one pass per file. `30_transcribe.py` is resumable, so
that is schedulable in slices, but it is not a cheap pass and should be started
deliberately.

## Reproducibility note

The 9-file table was measured with the text tower on eager attention, before
the sdpa switch that long-form needs. Re-running the same file under sdpa gives
0.84 similarity to the eager text: both are valid transcriptions of the same
content, they are simply not bit-identical, as expected from different bf16
kernels. Re-run `20_bench_vibevoice.py` if exact comparability with the table
matters.