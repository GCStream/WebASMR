# How many concurrent workers?

Measured with `scripts/40_concurrency_bench.py`: the same 12 files / 1.69
audio-hours, transcribe the whole workload at several worker counts, GPU
otherwise idle.

    python scripts/40_concurrency_bench.py --files 12 --band 300 900 --workers 1 2 3 4

## Result

| workers | wall  | audio-h transcribed | aggregate audio-h/h | speedup | peak VRAM | valid |
|---------|-------|--------------------|---------------------|---------|-----------|-------|
| 1       | 521 s | 1.69 h  | 11.7 | 1.00x | 22.1 GB | yes |
| 2       | 648 s | 3.38 h  | 18.8 | **1.61x** | 44.0 GB | yes |
| 3       | 968 s | 5.07 h  | 18.9 | 1.61x | 65.2 GB | ~ (2 min contended) |
| 4       | 1814 s| 6.76 h  | 13.4 | 1.15x | 110.3 GB | **no** |

**Sweet spot: 2 workers.** Doubling gives 1.61x; the third worker adds
nothing measurable.

Each worker is a full independent process holding its own 16.7 GB copy of the
model, so VRAM is not the binding constraint until ~8 workers. Throughput is.

## Why it saturates at 2

Decode is batch-1: one sequence, one token per step, over a 7B model. Every
step streams all 16.7 GB of weights from HBM, so a worker is
memory-bandwidth-bound rather than compute-bound, and a second process does not
share that traffic, it doubles it. The 1.61x that does come from overlapping
the phases that are not bandwidth-bound (the 27k-token prefill and the conv
acoustic tokenizer). Per-worker throughput collapses as workers are added:

| workers | per-worker files/h |
|---------|--------------------|
| 1       | 12.6 |
| 2       | 10.5 |
| 3       | 6.4 |

Which is the signature of a saturated shared resource, not of idle capacity
waiting to be used.

The way past the ceiling was assumed to be batching sequences **inside** one
process. Measured properly on an idle GPU, it is not: see "Correction" below.

## Correction: batching does not beat processes on an idle GPU

An earlier version of this file claimed batching reached 1.97x where a second
process only reached 1.61x, and made batching the corpus default. That
comparison was invalid: the batch numbers were measured while the other agent's
training job shared the GPU, and the process numbers on an idle GPU. Comparing
ratios taken under different load is meaningless.

Measured head to head on an idle GPU, identical workload
(`scripts/42_sweep_bench.py`, 6 files / 35 min, one table):

| config | wall  | audio-h/h | peak VRAM | note |
|--------|-------|-----------|-----------|------|
| 1x1 | 177 s | 11.91 | 21.1 GB | paired with 1x4 below |
| 1x1 | 201 s | 10.50 | 21.4 GB | earlier window |
| **1x4** | 217 s | **9.73** | 29.4 GB | paired run 1 |
| **1x4** | 221 s | **9.54** | 29.4 GB | paired run 2 |
| **2x1** | 233 s | **18.14** | 42.8 GB | paired with 2x2 below |
| 2x1 | 280 s | 15.08 | 40.1 GB | earlier window |
| 2x2 | 291 s | 14.49 | 49.1 GB | paired with 2x1 |
| 2x4 | 281 s | 15.03 | 58.8 GB | |
| 3x1 | 968 s | 18.9 | 65.2 GB | from the process sweep |

**Two worker processes at batch 1 is the most performant configuration**, and
batching is a consistent ~20% *regression* on an idle GPU (1x4 vs 1x1, run
twice: 9.73 and 9.54 against 11.91).

The cause is padding, not contention. The processor pads each clip's audio to
the longest member of the batch, so a batch of 5.0/5.4/5.7/6.1 min clips does
~33% redundant prefill work on the short ones. On an idle GPU there is spare
bandwidth for batching to reclaim, and the padding waste is not repaid. Under
contention the trade flips: batching reached 1.97x in `41_batch_bench.py`
because a single batch-1 process cannot fill SMs that are being time-sliced.

So the honest rule is:

| situation | best config | why |
|---|---|---|
| GPU free | 2-3 workers x batch 1 | fills the machine, no padding waste |
| GPU shared with another job | 1-2 workers x batch 4 | recovers SM utilisation, and the shared job keeps progressing |

The corpus pass defaults to 3 workers x batch 1 because it runs for hours on a
box whose GPU comes and goes:

    python scripts/50_run_corpus.py --workers 3 --batch-size 1

`--batch-size` is one flag away if the pass has to share the card.

### Measurement noise on a shared box

Repeat runs of an identical configuration varied by 2.4x here (1x1 took 177 s
and then 430 s), all flagged clean: the contamination check samples GPU
processes before and after each configuration, so load that starts and stops
*inside* one is invisible. Treat single runs as indicative and prefer paired
comparisons run back to back, as the 1x1/1x4 and 2x1/2x2 pairs above are.

## Batching inside one worker (measured under contention)

`scripts/41_batch_bench.py`, same 6 files / 35 min of audio, same process,
varying clips per `generate()` call. Rate uses summed per-batch generation time
so model load is excluded.

| batch | generate s | audio-h/h | vs batch=1 |
|-------|-----------|-----------|------------|
| 1 | 454 | 4.65 | 1.00x |
| 2 | 291 | 7.25 | 1.56x |
| 3 | 257 | 8.21 | 1.77x |
| **4** | 231 | 9.14 | **1.97x** |
| 6 | 260 | 8.12 | 1.75x |
| 8 | 226 | 9.34 | 2.01x |

**Sweet spot: batch 4**, at 1.97x, against 1.61x for doubling the worker count.
Batching wins because it removes the duplication rather than adding contention:
one weight read serves N sequences, whereas a second process issues its own.

Caveats on this table. The 6 and 8 rows ran as the *same* single batch of 6
(only 6 files exist), so their 260 s vs 226 s spread is run-to-run noise of
~15%; anything under that difference is not a real effect, which is why the
recommendation is 4 rather than 8. These numbers were also taken while the
other agent's LoRA run shared the GPU, so the absolute rates are depressed
against the idle-GPU figures above; the ratios are the usable result.

Batching does not change the output. Same three files at batch 1 vs batch 3:

| file | segments b1 / b3 | text similarity |
|------|------------------|-----------------|
| 寸止stage 1 规则介绍 | 45 / 45 | 0.990 |
| MINI姐/病娇抖s女友 | 65 / 65 | 0.964 |
| 竹曦/1-01(无下限爆骂) | 26 / 26 | 0.969 |

Identical segment counts, and the small text deltas are bf16 numerics from
different batch shapes, the same effect seen switching eager to sdpa.

Two implementation notes that batching forced:

- A `StoppingCriteria` can only stop the whole batch, so one looping clip
  would truncate its neighbours. `common/generation.py` instead masks the
  logits down to EOS per row (`LoopTerminate`), ending just that sequence.
- Audio is padded to the longest member, so files are batched in
  duration-sorted order and `--batch-max-audio-s` excludes long files, which
  are compute-bound in prefill anyway.

## The 4-worker row is invalid

It looks like throughput collapses past 3 workers, but that is an artifact:
this box is shared, and the other agent started an LoRA run (`sft-bi-v3`,
`runs/sft-bi-v3/run.json` at 06:15:26) partway through the sweep. Reconstructing
the windows from log mtimes:

| config | window | time sharing the GPU with training |
|--------|--------|------------------------------------|
| 1 | 05:42:15-05:50:56 | clean |
| 2 | 05:50:42-06:01:30 | clean |
| 3 | 06:01:44-06:17:52 | 2 of 16 min |
| 4 | 06:16:07-06:46:21 | **26 of 30 min** |

So 4 workers is simply unmeasured. Two independent processes plus a training
job is a different experiment from N ASR workers alone.

Practical consequence: this benchmark cannot be trusted unless the GPU is
verified idle for the whole run. Check before repeating it:

    rocm-smi --showpids      # expect no KFD PIDs
    rocm-smi --showuse       # expect 0%

## Corpus projection

Corpus is 578 files / **139.5 audio-hours** (the 131 h figure quoted earlier
counted only lowercase `.mp3` and missed 42 `.m4a` and 8 `.MP3`). At the
measured 18.8 audio-h/h with 2 workers on an idle GPU: **~7.4 hours**. Sharing
the GPU with a training job stretches that; the pass is resumable per shard, so
it can be paused and restarted without losing completed files.

## Why not vLLM

The documented serving path is a plugin run from `vllm/vllm-openai:v0.14.1`, a
CUDA image, and there is no `vllm-rocm` distribution on PyPI. This box is
MI300X/ROCm with no NVIDIA runtime, so that path would mean porting the plugin
to a ROCm vLLM build. What the plugin would actually buy here is continuous
batching, which the table above gets to 2x with no new dependency and no risk
to the shared environment. Port vLLM only if throughput needs to go beyond
that.

Run it with:

    python scripts/50_run_corpus.py --workers 2
    python scripts/60_merge_shards.py --shard-dir out/corpus --out out/segments.jsonl