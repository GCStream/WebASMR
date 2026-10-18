#!/usr/bin/env python3
"""Smoke test: transcribe one file with VibeVoice-ASR and print the segments.

The point is to see real output on ASMR audio before committing to a corpus
pass: whether it produces timestamps, how many speakers it finds, whether it
loops on moans, and what the wall-clock cost per minute of audio is.

    python scripts/10_probe_model.py --audio "/root/data_raw/audio/安柔/器材室.mp3"
    python scripts/10_probe_model.py --list-shortest 3
    python scripts/10_probe_model.py --audio <file> --max-minutes 2 --prompt "ASMR 角色扮演"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import load_24k_mono, probe_duration_s  # noqa: E402
from common.generation import loop_terminate, token_budget  # noqa: E402
from common.keys import iter_audio_files, source_key  # noqa: E402
from common.segments import (file_stats, normalize_segments, parse_generation,  # noqa: E402
                             transcript_of)

MODEL_ID = "microsoft/VibeVoice-ASR-HF"
AUDIO_ROOT = Path("/root/data_raw/audio")


SDPA_SUBMODELS = {"text_config": "sdpa"}
EAGER_SUBMODELS = ("acoustic_tokenizer_encoder_config", "semantic_tokenizer_encoder_config")


def load_model(model_id: str = MODEL_ID, dtype=torch.bfloat16, attn: str = "mixed"):
    """Load the model, optionally switching the text tower to sdpa.

    The conv-only acoustic tokenizers reject sdpa, but the Qwen2 text tower
    must not stay on eager: a 60 min single pass is ~33k audio tokens, and
    eager materializes the full attention matrix (28 heads x 33k x 33k), which
    asks for ~76 GB per softmax and OOMs a 192 GB card. sdpa keeps the text
    tower's prefill inside its memory-efficient kernel.
    """
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map="cuda", attn_implementation="eager").eval()
    if attn == "mixed":
        target = dict(SDPA_SUBMODELS)
        target.update({name: "eager" for name in EAGER_SUBMODELS})
        try:
            model.set_attn_implementation(target)
            print("attention: text_config=sdpa, acoustic tokenizers=eager", flush=True)
        except (ValueError, KeyError, AttributeError) as exc:
            print(f"attention: per-submodule switch failed ({exc}); staying eager",
                  flush=True)
    elif attn == "eager":
        print("attention: all eager", flush=True)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)
    return processor, model


def transcribe(processor, model, audio: np.ndarray, prompt: str | None = None,
               max_new_tokens: int = 8192,
               retries: int = 1) -> tuple[list, str, float, str]:
    """Generate and parse. Returns (segments, status, seconds).

    A truncated generation is retried with a larger token budget before being
    accepted, because a partial array silently loses the tail of the file.
    """
    return _generate(processor, model, [audio], prompt, max_new_tokens, retries)


def _eos_ids(processor) -> tuple[int, int | None]:
    eos = processor.tokenizer.eos_token_id
    pad = processor.tokenizer.pad_token_id
    return eos, pad


def _generate(processor, model, audios, prompt, max_new_tokens, retries,
              label: str = ""):
    """Generate and parse a batch. Returns ([(raw, status, elapsed, raw_text)], seconds)."""
    total_audio_s = sum(len(a) / 24000 for a in audios)
    budget = max_new_tokens or token_budget(max(len(a) / 24000 for a in audios))
    elapsed_all = 0.0
    results = []
    for attempt in range(retries + 1):
        cap = budget * (2 ** attempt)
        inputs = processor.apply_transcription_request(audio=list(audios), prompt=prompt)
        param_dtype = next(model.parameters()).dtype
        moved = {}
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                moved[k] = v
            elif v.is_floating_point():
                moved[k] = v.to("cuda", dtype=param_dtype)
            else:
                moved[k] = v.to("cuda")
        inputs = moved
        n_in = inputs["input_ids"].shape[1]
        eos, pad = _eos_ids(processor)
        terminator = loop_terminate(eos, pad)
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                                 logits_processor=[terminator])
        elapsed = time.time() - t0
        elapsed_all += elapsed
        decoded = processor.decode(ids[:, n_in:], return_format="raw", skip_special_tokens=True)
        if not isinstance(decoded, list):
            decoded = [decoded]
        while len(decoded) < len(audios):
            decoded.append("")

        attempt_rows, need_retry = [], False
        for row, raw_text in enumerate(decoded[:len(audios)]):
            raw_text = str(raw_text or "")
            raw, status = parse_generation(raw_text)
            attempt_rows.append((raw, status, raw_text))
            if status not in ("ok", "empty"):
                need_retry = True
        new_tokens = ids.shape[1] - n_in
        flags = []
        if terminator.triggered_rows:
            flags.append(f"degenerate-loop{terminator.triggered_rows}")
        if new_tokens >= cap:
            flags.append("hit-cap")
        print(f"  {label}batch={len(audios)} audio_tokens={n_in} "
              f"generated={new_tokens}/{cap} time={elapsed:.1f}s "
              f"({elapsed / max(1e-6, total_audio_s):.2f}x realtime) "
              f"{' '.join(flags)}", flush=True)
        if not need_retry or attempt == retries:
            return attempt_rows, elapsed_all
        results = attempt_rows
    return results, elapsed_all


def transcribe_batch(processor, model, audios, prompt: str | None = None,
                     max_new_tokens: int = 0, retries: int = 1):
    """Transcribe several clips in one generate() call.

    This is the same lever a serving engine pulls: with one sequence per step
    every step streams the whole 16.7 GB of weights from HBM, so decode is
    bandwidth-bound and extra worker processes just multiply that traffic.
    Batching keeps one weight read per step for N sequences. Clips must be of
    similar length, because the audio is padded to the longest member.
    """
    rows, elapsed = _generate(processor, model, audios, prompt, max_new_tokens, retries)
    return [(raw, status, raw_text) for raw, status, raw_text in rows], elapsed
    try:
        return json.loads(text[: cut + 1] + "]")
    except json.JSONDecodeError:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="path to an audio file")
    ap.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    ap.add_argument("--list-shortest", type=int, default=0,
                    help="list N shortest files and exit")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="truncate the audio to this many minutes (0 = whole file)")
    ap.add_argument("--prompt", default=None, help="hotwords / context hint")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--retries", type=int, default=1,
                    help="retry a truncated generation with 2x the token budget")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--json", type=Path, default=None, help="write segments here")
    args = ap.parse_args()

    if args.list_shortest:
        files = [(probe_duration_s(p), p) for p in iter_audio_files(args.audio_root)]
        for dur, p in sorted(files)[:args.list_shortest]:
            print(f"{dur / 60:7.2f} min  {source_key(p, args.audio_root)}")
        return

    if not args.audio:
        ap.error("--audio is required (or use --list-shortest)")

    path = Path(args.audio)
    duration = probe_duration_s(path)
    audio = load_24k_mono(path)
    if args.max_minutes:
        audio = audio[: int(args.max_minutes * 60 * 24000)]
        duration = len(audio) / 24000
    print(f"{source_key(path, args.audio_root)}  {duration / 60:.2f} min", flush=True)

    processor, model = load_model(args.model_id)
    raw, status, elapsed, raw_text = transcribe(processor, model, audio, args.prompt,
                                                args.max_new_tokens, args.retries)

    if status == "unparseable":
        print(f"PARSE FAILED ({status}), raw output was:\n{raw_text[:1500]}")
        return
    if status == "empty":
        print("model returned no segments for this window (reports no speech)")

    segments = normalize_segments(raw, 0.0, duration)
    stats = file_stats(segments, duration)
    print(f"\nsegments={len(segments)}")
    for seg in segments[:40]:
        print(f"  [{seg['start']:8.2f} -> {seg['end']:8.2f}] spk{seg['speaker']}  {seg['text']}")
    if len(segments) > 40:
        print(f"  ... {len(segments) - 40} more")
    print("\nstats: " + json.dumps(stats, ensure_ascii=False))
    print("\nfull transcript:\n" + transcript_of(segments)[:3000])

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"key": source_key(path, args.audio_root), "duration_s": round(duration, 2),
                       "status": status, "stats": stats, "elapsed_s": round(elapsed, 1),
                       "segments": segments}, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()