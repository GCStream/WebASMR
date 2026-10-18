#!/usr/bin/env python3
"""Few-step / no-CFG sweep for the finetuned F5-TTS.

Answers the only question that matters before spending days on distillation:
how much quality does a clip lose at 16, 8 or 4 steps, and what does dropping
classifier-free guidance cost? The web manifest ships `cfg_strength: 2.0`, so
each solver step is two DiT forwards; `cfg=0` halves that again.

Metrics per configuration:

  latency   wall seconds per generation (ref preprocessing + NFE + vocoder)
  CER       character error rate of Whisper-turbo on the generated audio,
            against the text that was requested (intelligibility proxy)
  mfcc_dtw  MFCC dynamic-time-warp distance to the 32-step/cfg-2 reference for
            the same prompt, text and seed (fidelity proxy)

The reference clips are chosen for a mix of breathy/whispered and ordinary
speech, because few-step flow models lose breath and close-mic texture before
they lose words. Listen to the saved wavs before trusting any table.

    PYTHONPATH=/root/F5-TTS/src python scripts/81_steps_sweep.py \
        --ckpt /root/F5-TTS/ckpts/asmr_zh_full/model_last.pt
"""

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TAG = re.compile(r"\[[^\]]+\]")
PUNCT = re.compile(r"[\s，。！？、；：,.!?;:…~\"'“”‘’()（）\[\]【】]+")


def norm(text: str) -> str:
    return PUNCT.sub("", TAG.sub("", text or "")).lower()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def pick_prompts(meta: Path, n_refs: int, n_texts: int) -> tuple[list, list]:
    rows = [r for r in csv.DictReader(open(meta, encoding="utf-8")) if r["wav"]]
    base = meta.parent
    refs = []
    # one breathy clip and one ordinary clip, both 4-12 s with clean text
    for want_vocal in (True, False):
        for r in rows:
            dur = float(r["duration"])
            voc = float(r["vocalization"] or 0)
            text = TAG.sub("", r["text"]).strip()
            if not (4.0 <= dur <= 12.0) or len(text) < 8:
                continue
            if want_vocal and voc < 0.25:
                continue
            if not want_vocal and voc > 0.1:
                continue
            refs.append({"wav": str((base / r["wav"]).resolve()),
                         "text": text, "duration": dur, "vocalization": voc})
            break
    for r in rows:  # fallback if the filters found nothing
        if len(refs) >= n_refs:
            break
        if float(r["duration"]) < 3:
            continue
        text = TAG.sub("", r["text"]).strip()
        if len(text) >= 8:
            refs.append({"wav": str((base / r["wav"]).resolve()), "text": text,
                         "duration": float(r["duration"]),
                         "vocalization": float(r["vocalization"] or 0)})
    refs = refs[:n_refs]

    texts, seen = [], set()
    for r in rows:
        t = TAG.sub("", r["text"]).strip()
        if 10 <= len(t) <= 40 and t not in seen:
            seen.add(t)
            texts.append(t)
        if len(texts) >= n_texts:
            break
    return refs, texts


def load_asr(device: str):
    """Load Whisper directly, not via pipeline.

    The transformers ASR pipeline imports torchcodec in preprocess even when it
    is handed a numpy array, which fails on this host. Calling the model on
    processor features avoids torchcodec entirely.
    """
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    mid = "openai/whisper-large-v3-turbo"
    proc = AutoProcessor.from_pretrained(mid)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        mid, dtype=dtype, device_map=device if device == "cuda" else None).eval()
    return proc, model


def transcribe(asr, wav: np.ndarray, sr: int) -> str:
    import torch

    proc, model = asr
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    inputs = proc(wav, sampling_rate=sr, return_tensors="pt")
    dtype = next(model.parameters()).dtype
    inputs = {k: (v.to("cuda", dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
                  else v.to("cuda") if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(**inputs, language="zh", task="transcribe",
                             max_new_tokens=128)
    return proc.batch_decode(ids, skip_special_tokens=True)[0]


def mfcc_dtw(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    import librosa

    ma = librosa.feature.mfcc(y=a.astype(np.float32), sr=sr, n_mfcc=20)
    mb = librosa.feature.mfcc(y=b.astype(np.float32), sr=sr, n_mfcc=20)
    if ma.shape[1] < 2 or mb.shape[1] < 2:
        return float("nan")
    d, _ = librosa.sequence.dtw(X=ma, Y=mb, metric="euclidean")
    return float(d[-1, -1] / (ma.shape[1] + mb.shape[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", type=Path,
                    default=Path("/root/F5-TTS/data/asmr_zh_full_pinyin/vocab.txt"))
    ap.add_argument("--metadata", type=Path,
                    default=ROOT / "out/dataset/metadata_committed.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out/overnight/sweep")
    ap.add_argument("--configs", default="32:2,16:2,8:2,8:1,8:0,4:2,4:0")
    ap.add_argument("--n-refs", type=int, default=2)
    ap.add_argument("--n-texts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    refs, texts = pick_prompts(args.metadata, args.n_refs, args.n_texts)
    if not refs or not texts:
        print("no prompts found")
        return
    print(f"refs {len(refs)} (vocalization "
          f"{[round(r['vocalization'], 2) for r in refs]}), texts {len(texts)}")
    for r in refs:
        print(f"  ref {Path(r['wav']).name} {r['duration']:.1f}s voc={r['vocalization']:.2f}")

    from f5_tts.api import F5TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = F5TTS(model="F5TTS_v1_Base", ckpt_file=args.ckpt,
                  vocab_file=str(args.vocab), device=device, use_ema=True)
    print(f"model loaded in {time.time() - t0:.1f}s on {device}")

    asr = None if args.no_asr else load_asr(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for spec in args.configs.split(","):
        steps, cfg = spec.split(":")
        configs.append((int(steps), float(cfg)))

    results, reference_wavs = [], {}
    for steps, cfg in configs:
        tag = f"s{steps}_cfg{cfg:g}"
        latencies, cers, dtws = [], [], []
        for ri, ref in enumerate(refs):
            for ti, text in enumerate(texts):
                t0 = time.time()
                wav, sr, _ = model.infer(
                    ref_file=ref["wav"], ref_text=ref["text"], gen_text=text,
                    nfe_step=steps, cfg_strength=cfg, sway_sampling_coef=-1,
                    seed=args.seed, show_info=lambda *a, **k: None,
                    progress=None,
                )
                dt = time.time() - t0
                latencies.append(dt)
                key = f"r{ri}t{ti}"
                out = args.out_dir / f"{tag}_{key}.wav"
                sf.write(str(out), wav, sr)
                if tag == "s32_cfg2":
                    reference_wavs[key] = wav
                else:
                    base = reference_wavs.get(key)
                    if base is not None:
                        dtws.append(mfcc_dtw(wav, base, sr))
                if asr is not None:
                    hyp = transcribe(asr, wav.astype(np.float32), sr)
                    cers.append(cer(text, hyp))
        results.append({
            "steps": steps, "cfg": cfg, "n": len(latencies),
            "latency_s_median": round(statistics.median(latencies), 2),
            "latency_s_mean": round(statistics.mean(latencies), 2),
            "cer": round(statistics.mean(cers), 4) if cers else None,
            "mfcc_dtw": round(statistics.mean(dtws), 4) if dtws else None,
        })
        r = results[-1]
        print(f"steps={steps:2d} cfg={cfg:g}  {r['latency_s_median']:5.2f}s/clip  "
              f"CER={r['cer'] if r['cer'] is not None else '-'}  "
              f"mfcc_dtw={r['mfcc_dtw'] if r['mfcc_dtw'] is not None else '-'}",
              flush=True)

    (args.out_dir / "sweep.json").write_text(
        json.dumps({"ckpt": args.ckpt, "refs": refs, "texts": texts,
                    "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(args.out_dir / "sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {args.out_dir}/sweep.json")


if __name__ == "__main__":
    main()