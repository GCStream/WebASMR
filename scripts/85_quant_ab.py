#!/usr/bin/env python3
"""Generate an fp16-vs-int4 listening A/B using the web voice 小莼 (xiaochun).

Uses the exported ONNX graphs exactly as the browser does: one `dit_step` call
per Euler step (CFG by running cond and uncond and combining), then the vocoder
graph and the ISTFT tail that the browser implements in JS. The reference mel is
the same `xiaochun.bin` the app ships, so the voice is the app's default.

Each item is rendered twice from the same noise seed: once with the fp16 graphs
and once with the int4 graphs. The index stores a blind A/B assignment so the
reviewer does not know which side is quantized.

    PYTHONPATH=/root/F5-TTS/src python scripts/85_quant_ab.py
    ... --texts 1        # smoke
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
WEBMODELS = Path("/root/LocalASMR/asmr-bilingual-webml/web/public/models/f5tts")
ONNX = ROOT / "models/onnx"
OUT = ROOT / "out/quant_ab"
VOCAB = Path("/root/F5-TTS/data/asmr_zh_full_pinyin/vocab.txt")

# Tongue twisters stress intelligibility, which is where quantization and
# few-step sampling damage first.
TEXTS = [
    "吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
    "四是四，十是十，十四是十四，四十是四十。",
    "红鲤鱼与绿鲤鱼与驴，黑化肥发灰会挥发，灰化肥挥发会发黑。",
]

# The shipped xiaochun prompt opens with ~2 s of "嗯，嗯，嗯，哇" moans, which the
# model continues into the target and garbles the first words. Drop those frames
# and the matching text; the voice is unchanged.
REF_TRIM_FRAMES = 200
REF_TEXT = "看来你真的很喜欢女生的脚呢。哼，"
N_MEL = 100
SR = 24000
HOP = 256


def load_sessions():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {}
    for variant in ("fp16", "int4"):
        base = ONNX / ("tts_" + variant)
        sessions[variant] = {
            "dit": ort.InferenceSession(str(base / "dit_step.onnx"), so,
                                        providers=["CPUExecutionProvider"]),
            "voc": ort.InferenceSession(str(base / "vocoder.onnx"), so,
                                        providers=["CPUExecutionProvider"]),
        }
    return sessions


def ref_mel() -> tuple[np.ndarray, dict]:
    meta = json.loads((WEBMODELS / "refs/xiaochun.json").read_text())
    n_mel, frames = meta["mel_shape"]
    data = np.fromfile(WEBMODELS / meta["mel_file"], dtype=np.float16)
    # the pack stores [n_mel, frames] channel-major (the JS loader indexes
    # mel[m * frames + f]); transposing keeps the reference voice intact
    mel = data.reshape(n_mel, frames).T.astype(np.float16)  # [frames, 100]
    mel = mel[REF_TRIM_FRAMES:]
    meta = {**meta, "ref_text": REF_TEXT,
            "trim_note": f"dropped {REF_TRIM_FRAMES} frames of leading moans"}
    return mel, meta


def text_ids(text: str) -> list[int]:
    from f5_tts.model.utils import convert_char_to_pinyin, list_str_to_idx

    vocab = {}
    for i, line in enumerate(VOCAB.read_text(encoding="utf-8").split("\n")):
        if line:
            vocab[line] = i
    tokens = convert_char_to_pinyin([text], polyphone=True)
    return list_str_to_idx(tokens, vocab)[0].tolist()


def velocity(sess, y, cond, ids, t, cfg):
    def run(c, ids_use):
        feeds = {
            "x": y.astype(np.float16)[None],
            "cond": c.astype(np.float16)[None],
            "text_ids": np.asarray([ids_use], dtype=np.int64),
            "time": np.asarray([t], dtype=np.float16),
            "mask": np.ones((1, y.shape[0]), dtype=bool),
        }
        return sess.run(["velocity"], feeds)[0][0].astype(np.float32)

    vc = run(cond, ids)
    if cfg < 1e-5:
        return vc
    vu = run(np.zeros_like(cond), [0] * len(ids))
    return vc + (vc - vu) * cfg


def synth(sessions, variant, ref, ids, frames, steps, cfg, sway, seed, ref_frames):
    cond = np.zeros((frames, N_MEL), dtype=np.float16)
    c = min(ref.shape[0], frames)
    cond[:c] = ref[:c]
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((frames, N_MEL)).astype(np.float16)
    t = np.linspace(0, 1, steps + 1)
    t = t + sway * (np.cos(np.pi / 2 * t) - 1 + t)
    dit = sessions[variant]["dit"]
    for k in range(steps):
        v = velocity(dit, y.astype(np.float32), cond.astype(np.float32),
                     ids, float(t[k]), cfg)
        y = (y.astype(np.float32) + (t[k + 1] - t[k]) * v).astype(np.float16)
    # F5 generates prompt + target, then keeps only the target mel
    y_target = y[ref_frames:]
    voc = sessions[variant]["voc"]
    planes = voc.run(["stft_planes"], {"mel": y_target.T[None].astype(np.float16)})[0][0]
    planes = planes.astype(np.float32)
    mag = np.minimum(np.exp(planes[:513]), 1e2)
    phase = planes[513:]
    import torch

    spec = torch.tensor(np.stack([mag * np.cos(phase), mag * np.sin(phase)], axis=-1))
    wav = torch.istft(torch.view_as_complex(spec)[None], n_fft=1024,
                      hop_length=HOP, win_length=1024,
                      window=torch.hann_window(1024), center=True)[0].numpy()
    return wav.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", type=int, default=len(TEXTS))
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--chars-per-second", type=float, default=4.0,
                    help="frames are sized so the text has room at this rate")
    ap.add_argument("--min-seconds", type=float, default=6.0)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    ref, meta = ref_mel()
    print(f"voice {meta['name_zh']} ref {ref.shape[0]} frames "
          f"({ref.shape[0] * HOP / SR:.2f}s)")
    sessions = load_sessions()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    rng = np.random.default_rng(7)
    ref_frames = ref.shape[0]
    ref_text = meta["ref_text"]
    for i, text in enumerate(TEXTS[:args.texts]):
        # F5 convention: prompt text + target text, and duration = prompt +
        # prompt_len/ref_text_len * gen_text_len (see infer_batch_process)
        ref_bytes = len(ref_text.encode("utf-8"))
        gen_bytes = len(text.encode("utf-8"))
        speed = 0.3 if gen_bytes < 10 else 1.0
        target_frames = int(ref_frames / ref_bytes * gen_bytes / speed)
        frames = ref_frames + max(target_frames, 64)
        ids = text_ids(ref_text + text)
        target_s = target_frames * HOP / SR
        item_id = f"q{i:02d}"
        durations = {}
        for variant in ("fp16", "int4"):
            t0 = time.time()
            wav = synth(sessions, variant, ref, ids, frames, args.steps,
                        args.cfg, args.sway, seed=1234 + i,
                        ref_frames=ref_frames)
            path = args.out_dir / f"{item_id}_{variant}.wav"
            sf.write(str(path), wav, SR, subtype="PCM_16")
            durations[variant] = round(len(wav) / SR, 2)
            print(f"  {item_id} {variant}: {len(wav) / SR:.2f}s in "
                  f"{time.time() - t0:.1f}s", flush=True)
        blind = ["fp16", "int4"] if rng.random() < 0.5 else ["int4", "fp16"]
        items.append({
            "id": item_id, "text": text, "voice": meta["name_zh"],
            "steps": args.steps, "cfg": args.cfg,
            "a": blind[0], "b": blind[1],
            "a_audio": f"{item_id}_{blind[0]}.wav",
            "b_audio": f"{item_id}_{blind[1]}.wav",
            "seconds": durations, "target_seconds": round(target_s, 1),
            "ref_text": ref_text,
        })
    (args.out_dir / "index.json").write_text(
        json.dumps({"voice": meta["name_zh"], "items": items}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out_dir}/index.json ({len(items)} items)")


if __name__ == "__main__":
    main()