#!/usr/bin/env python3
"""Reflow-distil F5-TTS to 4 steps with CFG baked into the student.

Why not just fine-tune at 4 steps: a plain flow-matching fine-tune teaches the
model the *instantaneous* velocity at uniformly sampled times. At inference with
4 Euler steps the solver needs the *average* velocity over a quarter of the
trajectory, and with the usual CFG it spends 2 forwards per step. So a 4-step
run of a 32-step model is slower than it needs to be and drifts.

This script trains the student to predict exactly what the 4-step sampler needs:

  1. freeze the fine-tuned teacher;
  2. sample x0 and the same masked conditioning the model sees in training;
  3. integrate the teacher (CFG 2.0) along a fine 32-step grid to get the true
     trajectory, and read it at the sway-warped 4-step nodes;
  4. train the student to output (x_{k+1} - x_k) / (t_{k+1} - t_k) at each node.

Because the teacher's velocity already includes guidance, the student needs no
CFG at inference: 4 forwards per clip instead of 64.

Output is a standard F5 trainer checkpoint, so `export_onnx_f5tts.py` loads it
unchanged.

    PYTHONPATH=/root/F5-TTS/src python scripts/82_distill_4step.py \
        --teacher /root/F5-TTS/ckpts/asmr_zh_full/model_last.pt \
        --out /root/WebASMR/models/f5tts_4step.pt --max-minutes 45

    ... --smoke        # 3 steps on 2 clips, for validating the loop
"""

import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
STEPS = 4
SWAY = -1.0


def sway_grid(steps: int, device, dtype=torch.float32) -> torch.Tensor:
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    return t + SWAY * (torch.cos(torch.pi / 2 * t) - 1 + t)


def build_vocab(path: Path) -> dict:
    vocab = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if line:
            vocab[line] = i
    if vocab.get(" ") != 0:
        raise SystemExit("vocab.txt must start with a space at index 0")
    return vocab


def load_clips(meta: Path, n: int) -> list[dict]:
    rows = [r for r in csv.DictReader(open(meta, encoding="utf-8")) if r["wav"]]
    base = meta.parent
    picked, seen = [], set()
    step = max(1, len(rows) // n)
    for r in rows[::step]:
        text = r["text"].strip()
        if not text or text in seen:
            continue
        seen.add(text)
        picked.append({"wav": str((base / r["wav"]).resolve()), "text": text,
                       "duration": float(r["duration"])})
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--vocab", type=Path,
                    default=Path("/root/F5-TTS/data/asmr_zh_full_pinyin/vocab.txt"))
    ap.add_argument("--metadata", type=Path,
                    default=ROOT / "out/dataset/metadata_committed.csv")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "models/f5tts_4step.pt")
    ap.add_argument("--n-clips", type=int, default=800)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cfg-teacher", type=float, default=2.0)
    ap.add_argument("--fine-steps", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from f5_tts.infer.utils_infer import load_model
    from f5_tts.model import DiT
    from f5_tts.model.modules import MelSpec
    from f5_tts.model.utils import lens_to_mask, list_str_to_idx, mask_from_frac_lengths
    import soundfile as sf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    vocab = build_vocab(args.vocab)

    ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    MEL = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
               target_sample_rate=24000, mel_spec_type="vocos")
    teacher = load_model(DiT, ARCH, args.teacher, "vocos", str(args.vocab),
                         use_ema=True, device=device).to(device=device,
                                                       dtype=torch.float32).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher).to(device=device, dtype=torch.float32).train()
    for p in student.parameters():
        p.requires_grad_(True)
    mel_spec = MelSpec(**MEL).to(device)
    print(f"teacher {args.teacher} loaded; student initialised from it")

    clips = load_clips(args.metadata, 2 if args.smoke else args.n_clips)
    if not clips:
        raise SystemExit("no clips found")
    print(f"clips: {len(clips)}")

    n_steps = 4 if args.smoke else 10**9
    max_s = 120 if args.smoke else args.max_minutes * 60
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.99))
    t_nodes = sway_grid(STEPS, device)
    t_fine = sway_grid(args.fine_steps, device)
    node_idx = [int(torch.argmin((t_fine - tk).abs()).item()) for tk in t_nodes]

    started = time.time()
    step = 0
    losses = []
    while step < n_steps and (time.time() - started) < max_s:
        for clip in clips:
            if step >= n_steps or (time.time() - started) >= max_s:
                break
            audio, sr = sf.read(clip["wav"], dtype="float32", always_2d=True)
            if sr != 24000:
                continue
            wav = torch.from_numpy(audio.T.copy()).to(device)  # [1, samples]
            with torch.no_grad():
                mel = mel_spec(wav)                      # [1, 100, T]
            x1 = mel.permute(0, 2, 1)                    # [1, T, 100]
            b, seq_len, _ = x1.shape
            lens = torch.tensor([seq_len], device=device)
            mask = lens_to_mask(lens, length=seq_len)
            frac = torch.empty(b, device=device).uniform_(0.7, 1.0)
            span = mask_from_frac_lengths(lens, frac) & mask
            cond = torch.where(span[..., None], torch.zeros_like(x1), x1)
            text = list_str_to_idx([clip["text"]], vocab).to(device)
            x0 = torch.randn_like(x1)

            with torch.no_grad():
                def teacher_v(x, t):
                    tt = torch.full((b,), float(t), device=device, dtype=dtype)
                    pred = teacher.transformer(
                        x=x, cond=cond, text=text, time=tt, mask=mask,
                        cfg_infer=True)
                    p, u = torch.chunk(pred, 2, dim=0)
                    return p + (p - u) * args.cfg_teacher

                xs, x = [], x0
                for i in range(args.fine_steps):
                    dt = t_fine[i + 1] - t_fine[i]
                    x = x + dt * teacher_v(x, t_fine[i])
                    xs.append(x)
                # read the trajectory at the 4-step nodes (x at t=0 is x0)
                traj = [x0] + xs
                nodes = [traj[idx] for idx in node_idx]

            loss = 0.0
            for k in range(STEPS):
                dt = (t_nodes[k + 1] - t_nodes[k]).clamp_min(1e-4)
                target = (nodes[k + 1] - nodes[k]) / dt
                tt = torch.full((b,), float(t_nodes[k]), device=device, dtype=dtype)
                pred = student.transformer(
                    x=nodes[k], cond=cond, text=text, time=tt, mask=mask,
                    drop_audio_cond=False, drop_text=False)
                loss = loss + F.mse_loss(pred[span], target[span])
            loss = loss / STEPS
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
            step += 1
            if step % 10 == 0:
                rate = step / max(time.time() - started, 1e-9)
                print(f"step {step:4d} loss {np.mean(losses[-50:]):.4f} "
                      f"({rate * 60:.1f} clips/min)", flush=True)

    elapsed = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": student.state_dict(),
                "ema_model_state_dict": student.state_dict(),
                "update": step}, str(args.out))
    print(f"saved {args.out} after {step} steps in {elapsed / 60:.1f} min "
          f"(mean loss {np.mean(losses):.4f})")


if __name__ == "__main__":
    main()