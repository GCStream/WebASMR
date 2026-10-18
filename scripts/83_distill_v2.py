#!/usr/bin/env python3
"""Longer 4-step distillation: distill -> export -> quantize -> sweep -> upload.

The 45-minute run improved 4:0 CER from 0.775 to 0.597 but is far from the
16-step 0.097. This runs the same reflow objective for 90 minutes over 2,000
clips, exports and quantizes the student, sweeps 4:0 and 8:0 with a 32:2
reference for fidelity, and uploads to the 4-step repos.

    python scripts/83_distill_v2.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F5 = Path("/root/F5-TTS")
SIBLING = Path("/root/LocalASMR/asmr-bilingual-webml")
TEACHER = F5 / "ckpts/asmr_zh_full/model_91750.pt"
STUDENT = ROOT / "models/f5tts_4step_v2.pt"
ONNX = ROOT / "models/onnx/tts4v2_fp16"
QUANT = ROOT / "models/onnx/tts4v2_int4"
SWEEP = ROOT / "out/overnight/sweep4v2"
LOG = ROOT / "out/overnight/logs"
DEADLINE = time.time() + 2 * 3600


def run(name, cmd, cwd, timeout, env=None):
    LOG.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {' '.join(cmd[:5])}...", flush=True)
    with open(LOG / f"{name}.log", "a") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f,
                              stderr=subprocess.STDOUT, timeout=timeout,
                              env={**__import__("os").environ, **(env or {})})
    print(f"[{time.strftime('%H:%M:%S')}] {name}: exit {proc.returncode}", flush=True)
    return proc.returncode


def main() -> None:
    if not TEACHER.exists():
        raise SystemExit(f"teacher missing: {TEACHER}")
    env = {"PYTHONPATH": str(F5 / "src")}

    run("distill_v2", [sys.executable, str(ROOT / "scripts/82_distill_4step.py"),
                       "--teacher", str(TEACHER), "--out", str(STUDENT),
                       "--n-clips", "2000", "--max-minutes", "90"],
        ROOT, 6000, env)
    if not STUDENT.exists():
        raise SystemExit("distillation produced no checkpoint")

    run("distill_v2_export", [sys.executable, "export/export_onnx_f5tts.py",
                              "--ckpt", str(STUDENT), "--out", str(ONNX),
                              "--fp16", "--no-verify"],
        SIBLING, 1800, env)
    if any(ONNX.glob("*.onnx")):
        run("distill_v2_quantize", [sys.executable, "export/quantize.py",
                                    "--mode", "tts", "--input", str(ONNX),
                                    "--out", str(QUANT), "--verify"],
            SIBLING, 1200, env)

    run("sweep4v2", [sys.executable, str(ROOT / "scripts/81_steps_sweep.py"),
                     "--ckpt", str(STUDENT), "--configs", "32:2,16:2,8:0,4:0",
                     "--out-dir", str(SWEEP)],
        ROOT, 1800, env)

    summary = {}
    if (SWEEP / "sweep.json").exists():
        summary = json.loads((SWEEP / "sweep.json").read_text())
        report = ROOT / "out/overnight/report.md"
        with open(report, "a", encoding="utf-8") as f:
            f.write("\n## 4-step distillation v2 (2,000 clips, 90 min)\n\n")
            f.write("| steps | cfg | s/clip | CER | mfcc_dtw |\n|---|---|---|---|---|\n")
            for r in summary["results"]:
                f.write(f"| {r['steps']} | {r['cfg']:g} | {r['latency_s_median']} | "
                        f"{r['cer']} | {r['mfcc_dtw']} |\n")

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        if STUDENT.exists():
            release = ROOT / "out/overnight/f5_4step_v2"
            release.mkdir(parents=True, exist_ok=True)
            __import__("shutil").copy2(STUDENT, release / "model_4step_v2.pt")
            vocab = F5 / "data/asmr_zh_full_pinyin/vocab.txt"
            if vocab.exists():
                __import__("shutil").copy2(vocab, release / "vocab.txt")
            (release / "README.md").write_text(
                "---\nlicense: cc-by-nc-4.0\ntags:\n- tts\n- f5-tts\n- asmr\n"
                "- few-step\n---\n\n# F5-TTS ASMR, 4-step CFG-free (v2)\n\n"
                "Reflow-distilled 90 min on 2,000 clips. Use 4 Euler steps, "
                "sway -1.0, cfg 0.\n", encoding="utf-8")
            api.upload_folder(folder_path=str(release),
                              repo_id="DownFlow/ASMR-f5tts-4step",
                              repo_type="model",
                              commit_message="Longer 4-step distillation (v2)")
        if any(QUANT.glob("*.onnx")):
            api.upload_folder(folder_path=str(ROOT / "models/onnx"),
                              repo_id="DownFlow/ASMR-onnx", repo_type="model",
                              commit_message="4-step v2 ONNX graphs")
        print("uploaded v2", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("upload failed:", type(exc).__name__, exc, flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()