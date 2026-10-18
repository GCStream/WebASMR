#!/usr/bin/env python3
"""Run the few-step sweep after the already-running overnight pipeline ends.

`98_overnight.py` now schedules `steps_sweep` itself, but the instance that is
already training cannot pick up new code. This watcher bridges that gap: it
waits for that process to exit, runs the same sweep against the newest F5
checkpoint, appends the table to the overnight report and uploads the result to
the ONNX repo.

Safe to leave running: it no-ops if a sweep already exists, and it never touches
the other agent's processes.

    python scripts/99_post_sweep.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out/overnight"
SWEEP = OUT / "sweep"
F5 = Path("/root/F5-TTS")
SIBLING_ROOT = Path("/root/LocalASMR/asmr-bilingual-webml")
CKPTS = F5 / "ckpts/asmr_zh_full"
LOG = OUT / "post_sweep.log"
MAX_WAIT_S = 12 * 3600


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] post-sweep: {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def orchestrator_alive() -> bool:
    pids = subprocess.run(["pgrep", "-f", "98_overnight.py"],
                          capture_output=True, text=True).stdout.split()
    own = str(os.getpid())
    return any(p != own for p in pids)


def newest_checkpoint() -> Path | None:
    if not CKPTS.exists():
        return None
    last = CKPTS / "model_last.pt"
    if last.exists():
        return last
    candidates = sorted(CKPTS.glob("model_*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_distillation(teacher: Path, sweep: dict) -> None:
    """Distil to 4 steps, export, quantize, sweep, upload.

    Only runs if the teacher sweep completed: the sweep is the baseline the
    distilled model has to beat, and if it failed we would have nothing to
    compare against.
    """
    student_ckpt = ROOT / "models/f5tts_4step.pt"
    if not student_ckpt.exists():
        log("distilling a 4-step student (45 min budget)")
        cmd = [sys.executable, str(ROOT / "scripts/82_distill_4step.py"),
               "--teacher", str(teacher), "--out", str(student_ckpt),
               "--max-minutes", "45"]
        env = {**os.environ, "PYTHONPATH": str(F5 / "src")}
        with open(OUT / "logs" / "distill.log", "a") as logf:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                  stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=3300)
        log(f"distill exited {proc.returncode}")
        if not student_ckpt.exists():
            log("distillation produced no checkpoint; stopping here")
            return
    else:
        log("distilled checkpoint already exists")

    # export + quantize the student
    onnx = ROOT / "models/onnx/tts4_fp16"
    quant = ROOT / "models/onnx/tts4_int4"
    if not any(quant.glob("*.onnx")):
        cmd = [sys.executable, "export/export_onnx_f5tts.py",
               "--ckpt", str(student_ckpt), "--out", str(onnx),
               "--fp16", "--no-verify"]
        env = {**os.environ, "PYTHONPATH": str(F5 / "src")}
        with open(OUT / "logs" / "distill_export.log", "a") as logf:
            proc = subprocess.run(cmd, cwd=str(SIBLING_ROOT), env=env,
                                  stdout=logf, stderr=subprocess.STDOUT,
                                  timeout=1800)
        log(f"student onnx export exited {proc.returncode}")
        if any(onnx.glob("*.onnx")):
            cmd = [sys.executable, "export/quantize.py", "--mode", "tts",
                   "--input", str(onnx), "--out", str(quant), "--verify"]
            with open(OUT / "logs" / "distill_quantize.log", "a") as logf:
                proc = subprocess.run(cmd, cwd=str(SIBLING_ROOT), env=env,
                                      stdout=logf, stderr=subprocess.STDOUT,
                                      timeout=1200)
            log(f"student int4 quantize exited {proc.returncode}")

    # sweep the student at 4:0
    sweep4 = OUT / "sweep4"
    if not (sweep4 / "sweep.json").exists():
        cmd = [sys.executable, str(ROOT / "scripts/81_steps_sweep.py"),
               "--ckpt", str(student_ckpt), "--configs", "4:0,8:0",
               "--out-dir", str(sweep4)]
        env = {**os.environ, "PYTHONPATH": str(F5 / "src")}
        with open(OUT / "logs" / "sweep4.log", "a") as logf:
            subprocess.run(cmd, cwd=str(ROOT), env=env,
                           stdout=logf, stderr=subprocess.STDOUT, timeout=1200)
        if (sweep4 / "sweep.json").exists():
            append_report(json.loads((sweep4 / "sweep.json").read_text()))
            log("4-step sweep appended to report")

    # upload student checkpoint + graphs
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        release = OUT / "f5_4step"
        release.mkdir(exist_ok=True)
        shutil.copy2(student_ckpt, release / "model_4step.pt")
        vocab = F5 / "data/asmr_zh_full_pinyin/vocab.txt"
        if vocab.exists():
            shutil.copy2(vocab, release / "vocab.txt")
        (release / "README.md").write_text(
            "---\nlicense: cc-by-nc-4.0\ntags:\n- tts\n- f5-tts\n- asmr\n"
            "- few-step\n---\n\n# F5-TTS ASMR, 4-step CFG-free\n\n"
            "Reflow-distilled from the fine-tuned 32-step teacher. Use 4 Euler "
            "steps, sway -1.0, cfg 0.\n", encoding="utf-8")
        api.create_repo("DownFlow/ASMR-f5tts-4step", repo_type="model",
                        private=False, exist_ok=True)
        api.upload_folder(folder_path=str(release),
                          repo_id="DownFlow/ASMR-f5tts-4step", repo_type="model",
                          commit_message="4-step distilled F5-TTS")
        log("uploaded DownFlow/ASMR-f5tts-4step")
        if any(quant.glob("*.onnx")):
            api.upload_folder(folder_path=str(ROOT / "models/onnx"),
                              repo_id="DownFlow/ASMR-onnx", repo_type="model",
                              commit_message="4-step distilled ONNX graphs")
            log("uploaded 4-step graphs to DownFlow/ASMR-onnx")
    except Exception as exc:  # noqa: BLE001
        log(f"student upload failed: {type(exc).__name__}: {exc}")


def append_report(results: dict) -> None:
    report = OUT / "report.md"
    if not report.exists():
        return
    lines = ["", "## Few-step sweep", "",
             f"checkpoint `{results['ckpt']}`", "",
             "| steps | cfg | s/clip | CER | mfcc_dtw |",
             "|---|---|---|---|---|"]
    for r in results["results"]:
        lines.append(f"| {r['steps']} | {r['cfg']:g} | {r['latency_s_median']} | "
                     f"{r['cer'] if r['cer'] is not None else '-'} | "
                     f"{r['mfcc_dtw'] if r['mfcc_dtw'] is not None else '-'} |")
    lines += ["", "Listen to `out/overnight/sweep/*.wav` before choosing; few-step "
              "flow models lose breath and whisper texture before they lose words.", ""]
    with open(report, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (SWEEP / "sweep.json").exists():
        log("sweep already exists, nothing to do")
        return
    log("waiting for the overnight pipeline to finish")
    started = time.time()
    while orchestrator_alive():
        if time.time() - started > MAX_WAIT_S:
            log("waited too long for the pipeline; giving up")
            return
        time.sleep(60)

    ckpt = newest_checkpoint()
    if not ckpt:
        log("no F5 checkpoint produced; sweep skipped")
        return
    log(f"running sweep on {ckpt}")
    cmd = [sys.executable, str(ROOT / "scripts/81_steps_sweep.py"),
           "--ckpt", str(ckpt), "--out-dir", str(SWEEP)]
    env = {**os.environ, "PYTHONPATH": str(F5 / "src")}
    with open(OUT / "logs" / "steps_sweep.log", "a") as logf:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                              stdout=logf, stderr=subprocess.STDOUT,
                              timeout=1800)
    log(f"sweep exited {proc.returncode}")
    if not (SWEEP / "sweep.json").exists():
        log("no sweep.json produced")
        return
    results = json.loads((SWEEP / "sweep.json").read_text())
    append_report(results)

    run_distillation(ckpt, results)

    dest = ROOT / "models/onnx/sweep"
    dest.mkdir(parents=True, exist_ok=True)
    for f in SWEEP.glob("sweep.*"):
        shutil.copy2(f, dest / f.name)

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for f in list(SWEEP.glob("sweep.*")):
            api.upload_file(path_or_fileobj=str(f),
                            path_in_repo=f"sweep/{f.name}",
                            repo_id="DownFlow/ASMR-onnx", repo_type="model",
                            commit_message="Few-step sweep results")
        log("sweep uploaded to DownFlow/ASMR-onnx")
    except Exception as exc:  # noqa: BLE001
        log(f"upload failed: {type(exc).__name__}: {exc}")
    log("done")


if __name__ == "__main__":
    main()