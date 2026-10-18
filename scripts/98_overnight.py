#!/usr/bin/env python3
"""Unattended overnight pipeline: train -> export -> quantize -> upload.

Runs with a hard wall-clock budget (default 8 h) and no interactive input. Every
stage is time-boxed, logged separately, and recorded in a state file so the run
can be inspected or resumed in the morning. A failed stage is recorded and the
pipeline moves on: a missing F5 export should not cost the LLM upload.

Stage order and budget (minutes of an 8 h run):

    prepare_f5     10   build the full 33.7 h F5 dataset
    train_f5      210   ~10 epochs on full data, checkpoint every 250 updates
    train_llm      60   10 epochs of LoRA on the reconciled instructions
    export_llm     45   merge LoRA, ONNX, int4 quantize
    export_f5      45   DiT + vocoder ONNX, int4 quantize
    upload         40   adapters, checkpoint, quantized graphs, model cards

    python scripts/98_overnight.py --hours 8
    tail -f out/overnight/status.log
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "models"
SIBLING = Path("/root/LocalASMR/asmr-bilingual-webml")
F5 = Path("/root/F5-TTS")
F5_DATA = F5 / "data/asmr_zh_full_pinyin"
F5_CKPTS = F5 / "ckpts/asmr_zh_full"
OUT = ROOT / "out/overnight"

STAGES = [
    ("prepare_f5", 10),
    ("train_f5", 210),
    ("train_llm", 60),
    ("export_llm", 45),
    ("export_f5", 45),
    ("steps_sweep", 20),
    ("distill_4step", 60),
    ("upload", 40),
]

STATE = OUT / "state.json"
STATUS = OUT / "status.log"


class Overnight:
    def __init__(self, hours: float):
        self.start = time.time()
        self.deadline = self.start + hours * 3600
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "logs").mkdir(exist_ok=True)
        self.state = self._load_state()
        self.state.setdefault("started", self.start)
        self.state.setdefault("stages", {})
        self.state["hours"] = hours

    def _load_state(self) -> dict:
        if STATE.exists():
            try:
                return json.loads(STATE.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def save(self) -> None:
        STATE.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(STATUS, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def elapsed_h(self) -> float:
        return (time.time() - self.start) / 3600

    def remaining_s(self) -> float:
        return self.deadline - time.time()

    def stage_deadline(self, minutes: int) -> float:
        return min(time.time() + minutes * 60, self.deadline - 300)

    def run(self, name: str, cmd: list[str], cwd: Path, timeout_s: float,
            env: dict | None = None, graceful: bool = False) -> dict:
        """Run a subprocess with a hard timeout. Returns a result record."""
        log_path = OUT / "logs" / f"{name}.log"
        started = time.time()
        self.log(f"start {name}: {' '.join(cmd[:6])} ... (cap {timeout_s / 60:.0f} min)")
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log,
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ, **(env or {})},
                                    start_new_session=True)
            try:
                proc.wait(timeout=max(30, timeout_s))
                code = proc.returncode
            except subprocess.TimeoutExpired:
                self.log(f"{name}: hit time cap, stopping")
                for sig in ([signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
                            if graceful else [signal.SIGKILL]):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        proc.wait(timeout=45)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                code = -9
        duration = time.time() - started
        result = {"cmd": cmd, "cwd": str(cwd), "returncode": code,
                  "duration_s": round(duration, 1), "log": str(log_path),
                  "ok": code == 0}
        self.state["stages"][name] = result
        self.save()
        self.log(f"end {name}: code={code} in {duration / 60:.1f} min")
        return result

    # ---------------- stages ----------------

    def prepare_f5(self, deadline: float) -> None:
        import csv

        committed = ROOT / "out/dataset/metadata_committed.csv"
        rows = [r for r in csv.DictReader(open(committed, encoding="utf-8")) if r["wav"]]
        base = (ROOT / "out/dataset").resolve()
        csv_path = OUT / "f5_full.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="|")
            w.writerow(["audio_file", "text"])
            for r in rows:
                w.writerow([str(base / r["wav"]), r["text"]])
        self.log(f"f5 csv: {len(rows)} clips")
        if (F5_DATA / "duration.json").exists() and (F5_DATA / "raw.arrow").exists():
            self.log("f5 dataset already prepared, skipping")
            return
        self.run("f5_prepare", [sys.executable,
                                "src/f5_tts/train/datasets/prepare_csv_wavs.py",
                                str(csv_path), str(F5_DATA), "--workers", "8"],
                 F5, deadline - time.time(), env={"PYTHONPATH": str(F5 / "src")})

    def train_f5(self, deadline: float) -> None:
        pretrain = ROOT / "models/f5_pretrained/F5TTS_v1_Base/model_1250000.safetensors"
        last = F5_CKPTS / "model_last.pt"
        if last.exists():
            if self.state["stages"].get("f5_train", {}).get("ok"):
                self.log("f5 training already completed, skipping")
                return
            # a crash left a checkpoint: continue from it rather than the base
            pretrain = last
            self.log(f"resuming f5 training from {last}")
        env = {"PYTHONPATH": str(F5 / "src")}
        self.run("f5_train", [
            sys.executable, "src/f5_tts/train/finetune_cli.py",
            "--exp_name", "F5TTS_v1_Base",
            "--dataset_name", "asmr_zh_full", "--tokenizer", "pinyin",
            "--finetune",
            "--pretrain", str(pretrain),
            "--epochs", "30", "--learning_rate", "1e-5",
            "--batch_size_per_gpu", "3200", "--batch_size_type", "frame",
            "--max_samples", "32", "--save_per_updates", "250",
            "--last_per_updates", "250", "--num_warmup_updates", "500",
        ], F5, deadline - time.time(), env=env, graceful=True)
        # pick the newest checkpoint for the export
        ckpts = sorted(F5_CKPTS.glob("model_*.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        self.state["f5_ckpt"] = str(ckpts[0]) if ckpts else ""
        self.log(f"f5 checkpoint: {self.state.get('f5_ckpt') or 'none'}")

    def train_llm(self, deadline: float) -> None:
        adapter = WEB / "llm_asmr_lora_v2"
        if (adapter / "adapter_model.safetensors").exists():
            self.log("llm adapter exists, skipping training")
            return
        self.run("llm_train", [
            sys.executable, "training/scripts/02_train_llm.py",
            "--data", str(ROOT / "out/dataset/instructions_committed.jsonl"),
            "--base", "models/checkpoints/qwen3.5-0.8b-text",
            "--out", str(adapter),
            "--run-out", str(WEB / "llm_run_v2"),
            "--run-id", "asmr-vv-02", "--no-registry",
            "--epochs", "10", "--samples-every", "1000",
            "--log", "/tmp/llm_asmr_v2.log",
        ], SIBLING, deadline - time.time())

    def export_llm(self, deadline: float) -> None:
        adapter = WEB / "llm_asmr_lora_v2"
        onnx_dir = WEB / "onnx/llm_fp16"
        quant_dir = WEB / "onnx/llm_int4"
        self.run("llm_export", [
            sys.executable, "export/export_onnx_llm.py",
            "--base", "models/checkpoints/qwen3.5-0.8b-text",
            "--adapter", str(adapter), "--out", str(onnx_dir),
            "--fp16", "--verify",
        ], SIBLING, min(deadline, time.time() + 30 * 60) - time.time())
        if onnx_dir.exists() and any(onnx_dir.glob("*.onnx")):
            self.run("llm_quantize", [
                sys.executable, "export/quantize.py", "--mode", "llm",
                "--input", str(onnx_dir), "--out", str(quant_dir), "--verify",
            ], SIBLING, max(120, deadline - time.time()))

    def export_f5(self, deadline: float) -> None:
        ckpt = self.state.get("f5_ckpt") or ""
        if not ckpt or not Path(ckpt).exists():
            self.log("f5 export skipped: no checkpoint")
            return
        onnx_dir = WEB / "onnx/tts_fp16"
        quant_dir = WEB / "onnx/tts_int4"
        self.run("f5_export", [
            sys.executable, "export/export_onnx_f5tts.py",
            "--ckpt", ckpt, "--out", str(onnx_dir), "--fp16",
            "--no-verify",
        ], SIBLING, min(deadline, time.time() + 25 * 60) - time.time())
        if onnx_dir.exists() and any(onnx_dir.glob("*.onnx")):
            self.run("f5_quantize", [
                sys.executable, "export/quantize.py", "--mode", "tts",
                "--input", str(onnx_dir), "--out", str(quant_dir), "--verify",
            ], SIBLING, max(120, deadline - time.time()))

    def steps_sweep(self, deadline: float) -> None:
        """Few-step / no-CFG quality sweep on the finetuned checkpoint.

        Cheap (minutes) and decides whether distillation is needed at all. The
        result is copied under models/onnx/sweep/ so the upload carries it.
        """
        ckpt = self.state.get("f5_ckpt") or ""
        if not ckpt or not Path(ckpt).exists():
            self.log("steps sweep skipped: no f5 checkpoint")
            return
        out_dir = OUT / "sweep"
        self.run("steps_sweep", [
            sys.executable, str(ROOT / "scripts/81_steps_sweep.py"),
            "--ckpt", ckpt, "--out-dir", str(out_dir),
        ], ROOT, deadline - time.time(), env={"PYTHONPATH": str(F5 / "src")})
        if (out_dir / "sweep.json").exists():
            dest = WEB / "onnx/sweep"
            dest.mkdir(parents=True, exist_ok=True)
            for f in out_dir.glob("sweep.*"):
                shutil.copy2(f, dest / f.name)
            self.log(f"sweep results copied to {dest}")

    def distill_4step(self, deadline: float) -> None:
        """Reflow-distil the teacher to 4 steps, CFG-free, then export int4."""
        teacher = self.state.get("f5_ckpt") or ""
        if not teacher or not Path(teacher).exists():
            self.log("distill skipped: no teacher checkpoint")
            return
        student = WEB / "f5tts_4step.pt"
        self.run("distill_4step", [
            sys.executable, str(ROOT / "scripts/82_distill_4step.py"),
            "--teacher", teacher, "--out", str(student),
            "--max-minutes", str(max(10, int((deadline - time.time()) / 60) - 15)),
        ], ROOT, deadline - time.time(), env={"PYTHONPATH": str(F5 / "src")})
        if not student.exists():
            self.log("distill produced no checkpoint")
            return
        onnx = WEB / "onnx/tts4_fp16"
        quant = WEB / "onnx/tts4_int4"
        self.run("distill_export", [
            sys.executable, "export/export_onnx_f5tts.py",
            "--ckpt", str(student), "--out", str(onnx), "--fp16", "--no-verify",
        ], SIBLING, min(deadline, time.time() + 20 * 60) - time.time(),
            env={"PYTHONPATH": str(F5 / "src")})
        if any(onnx.glob("*.onnx")):
            self.run("distill_quantize", [
                sys.executable, "export/quantize.py", "--mode", "tts",
                "--input", str(onnx), "--out", str(quant), "--verify",
            ], SIBLING, max(120, deadline - time.time()))

    def upload(self, deadline: float) -> None:
        from huggingface_hub import HfApi

        api = HfApi()

        def push(repo: str, folder: Path, card: str, message: str) -> None:
            if not folder.exists() or not any(folder.iterdir()):
                self.log(f"upload {repo}: folder missing, skipped")
                return
            api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
            (folder / "README.md").write_text(card, encoding="utf-8")
            self.log(f"upload {repo}: {sum(1 for _ in folder.rglob('*') if _.is_file())} files")
            api.upload_folder(folder_path=str(folder), repo_id=repo,
                              repo_type="model", commit_message=message)

        push("DownFlow/ASMR-llm-lora", WEB / "llm_asmr_lora_v2", LLM_CARD,
             "LLM LoRA adapter (Qwen3.5-0.8B, ASMR)")
        f5_dir = OUT / "f5_release"
        f5_dir.mkdir(exist_ok=True)
        ckpt = self.state.get("f5_ckpt")
        if ckpt and Path(ckpt).exists():
            shutil.copy2(ckpt, f5_dir / "model_last.pt")
        student = WEB / "f5tts_4step.pt"
        if student.exists():
            shutil.copy2(student, f5_dir / "model_4step.pt")
        vocab = F5_DATA / "vocab.txt"
        if vocab.exists():
            shutil.copy2(vocab, f5_dir / "vocab.txt")
        push("DownFlow/ASMR-f5tts", f5_dir, F5_CARD,
             "F5-TTS ASMR checkpoint")

        onnx_root = WEB / "onnx"
        if onnx_root.exists():
            push("DownFlow/ASMR-onnx", onnx_root, ONNX_CARD,
                 "Quantized ONNX graphs for browser inference")
        else:
            self.log("upload onnx: nothing to upload")

    def report(self) -> None:
        lines = ["# Overnight run", "",
                 f"started {time.strftime('%Y-%m-%d %H:%M')}",
                 f"budget {self.state.get('hours')} h, "
                 f"used {self.elapsed_h():.2f} h", ""]
        lines.append("| stage | ok | minutes | log |")
        lines.append("|---|---|---|---|")
        for name, result in self.state["stages"].items():
            lines.append(f"| {name} | {'yes' if result.get('ok') else 'NO'} | "
                         f"{result.get('duration_s', 0) / 60:.1f} | "
                         f"{Path(result.get('log', '')).name} |")
        lines.append("")
        for label, path in [("LLM adapter", WEB / "llm_asmr_lora_v2"),
                            ("F5 checkpoint", Path(self.state.get("f5_ckpt") or "")),
                            ("LLM ONNX", WEB / "onnx/llm_int4"),
                            ("TTS ONNX", WEB / "onnx/tts_int4")]:
            if str(path) and path.exists():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                lines.append(f"- {label}: `{path}` ({size / 1e6:.0f} MB)")
            else:
                lines.append(f"- {label}: missing")
        (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log(f"report written to {OUT / 'report.md'}")

    def cleanup_gpu(self) -> None:
        """No-op: stages kill their own process group on timeout.

        A global pkill here would also kill the other agent's training on this
        shared box, which is not ours to stop.
        """
        return


LLM_CARD = """---
license: apache-2.0
tags:
- asmr
- lora
- text-generation
---

# ASMR LLM director (LoRA)

LoRA adapter for Qwen3.5-0.8B trained to write Chinese ASMR roleplay lines with
`[tone:...]` control tokens. 14,389 instruction rows from the WebASMR segment
corpus (VibeVoice-ASR segments reconciled with Qwen3-ASR and 101 human labels).

Training: 10 epochs, LoRA r=16, explicit prompt masking, EOS appended.
"""

F5_CARD = """---
license: cc-by-nc-4.0
tags:
- tts
- f5-tts
- asmr
---

# F5-TTS ASMR

F5-TTS (F5TTS_v1_Base) finetuned on WebASMR clips: 14,389 clips cut on
utterance boundaries, 24 kHz mono, reconciled transcripts. DiT + vocoder.
`model_last.pt` is the raw trainer checkpoint; `vocab.txt` matches it.
"""

ONNX_CARD = """---
license: apache-2.0
tags:
- onnx
- webgpu
- int4
---

# WebASMR ONNX graphs

Quantized graphs for the browser runtime:

- `llm_int4/` - Qwen3.5-0.8B + ASMR LoRA, merged and exported, MatMulNBits int4
- `tts_int4/` - F5-TTS DiT step + vocoder, int4 MatMul, convs in session dtype

`llm_fp16/` and `tts_fp16/` hold the fp16 exports they were quantized from.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    args = ap.parse_args()

    run = Overnight(args.hours)
    run.log(f"overnight start, budget {args.hours} h")
    cursor = time.time()
    for name, minutes in STAGES:
        if run.remaining_s() < 300:
            run.log(f"deadline near, skipping {name}")
            continue
        stage_deadline = min(cursor + minutes * 60, run.deadline - 300)
        try:
            getattr(run, name)(stage_deadline)
        except Exception as exc:  # noqa: BLE001 - one bad stage must not stop the run
            run.log(f"{name} raised {type(exc).__name__}: {exc}")
            run.state["stages"][name] = {"ok": False,
                                         "error": f"{type(exc).__name__}: {exc}"}
            run.save()
        cursor = time.time()
        run.cleanup_gpu()
    try:
        run.report()
    except Exception as exc:  # noqa: BLE001
        run.log(f"report failed: {exc}")
    run.log("overnight done")


if __name__ == "__main__":
    main()