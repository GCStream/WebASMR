#!/usr/bin/env python3
"""Upload the v2 student as soon as it exists, before the 40 h uptime shutdown.

The distillation wrapper exports, sweeps and uploads in that order; with ~20
minutes left the sweep could push the upload past the shutdown. This waits for
the checkpoint, uploads it, then uploads the ONNX graphs when they appear.
"""

import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDENT = ROOT / "models/f5tts_4step_v2.pt"
ONNX = ROOT / "models/onnx/tts4v2_int4"
DEADLINE = time.time() + 45 * 60


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "out/overnight/emergency_upload.log", "a") as f:
        f.write(line + "\n")


def wait_for(path: Path, stable=5.0):
    seen_size = -1
    while time.time() < DEADLINE:
        if path.exists():
            size = path.stat().st_size
            if size == seen_size and size > 0:
                return True
            seen_size = size
            time.sleep(stable)
        else:
            time.sleep(3)
    return False


def main():
    from huggingface_hub import HfApi

    api = HfApi()
    log("waiting for v2 student checkpoint")
    if not wait_for(STUDENT):
        log("checkpoint never appeared")
        return
    release = ROOT / "out/overnight/f5_4step_v2"
    release.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STUDENT, release / "model_4step_v2.pt")
    vocab = Path("/root/F5-TTS/data/asmr_zh_full_pinyin/vocab.txt")
    if vocab.exists():
        shutil.copy2(vocab, release / "vocab.txt")
    (release / "README.md").write_text(
        "---\nlicense: cc-by-nc-4.0\ntags:\n- tts\n- f5-tts\n- asmr\n"
        "- few-step\n---\n\n# F5-TTS ASMR, 4-step CFG-free (v2)\n\n"
        "Reflow-distilled from the fine-tuned 32-step teacher with pinyin text "
        "conditioning fixed. Use 4 Euler steps, sway -1.0, cfg 0.\n", encoding="utf-8")
    log(f"uploading checkpoint {STUDENT.stat().st_size / 1e9:.2f} GB")
    api.upload_folder(folder_path=str(release), repo_id="DownFlow/ASMR-f5tts-4step",
                      repo_type="model", commit_message="v2 4-step distillation")
    log("checkpoint uploaded")

    log("waiting for int4 student graphs")
    if wait_for(ONNX / "dit_step.onnx", stable=3.0) and (ONNX / "vocoder.onnx").exists():
        api.upload_folder(folder_path=str(ONNX), path_in_repo="tts4v2_int4",
                          repo_id="DownFlow/ASMR-onnx", repo_type="model",
                          commit_message="v2 4-step int4 graphs")
        for extra in ("manifest.json", "quantization.json"):
            p = ONNX / extra
            if p.exists():
                api.upload_file(path_or_fileobj=str(p),
                                path_in_repo=f"tts4v2_int4/{extra}",
                                repo_id="DownFlow/ASMR-onnx", repo_type="model",
                                commit_message="v2 4-step manifest")
        log("int4 graphs uploaded")
    else:
        log("graphs not ready before deadline")
    log("done")


if __name__ == "__main__":
    main()