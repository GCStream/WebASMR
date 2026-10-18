#!/usr/bin/env python3
"""Base vs fine-tuned voice comparison with the 莼儿 reference.

Same reference mel, text, sampler settings and noise seed for both models, so
the only difference is the weights. Used by the /review/finetune page.
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import importlib

q = importlib.import_module("85_quant_ab")

OUT = ROOT / "out/voice_ft"
MODELS = {
    "base": ROOT / "models/onnx/base_fp16",
    "finetuned": ROOT / "models/onnx/tts_fp16",
}


def main() -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    def load(d: Path) -> dict:
        return {"dit": ort.InferenceSession(str(d / "dit_step.onnx"), so,
                                            providers=["CPUExecutionProvider"]),
                "voc": ort.InferenceSession(str(d / "vocoder.onnx"), so,
                                            providers=["CPUExecutionProvider"])}

    sessions = {name: load(d) for name, d in MODELS.items()}
    ref, meta = q.ref_mel()
    OUT.mkdir(parents=True, exist_ok=True)

    items = []
    for i, text in enumerate(q.TEXTS):
        ref_bytes = len(meta["ref_text"].encode("utf-8"))
        gen_bytes = len(text.encode("utf-8"))
        target_frames = int(ref.shape[0] / ref_bytes * gen_bytes)
        frames = ref.shape[0] + max(target_frames, 64)
        ids = q.text_ids(meta["ref_text"] + text)
        item_id = f"f{i:02d}"
        durations = {}
        for variant, sess in sessions.items():
            wav = q.synth({"fp16": sess}, "fp16", ref, ids, frames, 16, 2.0,
                          -1.0, seed=1234 + i, ref_frames=ref.shape[0])
            sf.write(str(OUT / f"{item_id}_{variant}.wav"), wav, 24000,
                     subtype="PCM_16")
            durations[variant] = round(len(wav) / 24000, 2)
            print(f"  {item_id} {variant}: {durations[variant]}s", flush=True)
        items.append({"id": item_id, "text": text, "voice": meta["name_zh"],
                      "ref_text": meta["ref_text"], "steps": 16, "cfg": 2.0,
                      "seconds": durations})
    (OUT / "index.json").write_text(
        json.dumps({"voice": meta["name_zh"], "items": items}, indent=1,
                   ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}/index.json")


if __name__ == "__main__":
    main()