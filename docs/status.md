# Status at shutdown (2026-10-18 03:05 UTC, 40 h uptime)

## Published

- `DownFlow/ASMR` - 19,803 clips, Qwen3-ASR transcripts, upstream schema (16.3 GB)
- `DownFlow/ASMR-llm-lora` - Qwen3.5-0.8B LoRA, 10 epochs
- `DownFlow/ASMR-f5tts` - teacher F5-TTS, 21 epochs / 91,750 updates (5.4 GB)
- `DownFlow/ASMR-f5tts-4step` - v1 4-step student. **Invalid**: trained before
  the pinyin-tokenisation fix, so text conditioning was all zeros. v2 replaces
  it if the emergency upload completed; otherwise rerun
  `scripts/83_distill_v2.py`.
- `DownFlow/ASMR-onnx` - teacher fp16/int4, v1 student fp16/int4, sweep results

## Open items

- LLM ONNX export is not implemented upstream (`export_onnx_llm.py` is a spike,
  sibling TODO-23), so only the adapter is published.
- Dataset viewer is disabled by Hugging Face for the `nsfw`-tagged dataset.
- Web worker prompt bug: `tts.worker.js` does not concatenate ref text, does
  not size output as ref + target, and does not slice the reference region
  before vocoding. Long text also gets squeezed by the `min(ref.frames, 8*96)`
  cap. Our `85_quant_ab.py` shows the correct recipe.
- Web app `tuning.json` sets `flowSteps: 32`, which overrides the manifest's
  16-step recommendation.
- 4-step distillation v2 (fixed pinyin) was running; see
  `out/overnight/logs/distill_v2.log` and `out/overnight/emergency_upload.log`.
