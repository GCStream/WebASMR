# Status at shutdown (2026-10-18 03:05 UTC, 40 h uptime)

## Published

- `DownFlow/ASMR` - 19,803 clips, Qwen3-ASR transcripts, upstream schema (16.3 GB)
- `DownFlow/ASMR-llm-lora` - Qwen3.5-0.8B LoRA, 10 epochs
- `DownFlow/ASMR-f5tts` - teacher F5-TTS, 21 epochs / 91,750 updates (5.4 GB)
- `DownFlow/ASMR-f5tts-4step` - v2 4-step student (fixed pinyin conditioning),
  `model_4step_v2.pt`. The v1 `model_4step.pt` is invalid (trained with all-zero
  text tokens); ignore/delete it.
- `DownFlow/ASMR-onnx` - teacher fp16/int4; student v2 fp16/int4 with MatMulNBits
  block 16 as the deployed `tts4v2_int4` (mean rel error 0.133 vs 0.159 block 32
  and 0.185 block 64), plus `tts4v2_int4_b32` as an alternative; sweep results

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
- 4-step v2 distillation completed: 4,984 steps / 90 min. Sweep on the student
  (same 莼儿 prompt, tongue twisters): 32:2 CER 0.111, 16:2 0.111, **8:0 0.028**,
  **4:0 0.152**. The pinyin fix moved 4:0 from 0.597 to 0.152; 8:0 is the
  strongest student config. MFCC-DTW vs the teacher (38-50) means the waveform
  differs as expected from distillation; judge by listening.
