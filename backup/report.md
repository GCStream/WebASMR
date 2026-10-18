# Overnight run

started 2026-10-17 20:58
budget 8.0 h, used 4.27 h

| stage | ok | minutes | log |
|---|---|---|---|
| f5_prepare | yes | 0.2 | f5_prepare.log |
| f5_train | NO | 210.0 | f5_train.log |
| llm_train | yes | 44.5 | llm_train.log |
| llm_export | NO | 0.1 | llm_export.log |
| f5_export | yes | 1.1 | f5_export.log |
| f5_quantize | yes | 0.1 | f5_quantize.log |

- LLM adapter: `/root/WebASMR/models/llm_asmr_lora_v2` (46 MB)
- F5 checkpoint: `/root/F5-TTS/ckpts/asmr_zh_full/model_91750.pt` (0 MB)
- LLM ONNX: missing
- TTS ONNX: `/root/WebASMR/models/onnx/tts_int4` (415 MB)

## Few-step sweep

checkpoint `/root/F5-TTS/ckpts/asmr_zh_full/model_last.pt`

| steps | cfg | s/clip | CER | mfcc_dtw |
|---|---|---|---|---|
| 32 | 2 | 0.7 | 0.0972 | - |
| 16 | 2 | 0.25 | 0.0972 | 2.6812 |
| 8 | 2 | 0.14 | 0.2803 | 68.9008 |
| 8 | 1 | 0.13 | 0.303 | 30.867 |
| 8 | 0 | 0.14 | 0.5821 | 49.7444 |
| 4 | 2 | 0.07 | 0.6831 | 92.5959 |
| 4 | 0 | 0.08 | 0.7753 | 62.964 |

Listen to `out/overnight/sweep/*.wav` before choosing; few-step flow models lose breath and whisper texture before they lose words.

## Few-step sweep

checkpoint `/root/WebASMR/models/f5tts_4step.pt`

| steps | cfg | s/clip | CER | mfcc_dtw |
|---|---|---|---|---|
| 4 | 0 | 0.11 | 0.5972 | - |
| 8 | 0 | 0.13 | 0.5581 | - |

Listen to `out/overnight/sweep/*.wav` before choosing; few-step flow models lose breath and whisper texture before they lose words.

## 4-step distillation v2 (2,000 clips, 90 min)

| steps | cfg | s/clip | CER | mfcc_dtw |
|---|---|---|---|---|
| 32 | 2 | 0.59 | 0.1111 | None |
| 16 | 2 | 0.25 | 0.1111 | 3.1836 |
| 8 | 0 | 0.14 | 0.0278 | 38.453 |
| 4 | 0 | 0.07 | 0.1515 | 49.7449 |
