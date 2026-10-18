# Few-step F5-TTS: sweep and a 4-step distillation

Measured with `scripts/81_steps_sweep.py` on the fine-tuned checkpoint
(`model_91750.pt`, 21 epochs on 33.9 h). Latency is torch on an idle MI300X;
the browser runs int4 ONNX, so absolute numbers differ, but the step/CFG
scaling carries over. CER is Whisper-turbo on the generated audio against the
requested text; MFCC-DTW is distance to the 32-step/CFG-2 output for the same
prompt, text and seed.

## Teacher (plain fine-tune)

| steps | cfg | forwards/clip | s/clip | CER | mfcc_dtw |
|---|---|---|---|---|---|
| 32 | 2 | 64 | 0.70 | 0.0972 | ref |
| **16** | **2** | **32** | **0.25** | **0.0972** | **2.68** |
| 8 | 2 | 16 | 0.14 | 0.2803 | 68.90 |
| 8 | 1 | 12 | 0.13 | 0.3030 | 30.87 |
| 8 | 0 | 8 | 0.14 | 0.5821 | 49.74 |
| 4 | 2 | 8 | 0.07 | 0.6831 | 92.60 |
| 4 | 0 | 4 | 0.08 | 0.7753 | 62.96 |

**16 steps at CFG 2 is free.** Identical CER, MFCC-DTW 2.68 (essentially the
same waveform family), 2.8x fewer forwards than the current default. This
should be the shipped default now.

8 steps is where quality breaks: CER roughly triples. Dropping CFG makes it
worse, so guidance is doing real work at this step count.

## 4-step reflow student

`scripts/82_distill_4step.py`: teacher 32-step/CFG-2 → student 4-step/CFG-free,
2,509 updates in 45 min on 800 clips (loss 0.276).

| steps | cfg | s/clip | CER |
|---|---|---|---|
| 4 | 0 | 0.11 | 0.5972 |
| 8 | 0 | 0.13 | 0.5581 |

The distillation **improved 4:0 from 0.775 to 0.597 CER** (23% relative), but
that is still far from the 32-step 0.097. The student at 8:0 (0.558) is also
worse than the teacher at 8:2 (0.280): 45 minutes and 800 clips is not enough
to absorb guidance for a 4x trajectory jump.

MFCC-DTW is missing for the student because the 4:0 sweep did not include a
32:2 reference run; add it if a fidelity number is needed.

## Recommendation

- Ship **16 steps, CFG 2** now. 2.8x DiT reduction at no measured quality cost.
- Keep 8:2 as the aggressive option; expect audible degradation.
- 4:0 needs a longer distillation: more updates, more clips, and possibly
  DMD-style adversarial loss. The student is available at
  `DownFlow/ASMR-f5tts-4step` for experimentation, not as the default.

Artifacts: `DownFlow/ASMR-f5tts` (teacher), `DownFlow/ASMR-f5tts-4step`
(student), `DownFlow/ASMR-onnx` (`tts_fp16`, `tts_int4`, `tts4_fp16`,
`tts4_int4`, `sweep/`).
