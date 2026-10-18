#!/usr/bin/env python3
"""Print the raw generated text for one file or window, unparsed.

Used to diagnose oddities the segment parser hides: an empty `[]` result, a
truncated array, or a degenerate loop.

    python scripts/11_raw_output.py --audio <file> [--start-s 0] [--duration-s 60]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.audio import load_24k_mono, probe_duration_s  # noqa: E402
from importlib import import_module  # noqa: E402

probe = import_module("10_probe_model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--chars", type=int, default=4000)
    args = ap.parse_args()

    path = Path(args.audio)
    total = probe_duration_s(path)
    audio = load_24k_mono(path)
    start = int(args.start_s * 24000)
    end = start + int(args.duration_s * 24000) if args.duration_s else len(audio)
    audio = audio[start:end]
    dur_s = len(audio) / 24000
    print(f"{path.name}  file={total / 60:.2f} min  window={dur_s / 60:.2f} min "
          f"@ {args.start_s:.0f}s", flush=True)

    processor, model = probe.load_model()
    inputs = processor.apply_transcription_request(audio=[audio], prompt=args.prompt)
    dtype = next(model.parameters()).dtype
    inputs = {k: (v.to("cuda", dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
                  else v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    raw = processor.decode(ids[:, n_in:], return_format="raw", skip_special_tokens=True)
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    print(f"\ninput_tokens={n_in} generated={ids.shape[1] - n_in}")
    print("\n---- raw ----\n" + str(raw)[:args.chars])


if __name__ == "__main__":
    main()