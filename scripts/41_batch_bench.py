#!/usr/bin/env python3
"""Find the batch size that maximizes throughput inside one worker.

`40_concurrency_bench.py` showed that adding worker processes saturates, because
batch-1 decode streams the full 16.7 GB of weights from HBM every step. This
script asks the other half of the question: how many clips per `generate()`
call before padding and memory stop paying off.

Every configuration transcribes the identical file set, and the reported rate
uses the summed per-batch generation times so model load is excluded.

    python scripts/41_batch_bench.py --files 6 --band 300 420 --sizes 1 2 3 4 6 8
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import probe_duration_s  # noqa: E402
from common.keys import iter_audio_files, source_key  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "30_transcribe.py"
BATCH_TIME = re.compile(r"batch of \d+ took (\d+)s")


def workload(audio_root: Path, n_files: int, band: tuple[float, float], path: Path):
    rows = [(probe_duration_s(p), source_key(p, audio_root))
            for p in iter_audio_files(audio_root)]
    lo, hi = band
    pool = sorted((d, k) for d, k in rows if lo <= d < hi)
    step = max(1, len(pool) // n_files)
    picked = pool[::step][:n_files]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([k for _, k in picked], ensure_ascii=False), encoding="utf-8")
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", type=Path, default=Path("/root/data_raw/audio"))
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--band", type=float, nargs=2, default=[300.0, 420.0],
                    metavar=("LO", "HI"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--out-dir", type=Path, default=Path("out/batchbench"))
    args = ap.parse_args()

    keys_file = args.out_dir / "workload_keys.json"
    picked = workload(args.audio_root, args.files, tuple(args.band), keys_file)
    audio_h = sum(d for d, _ in picked) / 3600
    print(f"workload: {len(picked)} files, {audio_h * 60:.0f} min of audio")
    for d, k in picked:
        print(f"  {d / 60:5.1f} min  {k}")
    print(flush=True)

    results = []
    for size in args.sizes:
        out_path = args.out_dir / f"b{size:02d}.jsonl"
        if out_path.exists():
            out_path.unlink()
        cmd = [sys.executable, str(SCRIPT), "--keys-file", str(keys_file),
               "--out", str(out_path), "--batch-size", str(size),
               "--batch-max-audio-s", "3600", "--audio-root", str(args.audio_root)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        gen_s = sum(float(m) for m in BATCH_TIME.findall(proc.stdout))
        rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()] \
            if out_path.exists() else []
        rate = audio_h / (gen_s / 3600) if gen_s else 0.0
        results.append({"batch": size, "wall_s": round(wall, 1), "gen_s": round(gen_s, 1),
                        "rows": len(rows), "audio_h_per_h": round(rate, 2)})
        print(f"batch={size:>2}  wall={wall:6.0f}s  generate={gen_s:6.0f}s  "
              f"{rate:6.2f} audio-h/h  rows={len(rows)}", flush=True)
        if proc.returncode != 0:
            print(proc.stdout[-800:], proc.stderr[-800:])

    base = next((r["audio_h_per_h"] for r in results if r["batch"] == 1), None)
    print(f"\n{'batch':>7}{'gen_s':>9}{'audio-h/h':>11}{'vs batch=1':>13}")
    for r in results:
        rel = f"{r['audio_h_per_h'] / base:.2f}x" if base else "-"
        print(f"{r['batch']:>7}{r['gen_s']:>9.0f}{r['audio_h_per_h']:>11.2f}{rel:>13}")
    best = max(results, key=lambda r: r["audio_h_per_h"])
    print(f"\nbest batch: {best['batch']} at {best['audio_h_per_h']:.2f} audio-h/h "
          f"(processes were 11.7 at batch 1, 18.8 with 2 workers)")
    Path(args.out_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()