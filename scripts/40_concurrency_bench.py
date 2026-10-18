#!/usr/bin/env python3
"""Find the worker count that maximizes corpus throughput on one GPU.

A VibeVoice pass is batch-1 decoding: one sequence, one token at a time, over
a 7B model. That underutilizes an MI300X, so running several independent
worker processes on the same card overlaps their small kernels and should
raise aggregate throughput well past what VRAM alone would suggest. The knee
is where added workers stop buying throughput, and it is an empirical question:
this script runs the same fixed workload at several worker counts and reports
audio-hours per wall-clock hour.

    python scripts/40_concurrency_bench.py --files 12 --band 300 900
    python scripts/40_concurrency_bench.py --workers 1 2 4 --files 12
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import probe_duration_s  # noqa: E402
from common.jsonlio import write_jsonl  # noqa: E402
from common.keys import iter_audio_files, source_key  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "30_transcribe.py"


def gpu_used_bytes() -> int:
    """VRAM in use across all cards, or -1 if rocm-smi is unavailable."""
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                             capture_output=True, text=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return -1
    header = [h.strip() for h in out.splitlines()[0].split(",")] if out else []
    if "VRAM Total Used Memory (B)" not in header:
        return -1
    col = header.index("VRAM Total Used Memory (B)")
    total = 0
    for line in out.splitlines()[1:]:
        cells = [c.strip() for c in line.split(",")]
        if len(cells) > col and cells[col].isdigit():
            total += int(cells[col])
    return total or -1


def build_workload(audio_root: Path, n_files: int, band: tuple[float, float],
                   out_path: Path) -> tuple[list[str], float]:
    rows = [(probe_duration_s(p), source_key(p, audio_root))
            for p in iter_audio_files(audio_root)]
    lo, hi = band
    pool = sorted((d, k) for d, k in rows if lo <= d < hi)
    step = max(1, len(pool) // n_files)
    picked = pool[::step][:n_files]
    keys = [k for _, k in picked]
    audio_h = sum(d for d, _ in picked) / 3600
    write_jsonl(out_path, [{"key": k} for k in keys])
    Path(out_path).write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
    return keys, audio_h


def run_config(workers: int, keys_file: Path, out_dir: Path, args) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for w in range(workers):
        out_path = out_dir / f"segments.w{w:02d}.jsonl"
        if out_path.exists():
            out_path.unlink()
        log = out_dir / f"w{w:02d}.log"
        logs.append(log.open("w"))
        cmd = [sys.executable, str(SCRIPT),
               "--keys-file", str(keys_file),
               "--out", str(out_path),
               "--audio-root", str(args.audio_root),
               "--prompt", args.prompt] if args.prompt else [
               sys.executable, str(SCRIPT),
               "--keys-file", str(keys_file),
               "--out", str(out_path),
               "--audio-root", str(args.audio_root)]
        procs.append(subprocess.Popen(cmd, stdout=logs[-1], stderr=subprocess.STDOUT))

    peak = 0
    start = time.time()
    while any(p.poll() is None for p in procs):
        peak = max(peak, gpu_used_bytes())
        time.sleep(2)
    wall = time.time() - start
    for p in procs:
        if p.returncode != 0:
            print(f"  worker exited {p.returncode}; see {out_dir}")
    for log in logs:
        log.close()

    rows = []
    for w in range(workers):
        path = out_dir / f"segments.w{w:02d}.jsonl"
        if path.exists():
            rows += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    return {
        "workers": workers,
        "wall_s": round(wall, 1),
        "rows": len(rows),
        "peak_vram_gb": round(peak / 1e9, 1) if peak > 0 else None,
        "failures": sum(1 for r in rows if r.get("status") in ("no_segments", "decode_failed", "partial")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", type=Path, default=Path("/root/data_raw/audio"))
    ap.add_argument("--files", type=int, default=12)
    ap.add_argument("--band", type=float, nargs=2, default=[300.0, 900.0],
                    metavar=("LO", "HI"), help="duration band in seconds")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", type=Path, default=Path("out/concurrency"))
    args = ap.parse_args()

    keys_file = args.out / "workload_keys.json"
    keys, audio_h = build_workload(args.audio_root, args.files, tuple(args.band), keys_file)
    print(f"workload: {len(keys)} files, {audio_h * 60:.0f} min of audio "
          f"({args.band[0]:.0f}-{args.band[1]:.0f}s band)")
    print(f"written to {keys_file}\n", flush=True)

    results = []
    for workers in args.workers:
        out_dir = args.out / f"w{workers:02d}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        print(f"--- {workers} worker(s) ---", flush=True)
        res = run_config(workers, keys_file, out_dir, args)
        done_h = sum(r.get("duration_s", 0.0) for r in rows) / 3600
        res["audio_h"] = round(audio_h, 3)
        res["audio_h_done"] = round(done_h, 3)
        res["audio_h_per_h"] = round(done_h / (res["wall_s"] / 3600), 2)
        res["files_per_h"] = round(res["rows"] / (res["wall_s"] / 3600), 2)
        results.append(res)
        print(f"  {res['wall_s']:.0f}s  {res['audio_h_per_h']:.1f} audio-h/h  "
              f"({done_h:.2f} h transcribed by {workers} workers)  "
              f"rows={res['rows']}  peak_vram={res['peak_vram_gb']} GB  "
              f"failures={res['failures']}", flush=True)

    base = next((r["audio_h_per_h"] for r in results if r["workers"] == 1), None)
    print(f"\n{'workers':>8}{'wall_s':>10}{'audio-h/h':>11}{'speedup':>9}"
          f"{'vram_gb':>9}{'per-worker':>12}")
    for r in results:
        speedup = f"{r['audio_h_per_h'] / base:.2f}x" if base else "-"
        print(f"{r['workers']:>8}{r['wall_s']:>10.0f}{r['audio_h_per_h']:>11.2f}"
              f"{speedup:>9}{str(r['peak_vram_gb']):>9}"
              f"{r['audio_h_per_h'] / r['workers']:>12.2f}")
    print("\nnote: each worker transcribes the whole workload, so audio-h/h here is\n"
          "aggregate across workers, not per worker.")

    best = max(results, key=lambda r: r["audio_h_per_h"])
    print(f"\nsweet spot: {best['workers']} workers at {best['audio_h_per_h']:.2f} audio-h/h")
    print(f"corpus projection (131 h): {131 / best['audio_h_per_h']:.1f} h")
    write_jsonl(args.out / "results.jsonl", results)


if __name__ == "__main__":
    main()