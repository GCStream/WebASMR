#!/usr/bin/env python3
"""Head-to-head: worker processes vs clips-per-generate() call.

The two mechanisms address the same bottleneck from opposite sides:

  processes  more independent model replicas, each streaming the full 16.7 GB
             of weights per decode step, so the traffic is duplicated
  batch      more sequences per step inside one replica, so one weight read
             serves all of them

`40_concurrency_bench.py` measured the first and `41_batch_bench.py` the
second, but on different workloads and under different GPU load, so their
numbers cannot be compared. This runs both dimensions over one fixed workload,
back to back, in a single table.

Every worker gets the identical file list (steady-state throughput, and the
same methodology as `40_concurrency_bench.py`), so the audio transcribed per
configuration is workers x workload. Because this box is shared, the GPU state
is sampled before and after each configuration and recorded: a configuration
whose samples show a foreign process is marked contaminated and should not be
compared against a clean one.

    python scripts/42_sweep_bench.py --files 6 --band 300 420 --configs 1x1 2x1 4x1 1x4 2x4 4x4
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
from common.jsonlio import write_jsonl  # noqa: E402
from common.keys import iter_audio_files, source_key  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "30_transcribe.py"
BATCH_TIME = re.compile(r"batch of \d+ took (\d+)s")


def gpu_snapshot(own_pids: set[int]) -> dict:
    try:
        out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                             text=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return {"foreign_procs": None}
    pids, vram = [], 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        for cell in parts[2:]:
            if cell.isdigit() and int(cell) > 1_000_000:
                vram = max(vram, int(cell))
        if pid not in own_pids and "python" in line:
            pids.append(pid)
    return {"foreign_procs": pids, "vram_gb": round(vram / 1e9, 1)}


def gpu_peak(own_pids: set[int]) -> int:
    try:
        out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True,
                             text=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return 0
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        for cell in parts[2:]:
            if cell.isdigit() and int(cell) > 1_000_000:
                total += int(cell)
    return total


def build_workload(audio_root: Path, n_files: int, band, path: Path):
    rows = [(probe_duration_s(p), source_key(p, audio_root))
            for p in iter_audio_files(audio_root)]
    lo, hi = band
    pool = sorted((d, k) for d, k in rows if lo <= d < hi)
    step = max(1, len(pool) // n_files)
    picked = pool[::step][:n_files]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([k for _, k in picked], ensure_ascii=False), encoding="utf-8")
    return picked


def run_config(workers: int, batch: int, keys_file: Path, out_dir: Path,
               audio_root: Path, batch_max_audio_s: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = gpu_snapshot(set())
    procs, logs = [], []
    for w in range(workers):
        out_path = out_dir / f"w{w:02d}.jsonl"
        if out_path.exists():
            out_path.unlink()
        log_path = out_dir / f"w{w:02d}.log"
        logs.append(log_path.open("w"))
        procs.append(subprocess.Popen(
            [sys.executable, str(SCRIPT), "--keys-file", str(keys_file),
             "--out", str(out_path), "--batch-size", str(batch),
             "--batch-max-audio-s", str(batch_max_audio_s),
             "--audio-root", str(audio_root)],
            stdout=logs[-1], stderr=subprocess.STDOUT))
    own = {p.pid for p in procs}

    peak = 0
    start = time.time()
    while any(p.poll() is None for p in procs):
        peak = max(peak, gpu_peak(own))
        time.sleep(2)
    wall = time.time() - start
    for log in logs:
        log.close()

    gen_s, rows, segs = 0.0, 0, 0
    for w in range(workers):
        text = (out_dir / f"w{w:02d}.log").read_text(errors="ignore")
        gen_s += sum(float(m) for m in BATCH_TIME.findall(text))
        path = out_dir / f"w{w:02d}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows += 1
                    segs += len(row.get("segments", []))

    after = gpu_snapshot(set())
    foreign = sorted(set(before["foreign_procs"] or []) | set(after["foreign_procs"] or []))
    return {
        "workers": workers,
        "batch": batch,
        "wall_s": round(wall, 1),
        "gen_s": round(gen_s, 1),
        "rows": rows,
        "segments": segs,
        "peak_vram_gb": round(peak / 1e9, 1),
        "foreign_pids": foreign,
        "clean": not foreign,
        "worker_failed": sum(1 for p in procs if p.returncode != 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", type=Path, default=Path("/root/data_raw/audio"))
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--band", type=float, nargs=2, default=[300.0, 420.0],
                    metavar=("LO", "HI"))
    ap.add_argument("--configs", nargs="+", default=["1x1", "2x1", "4x1", "1x4", "2x4", "4x4"],
                    help="WxB: workers x batch size")
    ap.add_argument("--batch-max-audio-s", type=float, default=3600.0)
    ap.add_argument("--out", type=Path, default=Path("out/sweep"))
    args = ap.parse_args()

    keys_file = args.out / "workload_keys.json"
    picked = build_workload(args.audio_root, args.files, tuple(args.band), keys_file)
    audio_h = sum(d for d, _ in picked) / 3600
    print(f"workload: {len(picked)} files, {audio_h * 60:.0f} min of audio "
          f"({args.band[0]:.0f}-{args.band[1]:.0f}s)")
    for d, k in picked:
        print(f"  {d / 60:5.1f} min  {k}")
    print(flush=True)

    results = []
    for spec in args.configs:
        workers, batch = (int(x) for x in spec.lower().split("x"))
        res = run_config(workers, batch, keys_file, args.out / f"{workers}x{batch}",
                         args.audio_root, args.batch_max_audio_s)
        done_h = audio_h * workers
        res["audio_h_per_h"] = round(done_h / (res["wall_s"] / 3600), 2)
        res["config"] = spec
        results.append(res)
        tag = "" if res["clean"] else f"  CONTAMINATED by pids {res['foreign_pids']}"
        print(f"{spec:>6}  wall={res['wall_s']:6.0f}s  gen={res['gen_s']:6.0f}s  "
              f"{res['audio_h_per_h']:6.2f} audio-h/h  peak_vram={res['peak_vram_gb']:5.1f}GB  "
              f"rows={res['rows']}{tag}", flush=True)

    clean = [r for r in results if r["clean"]]
    base = next((r["audio_h_per_h"] for r in clean if r["workers"] == 1 and r["batch"] == 1), None)
    print(f"\n{'config':>8}{'wall_s':>9}{'audio-h/h':>11}{'vs 1x1':>9}{'vram_gb':>9}"
          f"{'segs':>8}{'clean':>7}")
    for r in results:
        rel = f"{r['audio_h_per_h'] / base:.2f}x" if base and r["clean"] else "-"
        print(f"{r['config']:>8}{r['wall_s']:>9.0f}{r['audio_h_per_h']:>11.2f}{rel:>9}"
              f"{r['peak_vram_gb']:>9}{r['segments']:>8}{str(r['clean']):>7}")
    if clean:
        best = max(clean, key=lambda r: r["audio_h_per_h"])
        print(f"\nbest clean config: {best['config']} at {best['audio_h_per_h']:.2f} audio-h/h")
    write_jsonl(args.out / "results.jsonl", results)


if __name__ == "__main__":
    main()