#!/usr/bin/env python3
"""Summarize a benchmark file into a per-system comparison table.

    python tools/summarize_bench.py out/bench_vibevoice.jsonl
    python tools/summarize_bench.py out/bench_vibevoice.jsonl --samples
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.segments import repetition_ratio, vocalization_ratio  # noqa: E402


def mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def fmt(value):
    return f"{value:>13}" if value is not None else f"{'-':>13}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path)
    ap.add_argument("--samples", action="store_true", help="print side-by-side text")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.bench, encoding="utf-8") if l.strip()]
    systems = ["vibevoice"] + sorted({n for r in rows for n in r.get("baselines", {})})

    print(f"{len(rows)} files, "
          f"{sum(r['window_s'] for r in rows) / 60:.0f} min of audio compared\n")

    header = f"{'metric':<22}" + "".join(f"{s:>13}" for s in systems)
    print(header)
    print("-" * len(header))

    def col(fn):
        return [fn(r, s) for r in rows for s in [None] if s is None] if False else None

    metrics = {
        "chars/window": lambda r, s: (r["vibevoice"]["chars"] if s == "vibevoice"
                                      else r["baselines"][s]["chars"]),
        "units (segs/clips)": lambda r, s: (r["vibevoice"]["units"] if s == "vibevoice"
                                            else r["baselines"][s]["units"]),
        "chars/unit": lambda r, s: (round(r["vibevoice"]["chars"] / max(1, r["vibevoice"]["units"]), 1)
                                    if s == "vibevoice" else
                                    round(r["baselines"][s]["chars"] / max(1, r["baselines"][s]["units"]), 1)),
        "repetition": lambda r, s: (r["vibevoice"]["repetition"] if s == "vibevoice"
                                    else r["baselines"][s]["repetition"]),
        "vocalization": lambda r, s: (r["vibevoice"]["vocalization"] if s == "vibevoice"
                                      else r["baselines"][s]["vocalization"]),
        "coverage": lambda r, s: (r["vibevoice"]["coverage"] if s == "vibevoice"
                                  else r["baselines"][s]["coverage"]),
        "cut_rate": lambda r, s: (r["vibevoice"]["cut_rate"] if s == "vibevoice"
                                  else r["baselines"][s]["cut_rate"]),
        "agreement": lambda r, s: (None if s == "vibevoice"
                                   else r["baselines"][s]["agreement_with_vibevoice"]),
    }
    for name, fn in metrics.items():
        cells = "".join(fmt(mean([fn(r, s) for r in rows if s in r.get("baselines", {})
                                  or s == "vibevoice"])) for s in systems)
        print(f"{name:<22}{cells}")

    for label, fn in [("speakers found", lambda r: r["vibevoice"]["speakers"]),
                      ("wall clock (s)", lambda r: r["elapsed_s"]),
                      ("realtime factor", lambda r: r["realtime_factor"])]:
        cells = "".join(fmt(mean([fn(r) for r in rows])) if s == "vibevoice" else fmt(None)
                        for s in systems)
        print(f"{label:<22}{cells}")

    statuses = {}
    for r in rows:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
    print(f"\nparse status: {statuses}")

    looped = [r for r in rows if repetition_ratio(r["vibevoice"]["text"]) > 0.6]
    print(f"windows with runaway text: {len(looped)}")

    emptyish = [r for r in rows
                if vocalization_ratio(r["vibevoice"]["text"]) > 0.5]
    print(f"windows that are mostly vocalization: {len(emptyish)}")

    print("\nper file:")
    for r in rows:
        vv = r["vibevoice"]
        print(f"  [{r.get('status','?'):>9}] {r['key'][:46]:<48} "
              f"segs={vv['units']:>3} spk={vv['speakers']} chars={vv['chars']:>5} "
              f"rep={vv['repetition']:.2f} cut={vv['cut_rate']:.2f} "
              f"{r['realtime_factor']:.1f}x")

    if args.samples:
        print("\n" + "=" * 100)
        for r in rows:
            print(f"\n### {r['key']}  ({r['duration_s'] / 60:.1f} min, "
                  f"window {r['window_s'] / 60:.1f} min, status {r.get('status')})")
            print(f"  vibevoice ({r['vibevoice']['units']} segments, "
                  f"{r['vibevoice']['speakers']} speakers):\n    {r['vibevoice']['text'][:700]}")
            for n, b in r.get("baselines", {}).items():
                print(f"  {n} ({b['units']} clips, agreement "
                      f"{b['agreement_with_vibevoice']}):\n    {b['text'][:700]}")


if __name__ == "__main__":
    main()