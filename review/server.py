#!/usr/bin/env python3
"""Segment-level accuracy review console.

Purpose: a human listens to one clip, reads the transcript, and labels accuracy.
Those labels are the only ground truth this project can get, which is what turns
"recall and loop-rate look reasonable" into a measurable accuracy number.

Design choices that matter for throughput of review:

  * the whole index is sent once (~5k light rows), so next/previous and progress
    are instant and filtering is local; full text and the baseline transcript are
    fetched per item
  * labels auto-save on click, with keyboard shortcuts and auto-advance, because
    a review pass is thousands of decisions and every extra click costs minutes
  * the Qwen3-ASR transcript for the same time window is shown beside ours, as a
    second opinion: where the two agree the reviewer can label quickly, where
    they disagree the reviewer's attention is worth spending
  * the audio is fetched with auth and decoded client-side, so the waveform and
    the player both work without exposing the secret in an <audio src> attribute
  * labels are append-only JSONL, last write wins per item, so a restart or a
    killed browser never loses a decision

Run (bound to loopback, mounted at /review/ by Caddy):

    QC_SECRET=$(cat /root/.qc_secret) python3 review/server.py --port 8141
"""

import argparse
import csv
import hashlib
import hmac
import io
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
METADATA = ROOT / "out/dataset/metadata.csv"
BASELINE = Path("/root/LocalASMR/asmr-bilingual-webml/data/processed/"
                "zh_retranscribe/transcripts_qwen3asr.jsonl")
LABELS = ROOT / "out/review/labels.jsonl"
AUDIO_ROOT = Path("/root/data_raw/audio")
CONTEXT_CACHE = ROOT / "out/review/ctx"
CONTEXT_PAD_S = 1.0

ACCURACY = ("correct", "minor", "major", "unusable")
ERROR_TYPES = ("wrong_words", "missing_words", "hallucinated", "bad_boundary",
               "non_speech", "other")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
SECRET = os.environ.get("QC_SECRET", "")
PUBLIC = {"/", "/index.html", "/app.js", "/style.css"}
LOCK = threading.Lock()

_state = {"items": [], "by_id": {}, "mtime": 0.0, "spans": None, "spans_mtime": 0.0}


def authorized(request: Request) -> bool:
    if not SECRET:
        return False
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], SECRET):
        return True
    key = request.query_params.get("key", "")
    return bool(key) and hmac.compare_digest(key, SECRET)


@app.middleware("http")
async def gate(request: Request, call_next):
    if request.url.path in PUBLIC:
        return await call_next(request)
    if not authorized(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


def baseline_spans() -> dict:
    """Absolute [start, end) spans for each Qwen3-ASR clip, from cumulative durations."""
    if not BASELINE.exists():
        return {}
    mtime = BASELINE.stat().st_mtime
    if _state["spans"] is not None and mtime == _state["spans_mtime"]:
        return _state["spans"]
    by_src = defaultdict(list)
    with open(BASELINE, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key", "")
            src, _, idx = key.rpartition("#")
            if not idx:
                continue
            by_src[src].append((int(idx), row.get("duration_s") or 0.0,
                                (row.get("transcript") or "").strip()))
    spans = {}
    for src, clips in by_src.items():
        clips.sort()
        t, out = 0.0, []
        for _, dur, text in clips:
            out.append((t, t + dur, text))
            t += dur
        spans[src] = out
    _state["spans"], _state["spans_mtime"] = spans, mtime
    return spans


def load_items() -> list[dict]:
    if not METADATA.exists():
        return []
    mtime = METADATA.stat().st_mtime
    if _state["items"] and mtime == _state["mtime"]:
        return _state["items"]
    items = []
    with open(METADATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("wav"):
                continue
            dur = float(row["duration"])
            chars = len(row["text"])
            items.append({
                "id": f"{row['source_file']}#{row['segment_index']}",
                "source_file": row["source_file"],
                "performer": row["source_file"].split("/")[0],
                "segment_index": int(row["segment_index"]),
                "wav": row["wav"],
                "text": row["text"],
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "duration": dur,
                "cps": round(chars / dur, 2) if dur else 0.0,
                "chars": chars,
                "speaker": row["speaker"],
                "n_speakers": int(row["n_speakers_in_file"] or 0),
                "tags": row.get("tags", ""),
                "repetition": float(row.get("repetition") or 0),
                "vocalization": float(row.get("vocalization") or 0),
            })
    items.sort(key=lambda r: (r["source_file"], r["segment_index"]))
    _state["items"] = items
    _state["by_id"] = {r["id"]: r for r in items}
    _state["mtime"] = mtime
    return items


def load_labels() -> dict:
    labels = {}
    if not LABELS.exists():
        return labels
    with open(LABELS, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                labels[row["id"]] = row
    return labels


def baseline_for(item: dict) -> str:
    spans = baseline_spans().get(item["source_file"])
    if not spans:
        return ""
    parts = [text for start, end, text in spans
             if text and end > item["start_s"] + 0.05 and start < item["end_s"] - 0.05]
    return " ".join(parts)


@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")


@app.get("/index.html")
def index_html():
    return FileResponse(STATIC / "index.html")


@app.get("/app.js")
def app_js():
    return FileResponse(STATIC / "app.js", media_type="application/javascript")


@app.get("/style.css")
def style_css():
    return FileResponse(STATIC / "style.css", media_type="text/css")


@app.get("/api/meta")
def meta():
    items = load_items()
    labels = load_labels()
    performers = defaultdict(int)
    for item in items:
        performers[item["performer"]] += 1
    counts = defaultdict(int)
    for label in labels.values():
        counts[label.get("accuracy", "unlabeled")] += 1
    return {
        "n_items": len(items),
        "n_labeled": len(labels),
        "counts": dict(counts),
        "performers": dict(sorted(performers.items(), key=lambda kv: -kv[1])[:40]),
        "accuracy_levels": list(ACCURACY),
        "error_types": list(ERROR_TYPES),
        "baseline_available": BASELINE.exists(),
        "metadata_mtime": _state["mtime"],
    }


@app.get("/api/index")
def api_index():
    items = load_items()
    labels = load_labels()
    light = [{k: item[k] for k in ("id", "performer", "source_file", "duration",
                                   "cps", "chars", "tags", "n_speakers", "wav",
                                   "text")}
             | {"label": labels.get(item["id"], {}).get("accuracy")}
             for item in items]
    return {"items": light, "n_labeled": len(labels)}


@app.get("/api/item/{item_id:path}")
def api_item(item_id: str):
    items = load_items()
    item = _state["by_id"].get(item_id)
    if not item:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    label = load_labels().get(item_id, {})
    return {**item, "baseline": baseline_for(item), "label": label,
            "total": len(items)}


@app.get("/api/audio/{item_id:path}")
def api_audio(item_id: str):
    item = _state["by_id"].get(item_id)
    if not item:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    path = ROOT / "out/dataset" / item["wav"]
    if not path.exists():
        return JSONResponse({"error": "missing wav"}, status_code=404)
    return FileResponse(path, media_type="audio/wav",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/context/{item_id:path}")
def api_context(item_id: str):
    """The clip plus `CONTEXT_PAD_S` seconds of source audio on each side.

    Tightly cut clips make boundary errors impossible to judge from the clip
    alone: a reviewer cannot tell whether a word was clipped unless they hear
    the audio leading into it. Decoded with ffmpeg input seeking (fast) and
    cached, since the same clip is often revisited.
    """
    item = _state["by_id"].get(item_id)
    if not item:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    src = AUDIO_ROOT / item["source_file"]
    if not src.exists():
        return JSONResponse({"error": "source missing"}, status_code=404)
    CONTEXT_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:16]
    out = CONTEXT_CACHE / f"{digest}.wav"
    if not out.exists():
        start = max(0.0, item["start_s"] - CONTEXT_PAD_S)
        dur = (item["end_s"] - item["start_s"]) + 2 * CONTEXT_PAD_S
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}",
               "-t", f"{dur:.3f}", "-i", str(src), "-ac", "1", "-ar", "24000",
               "-c:a", "pcm_s16le", str(out)]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0 or not out.exists():
            return JSONResponse({"error": "decode failed"}, status_code=500)
    return FileResponse(out, media_type="audio/wav",
                        headers={"Cache-Control": "public, max-age=86400",
                                 "X-Context-Start-S": f"{max(0.0, item['start_s'] - CONTEXT_PAD_S):.3f}",
                                 "X-Context-Pad-S": f"{CONTEXT_PAD_S:.3f}"})


@app.post("/api/label")
async def api_label(request: Request):
    body = await request.json()
    item_id = body.get("id", "")
    if _state["by_id"] and item_id not in _state["by_id"]:
        return JSONResponse({"error": "unknown id"}, status_code=400)
    accuracy = body.get("accuracy")
    if accuracy not in ACCURACY and accuracy is not None:
        return JSONResponse({"error": "bad accuracy"}, status_code=400)
    errors = [e for e in (body.get("errors") or []) if e in ERROR_TYPES]
    row = {"id": item_id, "accuracy": accuracy, "errors": errors,
           "corrected": (body.get("corrected") or "")[:2000],
           "notes": (body.get("notes") or "")[:2000],
           "ts": int(time.time())}
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, open(LABELS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
    return {"ok": True, "saved": row}


@app.get("/api/labels")
def api_labels():
    return {"labels": load_labels()}


@app.get("/api/export.csv")
def api_export():
    items = load_items()
    queued = io.StringIO()
    writer = csv.writer(queued)
    writer.writerow(["id", "source_file", "start_s", "end_s", "duration", "cps",
                     "text", "accuracy", "errors", "corrected", "notes", "ts"])
    labels = load_labels()
    for item in items:
        label = labels.get(item["id"])
        if not label:
            continue
        writer.writerow([item["id"], item["source_file"], item["start_s"],
                         item["end_s"], item["duration"], item["cps"],
                         item["text"], label.get("accuracy", ""),
                         ";".join(label.get("errors") or []),
                         label.get("corrected", ""), label.get("notes", ""),
                         label.get("ts", "")])
    return Response(queued.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition":
                             'attachment; filename="review_labels.csv"'})


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if not SECRET:
        print("warning: QC_SECRET is empty; every data endpoint will return 401")
    print(f"review console on http://{args.host}:{args.port} "
          f"({len(load_items())} items, {len(load_labels())} labeled)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()