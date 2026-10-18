"""Segment normalization and quality metrics.

VibeVoice-ASR emits a JSON list of `{"Start", "End", "Speaker", "Content"}`
where the timestamps are relative to the audio window that was submitted.
Everything here works in absolute file time so overlapping passes can be
stitched and so segment boundaries can be compared against the baseline
transcripts.

The metrics exist because ASMR is the hard case for ASR: long stretches of
moaning and breathing decode as runaway repetition ("啊啊啊啊..." x100), and
that failure has to be measurable rather than eyeballed.
"""

import json
import re
import unicodedata

MAX_RUN = 3

CUT_STARTERS = set("的地得了着过吗呢吧啊呀哦嗯，,。、！!?")

VOICE_CHARS = set("啊阿嗯唔哦噢喔嗯哼唉诶欸呵呼嘿嘻吼呃噢嗯嗯")
PUNCT = set("，。！？、；：,.!?;:…~ ")

CJK_RUN = re.compile(r"(.{1,10}?)\1{%d,}" % MAX_RUN)
SPACED_RUN = re.compile(r"(\S+(?: \S+){0,2})(?: \1){%d,}" % MAX_RUN)


def compress_runs(text: str) -> str:
    text = SPACED_RUN.sub(lambda m: " ".join([m.group(1)] * MAX_RUN), text)
    text = CJK_RUN.sub(lambda m: m.group(1) * MAX_RUN, text)
    return text


def repetition_ratio(text: str) -> float:
    """Fraction of characters inside a 3+ repetition of the same unit."""
    if not text:
        return 1.0
    runs = sum(len(m.group(0)) for m in
               re.finditer(r"(\S+)(?: \1){2,}|(.{1,10}?)\2{2,}", text))
    return runs / max(1, len(text))


def vocalization_ratio(text: str) -> float:
    """Fraction of non-punctuation characters that are bare vocalizations.

    High values mean the model is transcribing moans rather than words,
    which is expected for some ASMR but should not dominate the corpus.
    """
    body = [c for c in text if c not in PUNCT]
    if not body:
        return 1.0
    return sum(1 for c in body if c in VOICE_CHARS) / len(body)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return compress_runs(text)


SEGMENT_ANCHOR = '{"Start"'
NUM = r"(-?\d+(?:\.\d+)?)"

FIELD_START = re.compile(r'Start"?\s*:\s*' + NUM, re.S)
FIELD_END = re.compile(r'End"?\s*:\s*' + NUM, re.S)
FIELD_SPEAKER = re.compile(r'Speaker"?\s*:\s*(-?\d+)', re.S)
FIELD_CONTENT = re.compile(r'Content"?\s*:\s*"', re.S)


def _scan_segments(text: str) -> list[dict]:
    """Extract segment objects by field regex instead of strict JSON.

    The model occasionally emits a malformed object, e.g. a missing comma
    (`{"Start":81.33End":92.69,...}`). One bad object makes `json.loads`
    reject a whole file, and silently dropping ~15 minutes of good
    transcription over one typo is not acceptable, so each `{"Start"` chunk is
    read field by field. Content is taken as everything between the first
    quote after `"Content":` and the last quote in the chunk, which keeps
    unescaped inner quotes instead of truncating the segment.
    """
    out = []
    for chunk in text.split(SEGMENT_ANCHOR)[1:]:
        chunk = '"Start"' + chunk
        m_start = FIELD_START.search(chunk)
        m_end = FIELD_END.search(chunk)
        m_content = FIELD_CONTENT.search(chunk)
        if not (m_start and m_end and m_content):
            continue
        tail = chunk[m_content.end():]
        if '"' not in tail:
            continue
        content = tail[:tail.rindex('"')]
        content = content.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
        m_speaker = FIELD_SPEAKER.search(chunk)
        out.append({
            "Start": float(m_start.group(1)),
            "End": float(m_end.group(1)),
            "Speaker": int(m_speaker.group(1)) if m_speaker else 0,
            "Content": content,
        })
    return out


def parse_generation(raw_text: str) -> tuple[list, str]:
    """Parse a generation into segments, reporting why it failed.

    The model's output is a JSON array, but it is not reliably well-formed:
    generation can stop early (leaving the array unterminated) and individual
    objects can be malformed. Returning an empty list for either case is
    indistinguishable from "the file contains no speech", which would let a
    parse failure masquerade as a silent recording, so the status is explicit.

    Returns (segments, status) where status is one of:
      ok          complete, valid JSON array
      repaired    recovered field by field from a malformed array
      truncated   unterminated array, recovered through the last complete segment
      empty       valid array with no segments (no speech found)
      unparseable no recoverable segment at all
    """
    text = (raw_text or "").strip()
    if text.startswith("assistant"):
        text = text[len("assistant"):].strip()
    if not text:
        return [], "unparseable"

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed, ("ok" if parsed else "empty")
    except json.JSONDecodeError:
        pass

    if SEGMENT_ANCHOR not in text:
        return [], "unparseable"

    repaired = _scan_segments(text)
    if repaired:
        closed = text.rstrip().endswith("]")
        return repaired, ("truncated" if not closed else "repaired")
    return [], "unparseable"


def normalize_segments(raw, offset_s: float = 0.0, duration_s: float | None = None) -> list[dict]:
    """Coerce one pass's decoded output into absolute-time segment dicts.

    Accepts the parsed list from `processor.decode(..., return_format="parsed")`
    or the raw string form, so a malformed pass degrades to a warning instead
    of an exception.
    """
    if isinstance(raw, str):
        raw, _ = parse_generation(raw)
    if not isinstance(raw, list):
        return []

    out = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        text = normalize_text(str(seg.get("Content", "")))
        if not text:
            continue
        try:
            start = float(seg.get("Start", 0.0)) + offset_s
            end = float(seg.get("End", 0.0)) + offset_s
        except (TypeError, ValueError):
            start, end = offset_s, offset_s
        if end < start:
            start, end = end, start
        if duration_s is not None:
            start = min(max(start, 0.0), duration_s)
            end = min(max(end, 0.0), duration_s)
        try:
            speaker = int(seg.get("Speaker", 0))
        except (TypeError, ValueError):
            speaker = 0
        out.append({"start": round(start, 2), "end": round(end, 2),
                    "speaker": speaker, "text": text})
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def dedupe_overlap(segments: list[dict], tolerance_s: float = 1.0) -> list[dict]:
    """Collapse identical text repeated across overlapping pass windows."""
    kept: list[dict] = []
    for seg in segments:
        dup = None
        for prev in reversed(kept[-6:]):
            if abs(prev["start"] - seg["start"]) <= tolerance_s and prev["text"] == seg["text"]:
                dup = prev
                break
        if dup is None:
            kept.append(seg)
        elif seg["end"] > dup["end"]:
            dup["end"] = seg["end"]
    return kept


def file_stats(segments: list[dict], duration_s: float) -> dict:
    """Aggregate signals used to judge one transcribed file."""
    text = " ".join(s["text"] for s in segments)
    speech_s = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    monotonic = all(a["end"] <= b["start"] + 1.0 for a, b in zip(segments, segments[1:]))
    tail_gap = (duration_s - segments[-1]["end"]) if segments and duration_s else 0.0
    return {
        "n_segments": len(segments),
        "n_speakers": len({s["speaker"] for s in segments}),
        "speech_s": round(speech_s, 2),
        "coverage": round(speech_s / duration_s, 4) if duration_s else 0.0,
        "chars": len(text),
        "repetition": round(repetition_ratio(text), 4),
        "vocalization": round(vocalization_ratio(text), 4),
        "monotonic": monotonic,
        "tail_gap_s": round(max(0.0, tail_gap), 2),
    }


def transcript_of(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments).strip()