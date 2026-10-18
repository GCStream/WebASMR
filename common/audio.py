"""Audio loading for VibeVoice-ASR.

The model expects 24 kHz mono float32 (its acoustic tokenizer runs at
24 kHz), which is also the F5-TTS native rate, so decoded audio is reusable
downstream without a second resample.

Long files are split into overlapping passes: VibeVoice-ASR is documented for
up to 60 minutes in a single 64K-token window, and 11 of the 578 source files
exceed that. Overlap lets a sentence straddling a window edge survive in one
of the two passes.
"""

import subprocess
from pathlib import Path

import numpy as np

TARGET_SR = 24000
MAX_PASS_S = 3600.0
OVERLAP_S = 120.0


def load_24k_mono(path: str | Path) -> np.ndarray:
    """Decode any input container to 24 kHz mono float32."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(TARGET_SR),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg failed for {path}: {proc.stderr.decode()[:400]}")
    x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if x.size == 0:
        raise RuntimeError(f"decoded empty audio: {path}")
    return x


def probe_duration_s(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    try:
        return float(out.splitlines()[0])
    except (IndexError, ValueError):
        return 0.0


def pass_windows(duration_s: float, max_pass_s: float = MAX_PASS_S,
                 overlap_s: float = OVERLAP_S) -> list[tuple[float, float]]:
    """Split a file into [start, end) second windows to feed the model."""
    if duration_s <= max_pass_s:
        return [(0.0, duration_s)]
    windows, start = [], 0.0
    stride = max_pass_s - overlap_s
    while start < duration_s:
        end = min(start + max_pass_s, duration_s)
        windows.append((start, end))
        if end >= duration_s:
            break
        start += stride
    return windows