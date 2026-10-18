"""Key conventions.

Every corpus artifact is addressed the same way the upstream WebAMSR dataset
addresses it: the source file path relative to the audio root, e.g.

    XiaoMei/443 学妹寸止--小美.mp3

Clip-level keys keep the upstream `<source_file>#<clip_index>` suffix so rows
stay joinable against the existing whisper-turbo / Qwen3-ASR transcripts.
"""

import os
from pathlib import Path


def source_key(path: Path, audio_root: Path) -> str:
    return str(Path(path).relative_to(audio_root)).replace("\\", "/")


def clip_key(src: str, clip_index: int) -> str:
    return f"{src}#{clip_index}"


def parse_clip_key(key: str) -> tuple[str, int | None]:
    src, sep, idx = key.rpartition("#")
    return (src, int(idx)) if sep else (key, None)


DEFAULT_AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac")


def iter_audio_files(audio_root: Path,
                     suffixes=DEFAULT_AUDIO_SUFFIXES) -> list[Path]:
    """All audio files under `audio_root`, matched case-insensitively.

    Case matters here: the upstream dataset contains `.MP3` alongside `.mp3`
    (8 files), and a case-sensitive glob silently drops them from a corpus
    pass that otherwise looks complete.
    """
    lowered = {s.lower() for s in suffixes}
    root = Path(audio_root)
    found, seen = [], set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if Path(name).suffix.lower() not in lowered:
                continue
            path = Path(dirpath) / name
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            found.append(path)
    return found