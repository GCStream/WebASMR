#!/usr/bin/env python3
"""Dataset export tests: python3 tests/test_dataset_export.py

The naming test is the important one. The upstream pipeline wrote
`clip_{clip_index}.wav` while `clip_index` restarts per source file, so 6,683
metadata rows pointed at 186 wav files and every text was paired with another
recording's audio.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from importlib import import_module  # noqa: E402

bd = import_module("70_build_dataset")

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def test_clip_names_are_unique_across_files():
    """Two source files whose clip_index both start at 0 must not collide."""
    a = bd.clip_name("artistA/song.mp3", 0, 0.0)
    b = bd.clip_name("artistB/song.mp3", 0, 0.0)
    check("different source files differ", a != b, True)
    check("stable for the same input", a, bd.clip_name("artistA/song.mp3", 0, 0.0))
    check("index differs", a != bd.clip_name("artistA/song.mp3", 1, 0.0), True)
    check("start differs", a != bd.clip_name("artistA/song.mp3", 0, 5.0), True)

    names = {bd.clip_name(f"artist{i}/s.mp3", j, 0.0) for i in range(50) for j in range(20)}
    check("1000 combos all distinct", len(names), 1000)


def test_clip_name_is_a_safe_filename():
    name = bd.clip_name("樱 桃/奇怪的名字？*.mp3", 12, 3.7)
    check("ascii only", name.isascii(), True)
    check("no slash", "/" in name, False)
    check("has index", "_00012_" in name, True)


def test_partition_splits_the_three_kinds():
    class Args:
        min_chars = 6
        min_clip_s = 1.0
        max_clip_s = 20.0
        max_repetition = 0.6
        max_vocalization = 0.75
        sfx_max_s = 30.0
        min_cps = 1.5
        max_cps = 9.0

    segments = [
        {"text": "弟弟你回来啦，我给你做饭", "start": 0.0, "end": 5.0, "speaker": 0,
         "source_file": "f"},
        {"text": "[Human Sounds]", "start": 5.0, "end": 8.0, "speaker": 0,
         "source_file": "f"},
        {"text": "嗯", "start": 8.0, "end": 8.8, "speaker": 1, "source_file": "f"},
        {"text": "[Music]", "start": 8.8, "end": 60.0, "speaker": 0, "source_file": "f"},
        {"text": "", "start": 60.0, "end": 61.0, "speaker": 0, "source_file": "f"},
    ]
    speech, nonspeech, vocal = bd.partition(segments, Args())
    check("one speech segment", len(speech), 1)
    check("two non-speech stretches", len(nonspeech), 2)
    check("short interjection banked", len(vocal), 1)
    check("short stretch is an event", nonspeech[0]["kind"], "event")
    check("long stretch is a bed", nonspeech[1]["kind"], "bed")
    check("bed labelled", nonspeech[1]["label"], "Music")
    check("tags stripped from speech", speech[0]["tags"], [])


def test_partition_drops_degenerate_speech():
    class Args:
        min_chars = 6
        min_clip_s = 1.0
        max_clip_s = 20.0
        max_repetition = 0.6
        max_vocalization = 0.75
        sfx_max_s = 30.0
        min_cps = 1.5
        max_cps = 9.0

    degenerate = [
        {"text": "啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊", "start": 0.0, "end": 5.0,
         "speaker": 0, "source_file": "f"},
        {"text": "这是一段很长很长很长很长很长的文本内容需要被过滤掉因为超过了二十秒",
         "start": 5.0, "end": 30.0, "speaker": 0, "source_file": "f"},
        {"text": "好的", "start": 30.0, "end": 31.0, "speaker": 0, "source_file": "f"},
    ]
    speech, nonspeech, vocal = bd.partition(degenerate, Args())
    check("loops rejected", len(speech), 0)
    check("loops not banked as sfx", len(nonspeech), 0)
    check("only the short utterance is banked", len(vocal), 1)
    check("overrun rejected, not banked", [v["text"] for v in vocal], ["好的"])


for fn in [test_clip_names_are_unique_across_files, test_clip_name_is_a_safe_filename,
           test_partition_splits_the_three_kinds, test_partition_drops_degenerate_speech]:
    fn()

if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all tests passed")