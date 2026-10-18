#!/usr/bin/env python3
"""Parser and metric tests: python3 tests/test_segments.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.audio import pass_windows  # noqa: E402
from common.keys import clip_key, parse_clip_key, source_key  # noqa: E402
from common.segments import (dedupe_overlap, file_stats, normalize_segments,  # noqa: E402
                             parse_generation, repetition_ratio, vocalization_ratio)

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def test_parse_generation_status():
    ok = '[{"Start":0,"End":1,"Speaker":0,"Content":"hi"}]'
    check("ok", parse_generation("assistant\n" + ok)[1], "ok")
    check("empty", parse_generation("[]")[1], "empty")
    check("blank", parse_generation("")[1], "unparseable")
    check("prose", parse_generation("no json here")[1], "unparseable")
    check("not a list", parse_generation('{"a":1}')[1], "unparseable")

    cut = '[{"Start":0,"End":1,"Speaker":0,"Content":"a"},{"Start":1,"End":2,"Speaker'
    segs, status = parse_generation(cut)
    check("truncated status", status, "truncated")
    check("truncated keeps head", len(segs), 1)
    check("truncated text", segs[0]["Content"], "a")

    unterminated_string = '[{"Start":0,"End":1,"Speaker":0,"Content":"cut off'
    check("no closing brace", parse_generation(unterminated_string)[1], "unparseable")


def test_parse_generation_repairs_malformed_objects():
    bad_comma = ('[{"Start":0,"End":1,"Speaker":0,"Content":"first"},'
                 '{"Start":1.5End":3,"Speaker":0,"Content":"second"},'
                 '{"Start":3,"End":4,"Content":"third"}]')
    segs, status = parse_generation(bad_comma)
    check("malformed status", status, "repaired")
    check("all objects recovered", [s["Content"] for s in segs], ["first", "second", "third"])
    check("malformed start parsed", segs[1]["Start"], 1.5)
    check("absent speaker defaults", segs[2]["Speaker"], 0)

    truncated_bad = bad_comma[:-1] + ',{"Start":4,"End":5,"Speaker":0,"Content":"cut'
    segs, status = parse_generation(truncated_bad)
    check("truncated malformed status", status, "truncated")
    check("truncated malformed keeps all complete", len(segs), 3)

    inner_quotes = '[{"Start":0,"End":1,"Speaker":0,"Content":"他说"好"，然后走了"}]'
    segs, status = parse_generation(inner_quotes)
    check("inner quotes kept", segs[0]["Content"], '他说"好"，然后走了')

    escaped = '[{"Start":0,"End":1,"Speaker":0,"Content":"say \\"hi\\" now"}]'
    check("escaped quotes", parse_generation(escaped)[0][0]["Content"], 'say "hi" now')


def test_normalize_segments():
    raw = [
        {"Start": 10.0, "End": 12.0, "Speaker": 1, "Content": "second"},
        {"Start": 0.0, "End": 2.0, "Speaker": 0, "Content": "first"},
        {"Start": 5.0, "End": 4.0, "Content": "reversed"},
        {"Start": 3.0, "End": 4.0, "Speaker": 0, "Content": "   "},
        "junk",
    ]
    segs = normalize_segments(raw, offset_s=100.0, duration_s=110.0)
    check("sorted and offset", [s["text"] for s in segs], ["first", "reversed", "second"])
    check("offset applied", segs[0]["start"], 100.0)
    check("reversed span fixed", (segs[1]["start"], segs[1]["end"]), (104.0, 105.0))
    check("missing speaker defaults", segs[1]["speaker"], 0)
    check("clamped to duration", max(s["end"] for s in segs), 110.0)


def test_dedupe_overlap():
    pass_one = normalize_segments([
        {"Start": 3480.0, "End": 3490.0, "Speaker": 0, "Content": "same"},
        {"Start": 3495.0, "End": 3500.0, "Speaker": 0, "Content": "next"},
    ])
    pass_two = normalize_segments([
        {"Start": 0.0, "End": 10.0, "Speaker": 0, "Content": "same"},
        {"Start": 12.0, "End": 20.0, "Speaker": 0, "Content": "new"},
    ], offset_s=3480.0)
    kept = dedupe_overlap(pass_one + pass_two)
    check("overlap deduped", len(kept), 3)
    check("longer span wins", kept[0]["end"], 3490.0)
    check("kept distinct text", [s["text"] for s in kept], ["same", "next", "new"])


def test_metrics():
    check("loop detected", repetition_ratio("啊啊啊啊啊啊啊啊啊啊") > 0.6, True)
    check("clean text", repetition_ratio("弟弟你回来啦我给你做饭") < 0.1, True)
    check("empty is all repetition", repetition_ratio(""), 1.0)
    check("moan ratio", vocalization_ratio("啊，嗯，哦") > 0.9, True)
    check("speech ratio", vocalization_ratio("弟弟你回来啦") < 0.1, True)


def test_file_stats():
    segs = normalize_segments([
        {"Start": 0.0, "End": 5.0, "Speaker": 0, "Content": "a"},
        {"Start": 5.0, "End": 9.0, "Speaker": 1, "Content": "b"},
    ])
    stats = file_stats(segs, duration_s=10.0)
    check("speakers", stats["n_speakers"], 2)
    check("coverage", stats["coverage"], 0.9)
    check("monotonic", stats["monotonic"], True)
    check("tail gap", file_stats([], 10.0)["n_segments"], 0)


def test_pass_windows():
    check("short file is one pass", pass_windows(600.0), [(0.0, 600.0)])
    wins = pass_windows(3600.0 + 300.0)
    check("long file splits", len(wins) > 1, True)
    check("windows start at zero", wins[0][0], 0.0)
    check("covers the end", wins[-1][1], 3900.0)
    check("overlap present", wins[1][0] < wins[0][1], True)
    check("no window over cap", max(e - s for s, e in wins) <= 3600.0, True)


def test_iter_audio_files_is_case_insensitive(tmp_root=None):
    import tempfile
    from common.keys import iter_audio_files
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "artist").mkdir()
        (root / "artist" / "a.mp3").write_bytes(b"x")
        (root / "artist" / "b.MP3").write_bytes(b"x")
        (root / "artist" / "c.m4a").write_bytes(b"x")
        (root / "artist" / "notes.txt").write_bytes(b"x")
        found = {p.name for p in iter_audio_files(root)}
    check("finds all audio, ignores text", found, {"a.mp3", "b.MP3", "c.m4a"})


def test_keys():
    check("clip key", clip_key("a/b.mp3", 3), "a/b.mp3#3")
    check("clip roundtrip", parse_clip_key("a/b.mp3#3"), ("a/b.mp3", 3))
    check("no index", parse_clip_key("a/b.mp3"), ("a/b.mp3", None))
    check("source key", source_key(Path("/root/data_raw/audio/x/y.mp3"),
                                   Path("/root/data_raw/audio")), "x/y.mp3")


for fn in [test_parse_generation_status, test_parse_generation_repairs_malformed_objects,
           test_normalize_segments, test_dedupe_overlap,
           test_metrics, test_file_stats, test_pass_windows,
           test_iter_audio_files_is_case_insensitive, test_keys]:
    fn()

if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all tests passed")