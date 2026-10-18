#!/usr/bin/env python3
"""Loop-detector tests: python3 tests/test_generation.py"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.generation import (DegenerateLoopStop, loop_terminate,  # noqa: E402
                              token_budget)

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def ids(seq):
    return torch.tensor([seq], dtype=torch.long)


def fired(stop, seq):
    return bool(stop(ids(seq), None))


def test_stops_on_exact_loop():
    stop = DegenerateLoopStop(window=8, min_periods=4)
    seq = list(range(1000, 1008)) * 6
    check("exact loop stops", fired(stop, seq), True)


def test_stops_on_drifting_loop():
    stop = DegenerateLoopStop(window=8, min_periods=4, similarity=0.75)
    base = [500, 501, 502, 503, 504, 505, 506, 507]
    seq = []
    for i in range(6):
        block = list(base)
        block[i % 8] = 900 + i
        seq += block
    check("drifting loop stops", fired(stop, seq), True)
    check("exact comparator would miss it",
          all(block != seq[-8:] for block in (seq[0:8], seq[8:16], seq[16:24])), True)


def test_ignores_normal_speech():
    stop = DegenerateLoopStop(window=8, min_periods=4)
    seq = list(range(3000, 3000 + 8 * 5))
    check("varied tokens continue", fired(stop, seq), False)
    check("short sequence continues", fired(stop, [1, 2, 3]), False)


def test_tiny_cycle_counts_as_degenerate():
    stop = DegenerateLoopStop(window=8, min_periods=4)
    check("two-token cycle stops", fired(stop, [1, 2] * 40), True)


def test_trigger_flag():
    stop = DegenerateLoopStop(window=8, min_periods=4)
    stop(ids(list(range(10, 18)) * 6), None)
    check("triggered flag set", stop.triggered, True)
    check("flag stays false otherwise", DegenerateLoopStop().triggered, False)


def test_token_budget():
    check("short audio floor", token_budget(10), 1024)
    check("scales with length", token_budget(300), 3600)
    check("long file", token_budget(3600), 43200)


def test_loop_terminate_targets_one_row():
    stop = loop_terminate(eos_token_id=643, pad_token_id=655)
    ids = torch.tensor([list(range(100, 132)) * 4,       # looping
                        list(range(1000, 1132))])        # varied, not looping
    scores = torch.zeros(2, 1000)
    out = stop(ids, scores)
    check("only looping row terminated", sorted(stop.triggered_rows), [0])
    check("looping row forced to eos", int(out[0].argmax()), 643)
    check("other row untouched", int(out[1].argmax()), int(scores[1].argmax()))
    check("other row not masked", bool((out[1][:643] > -1e30).any()), True)


def test_loop_terminate_skips_finished_rows():
    stop = loop_terminate(eos_token_id=643, pad_token_id=655)
    loop_tail = list(range(100, 132)) * 4
    ids = torch.tensor([loop_tail + [643] * 4, loop_tail])
    stop(ids, torch.zeros(2, 1000))
    check("already-finished row skipped", sorted(stop.triggered_rows), [1])


def test_loop_terminate_is_idempotent():
    stop = loop_terminate(eos_token_id=643, pad_token_id=655)
    ids = torch.tensor([list(range(100, 132)) * 4])
    stop(ids, torch.zeros(1, 1000))
    scores = torch.zeros(1, 1000)
    stop(ids, scores)
    check("row recorded once", len(stop.triggered_rows), 1)
    check("triggered flag", stop.triggered, True)


for fn in [test_stops_on_exact_loop, test_stops_on_drifting_loop, test_ignores_normal_speech,
           test_tiny_cycle_counts_as_degenerate, test_trigger_flag, test_token_budget]:
    fn()

if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all tests passed")