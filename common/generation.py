"""Generation controls for long-form ASR passes.

A 9B decoder on 300 s of ASMR will occasionally fall into a repetition loop
instead of emitting EOS. Left alone it burns the whole token budget (6144+
tokens for five minutes of audio) and the pass cost balloons, so generation
is stopped as soon as the tail of the sequence is provably cycling.
"""

import torch
from transformers import (LogitsProcessor, StoppingCriteria,
                          StoppingCriteriaList)

TOKENS_PER_SECOND = 12
MIN_NEW_TOKENS = 1024


def token_budget(duration_s: float, tokens_per_second: int = TOKENS_PER_SECOND) -> int:
    """Token cap scaled to the audio length.

    Measured density on this corpus is ~8 generated tokens per second of
    speech, so 12/s leaves headroom without letting a degenerate pass run
    away on a long file.
    """
    return max(MIN_NEW_TOKENS, int(duration_s * tokens_per_second))


class DegenerateLoopStop(StoppingCriteria):
    """Stop when the tail of the sequence keeps repeating the same tokens.

    Compares the last `window` tokens against the equally sized blocks before
    it and counts how many are at least `similarity` identical. Exact equality
    is too strict: a looping generation drifts slightly between cycles, and
    without this the pass burns its whole token budget. The comparison is
    token-level so genuinely repeated words in speech ("真的吗" x3) do not
    trip it, while a minute of "啊啊啊啊" does.
    """

    def __init__(self, window: int = 32, min_periods: int = 4, similarity: float = 0.9):
        self.window = window
        self.min_periods = min_periods
        self.similarity = similarity
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        need = self.window * self.min_periods
        if input_ids.shape[-1] < need:
            return False
        seq = input_ids[0]
        tail = seq[-need:]
        block = tail[-self.window:]
        matches = 0
        for i in range(0, need - self.window, self.window):
            other = tail[i:i + self.window]
            agree = (other == block).sum().item()
            if agree >= self.window * self.similarity:
                matches += 1
        if matches >= self.min_periods - 1:
            self.triggered = True
            return True
        return False


def stopping_criteria(window: int = 32, min_periods: int = 4,
                      similarity: float = 0.9) -> StoppingCriteriaList:
    return StoppingCriteriaList([DegenerateLoopStop(window, min_periods, similarity)])


def _tail_is_looping(seq, window: int, min_periods: int, similarity: float) -> bool:
    need = window * min_periods
    if seq.shape[-1] < need:
        return False
    tail = seq[-need:]
    block = tail[-window:]
    matches = 0
    for i in range(0, need - window, window):
        agree = (tail[i:i + window] == block).sum().item()
        if agree >= window * similarity:
            matches += 1
    return matches >= min_periods - 1


class LoopTerminate(LogitsProcessor):
    """Force EOS on rows whose generation has fallen into a repetition loop.

    A `StoppingCriteria` can only stop the whole batch, so with several clips
    per batch one degenerate clip would truncate the others. Masking the logits
    down to EOS ends just that row, letting its neighbours run to their own EOS
    while the wasted tail is capped at one token instead of the whole budget.
    """

    def __init__(self, eos_token_id: int, pad_token_id: int | None = None,
                 window: int = 32, min_periods: int = 4, similarity: float = 0.9):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.window = window
        self.min_periods = min_periods
        self.similarity = similarity
        self.triggered_rows: set[int] = set()

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            if row in self.triggered_rows:
                continue
            seq = input_ids[row]
            if self.pad_token_id is not None and int(seq[-1]) == self.pad_token_id:
                continue
            if _tail_is_looping(seq, self.window, self.min_periods, self.similarity):
                self.triggered_rows.add(row)
                scores[row, :] = torch.finfo(scores.dtype).min
                scores[row, self.eos_token_id] = 0
        return scores

    @property
    def triggered(self) -> bool:
        return bool(self.triggered_rows)


def loop_terminate(eos_token_id: int, pad_token_id: int | None = None,
                   window: int = 32, min_periods: int = 4,
                   similarity: float = 0.9) -> LoopTerminate:
    return LoopTerminate(eos_token_id, pad_token_id, window, min_periods, similarity)