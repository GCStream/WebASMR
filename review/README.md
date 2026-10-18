# Accuracy review console

Human accuracy labeling for the VibeVoice-ASR segments. Served at `/review/` on
the same host as the transcripts console (Caddy strips the prefix and proxies to
`127.0.0.1:8141`).

    ./review/run.sh          # reads /root/.qc_secret, serves on 8141
    open http://<host>/review/

The transcripts console is untouched: its routes are `/` and `/qc/*`, this one
is `/review/*`.

## Why it exists

There is no ground truth for this corpus, so accuracy can only be estimated from
human labels. Everything else the audit produces (recall vs Qwen3-ASR, loop rate,
CPS sanity) is a proxy. A few hundred labels turn "looks reasonable" into a
number with a confidence interval.

4,595 clips is far more than one person should label, so the toolbar has a
`sample` box: set it to 200-400 and the queue becomes a fixed random sample.
The same seed is reused, so a reload resumes the same sample.

## What the reviewer sees

- the clip waveform, with a progress overlay and click-to-seek
- `context ±1s` plays the clip plus one second either side, shaded in the
  waveform, because a boundary error cannot be judged from a tightly cut clip
- our transcript, and the Qwen3-ASR 30 s clip covering the same window as a
  second opinion (it spans more audio; the UI says so)
- the clip's absolute timestamps, duration, characters/second, event tags and
  speaker count

## Labels

| key | meaning |
|---|---|
| `1` | correct — words match the audio |
| `2` | minor — a wrong character or punctuation that does not change meaning |
| `3` | major — wrong words, missing clause, or text for audio that is not there |
| `4` | unusable — not a single clean utterance, or the audio is not speech |

Error toggles: `w` wrong words, `m` missing words, `h` hallucinated,
`b` bad boundary, `n` non-speech, `o` other.
`space` play/pause, `r` replay, `←/→` move, `s` skip, `e` corrected-text field,
`?` help. Accuracy clicks auto-save and auto-advance (toggleable).

Labels are append-only JSONL at `out/review/labels.jsonl`, last write wins per
item, so a restart or a closed tab never loses a decision. Candidates for the
`corrected` field are one click away via `use as correction` on the baseline.

## Accuracy estimate

The side panel computes the correct share with a 95% Wilson interval as labels
come in. Treat it as the dataset's measured accuracy once n is in the hundreds:
at n=100 a 90% accuracy carries roughly a ±6 point interval.

Export for analysis:

    curl -H "Authorization: Bearer $(cat /root/.qc_secret)" \
      http://127.0.0.1:8141/api/export.csv > review_labels.csv

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/api/meta` | counts, performers, label levels |
| GET | `/api/index` | light rows for the whole queue (id, text, cps, label) |
| GET | `/api/item/{id}` | full item + baseline text + current label |
| GET | `/api/audio/{id}` | the clip |
| GET | `/api/context/{id}` | clip + 1 s padding, cached under `out/review/ctx/` |
| POST | `/api/label` | append a label |
| GET | `/api/labels` | all labels |
| GET | `/api/export.csv` | CSV of labeled items |

Data endpoints require the console secret (`Authorization: Bearer`, or `?key=`
for links). The page shell is public so the login box can render.

The item list reloads automatically when `out/dataset/metadata.csv` changes, so
re-running `70_build_dataset.py` after the corpus pass finishes is enough to
extend the queue.
