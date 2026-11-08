# Kuku Promo Finder

Reads a folder of microdrama screenplays and picks the best 20–60s promo clips: each one opens on a hook,
ends on a cliffhanger, and should make sense to someone who has never seen the show. For every promo it
outputs the episode and start/end timestamps, the hook, the cliffhanger and one line on why it works,
then ranks them and flags weak picks for a human editor.

## Run it

Python 3.11+.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
cp .env.example .env                             # then paste a free Gemini key: https://aistudio.google.com/apikey
.venv/bin/python run.py --input path/to/episodes --title "Show title" --out outputs/raw
```

- `--input`: a folder of episode `.txt` files (format below). Works on any show in this format.
- `--title`: optional; used to judge which moments sell the show's title premise.
- `--episodes 1-3,7`: run the per-episode step on a subset (quick trial).
- `--dry-run`: parse only, and write the story-map prompt so you can inspect it. No LLM calls.

The main output is `outputs/raw/promos.md` (ranked report). Alongside it: `promos.json` (the same data,
plus dropped and rejected clips), `candidates.json` (raw model output per episode), `story_map.json`,
`beats.json`, and `usage.jsonl` (tokens and latency per LLM call). Outputs contain script text: keep them
out of public repos. `outputs/raw/` and `data/` are gitignored.

LLM replies are cached in `.cache/llm/`, so a re-run is free and gives identical output. If a free-tier
daily quota runs out mid-run, run again later and it resumes where it stopped.

## Swap the model

Every provider goes through one adapter (`promo_finder/llm.py`) that speaks the OpenAI chat-completions
protocol, which Gemini, Groq, Ollama, OpenAI, OpenRouter and most in-house gateways accept. Change
`provider` and `model` in `config.yaml`, or override for one run:

```bash
PROMO_PROVIDER=groq PROMO_MODEL=llama-3.3-70b-versatile python run.py --input ...     # needs GROQ_API_KEY
PROMO_PROVIDER=ollama PROMO_MODEL=qwen2.5:14b python run.py --input ...               # local, no key
PROMO_BASE_URL=https://your-gateway/v1 PROMO_API_KEY_ENV=YOUR_KEY_VAR PROMO_MODEL=... python run.py --input ...
```

`config.yaml` also sets per-step models (`step_models`), temperature, the requests-per-minute throttle and
parallelism for free tiers.

## How it works

| Step | What | LLM? |
|---|---|---|
| 1. Parse | Screenplay → timed beats (camera, action, dialogue, sound), with speaker, emotion tags, and a flag for lines that continue the previous one. | no |
| 2. Story map | One call over all loglines + continuity notes + every speaker label → characters, speaker aliases (per episode where a label is reused for different people), story threads, title-premise threads, reveals per episode. | 1 call |
| 3. Candidates | One call per episode (in parallel) → hook beats, cliffhanger beats, and the 1–3 best clips with rubric scores (hook, cliffhanger, standalone, intensity, premise, pacing). The model returns **beat IDs only**; code maps them to timestamps. | 1 per episode |
| 5. Validate | Cut each clip on beat edges: never open or cut mid-line, 20–60s, hook within the first 3s. Errors go back to the model to fix inside step 3; anything still broken is rejected and listed. | no |
| 7. Rank + report | Weighted rubric score minus a penalty per cut adjustment; overlapping clips removed; max 2 per episode; weak picks flagged. | no |

Every LLM reply is validated against a pydantic schema plus semantic checks (IDs exist, clip lengths, etc.);
errors are sent back to the model for up to 2 fix rounds.

Not built yet: step 4 (stitching moments across episodes into one promo) and step 6 (an independent
"standalone judge" model that reads only the clip).

## Input format

```
EPISODE NUMBER: Episode 1
EPISODE LOGLINE: <one sentence>
CONTINUITY NOTES: <how it connects>
--- SCRIPT START ---
SCENE 01
Scene heading: INT. CATHEDRAL – DAY
Scene start: 00:00.00
Scene end: 02:18.00
**CAMERA:**
Start: (00:00.00) | End: (00:04.80)
<shot>
**SPEAKER NAME (Emotion, Tags):**
Start: (00:11.30) | End: (00:13.10)
"Line."
--- SCRIPT END ---
```

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

Fully offline (a fake transport replaces the API). The one test that reads the real input pack is skipped
when `data/` is absent.
