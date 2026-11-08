"""Step 3: per-episode candidates. One LLM call per episode.

The model sees one episode's beats (IDs + timestamps + speaker/emotion + text) and a slice
of the story map. It returns hook beats, cliffhanger beats, and the 1-3 best single-stretch
clips with rubric scores. It returns beat IDs only; code (clips.cut_clip) turns them into
timestamps and checks them, and those check errors go back to the model to fix.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from .clips import cut_clip, normalise_id
from .llm import LLMClient, LLMError
from .parser import Beat, Episode, Show, fmt_ts
from .story_map import StoryMap

def _score():
    return Field(ge=1, le=10)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class HookMoment(BaseModel):
    beat_id: str
    kind: Literal["physical_shock", "shocking_line", "humiliation_or_reversal", "confrontation_question"]
    score: int = _score()
    text: str = Field(description="The moment in at most 12 words")


class CliffMoment(BaseModel):
    beat_id: str
    kind: Literal["reveal_in_progress", "unresolved_threat", "decision_point", "interrupted_or_entrance"]
    score: int = _score()
    text: str = Field(description="The moment in at most 12 words")


class Scores(BaseModel):
    hook: int = _score()
    cliffhanger: int = _score()
    standalone: int = _score()
    intensity: int = _score()
    premise: int = _score()
    pacing: int = _score()


class ClipPick(BaseModel):
    hook_beat: str = Field(description="Beat ID the clip opens on")
    cliffhanger_beat: str = Field(description="Beat ID the clip ends right after")
    hook: str = Field(description="What the hook is, at most 15 words")
    cliffhanger: str = Field(description="What is left unresolved, at most 15 words")
    why: str = Field(description="One line on why this works as a promo for new viewers")
    scores: Scores


class EpisodeCandidates(BaseModel):
    ends_on_cliffhanger: bool = Field(description="Does the episode itself end on an unresolved moment?")
    hooks: list[HookMoment] = Field(description="Up to 4 strongest hook beats")
    cliffhangers: list[CliffMoment] = Field(description="Up to 4 strongest cliffhanger beats")
    clips: list[ClipPick] = Field(description="The 1-3 best promos, each one continuous stretch")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM = """You are a promo editor for a short-drama app. You choose 20-60 second clips from an episode to post \
on TikTok, Reels and Shorts, to pull in NEW viewers who have never seen the show. You only select moments \
(start and end); there is no editing, music or text overlay. Refer to moments only by beat IDs from the listing."""

TASK = """SHOW: {title}
PREMISE: {premise}
TITLE PREMISE THREADS: {premise_threads}
CHARACTERS: {characters}

EPISODE {episode} of {n_episodes}
LOGLINE: {logline}
KEY REVEALS: {reveals}

RUBRIC
- Hook: the clip opens on it and its payoff lands within 3 seconds. Strong hooks: physical shock (slap, grab, \
wound reveal); a shocking line or reveal; public humiliation or status reversal; a confrontation that opens on \
an unignorable question.
- Cliffhanger: the clip cuts right AFTER this beat and BEFORE the reaction or answer. Strong cliffhangers: a \
reveal in progress; an unresolved threat; a decision point (about to confess, choose or retaliate); an \
interrupted action or unexpected entrance.
- Standalone: a stranger must follow it. Few named characters, relationships readable from the clip itself, \
familiar tropes.

TASK
1. hooks: up to 4 of the strongest hook beats in this episode.
2. cliffhangers: up to 4 of the strongest cliffhanger beats.
3. clips: the 1-3 best promos. Each is ONE continuous stretch that starts at hook_beat and ends right after \
cliffhanger_beat. Use the timestamps: the stretch must last 20-60 seconds. Prefer clips a stranger understands.
   Score each 1-10: hook, cliffhanger, standalone, intensity (emotional heat), premise (how directly it shows \
the title premise), pacing (no dead air between hook and cliffhanger).
   Be a harsh critic: 9-10 only for moments that would stop a scroll; most clips are 5-7.
4. ends_on_cliffhanger: does the episode itself end on an unresolved moment?

BEATS (id [start-end] SPEAKER (tags): "line" | CAMERA/ACTION/SFX: description)
{beats}"""


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def beat_line(b: Beat, max_desc: int = 120) -> str:
    span = f"[{fmt_ts(b.start)}-{fmt_ts(b.end)}]"
    if b.type == "dialogue":
        tags = ", ".join(([b.delivery] if b.delivery else []) + b.emotion)
        return f'{b.beat_id} {span} {b.speaker}' + (f" ({tags})" if tags else "") + f': "{b.text}"'
    return f"{b.beat_id} {span} {b.type.upper()}: {_clip(b.text, max_desc)}"


def build_prompt(show: Show, sm: StoryMap, ep: Episode, title: str | None) -> str:
    note = next((e for e in sm.episodes if e.episode == ep.episode), None)
    premise_threads = "; ".join(f"{t.title}: {t.summary}" for t in sm.threads if t.title_premise) or "(none marked)"
    characters = "; ".join(f"{c.name} ({c.role})" for c in sm.characters)
    return TASK.format(
        title=title or "(untitled)",
        premise=sm.premise,
        premise_threads=premise_threads,
        characters=characters,
        episode=ep.episode,
        n_episodes=len(show.episodes),
        logline=ep.logline,
        reveals="; ".join(note.key_reveals) if note and note.key_reveals else "(none listed)",
        beats="\n".join(beat_line(b) for b in ep.beats),
    )


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_candidates(c: EpisodeCandidates, ep: Episode) -> list[str]:
    ids = {b.beat_id for b in ep.beats}
    errs: list[str] = []
    for m in [*c.hooks, *c.cliffhangers]:
        if normalise_id(m.beat_id, ep.episode) not in ids:
            errs.append(f"beat {m.beat_id} is not in episode {ep.episode}")
    if not c.clips:
        errs.append("give at least one clip")
    for i, clip in enumerate(c.clips, 1):
        cut = cut_clip(ep, clip.hook_beat, clip.cliffhanger_beat)
        errs += [f"clip {i}: {e}" for e in cut.errors]
    return errs


def find_candidates(show: Show, sm: StoryMap, ep: Episode, client: LLMClient, title: str | None) -> EpisodeCandidates:
    return client.complete_json(
        step="candidates",
        system=SYSTEM,
        prompt=build_prompt(show, sm, ep, title),
        schema=EpisodeCandidates,
        check=lambda c: check_candidates(c, ep),
        on_check_fail="accept",  # the validator rejects any clip that is still broken
    )


def find_all_candidates(
    show: Show, sm: StoryMap, client: LLMClient, title: str | None, episodes: list[int] | None = None
) -> tuple[dict[int, EpisodeCandidates], dict[int, str], dict[int, str]]:
    """Run step 3 over the show in parallel.
    Returns ({episode: candidates}, {episode: error}, {episode: model that answered})."""
    eps = [e for e in show.episodes if episodes is None or e.episode in episodes]
    results: dict[int, EpisodeCandidates] = {}
    failed: dict[int, str] = {}
    models: dict[int, str] = {}

    def work(ep: Episode):
        try:
            results[ep.episode] = find_candidates(show, sm, ep, client, title)
            models[ep.episode] = client.last_model() or client.model_for("candidates")
            print(f"      ep {ep.episode:>3}: {len(results[ep.episode].clips)} clips ({models[ep.episode]})", flush=True)
        except LLMError as e:
            failed[ep.episode] = str(e)
            print(f"      ep {ep.episode:>3}: FAILED: {str(e)[:160]}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, client.cfg.concurrency)) as pool:
        list(pool.map(work, eps))
    return dict(sorted(results.items())), dict(sorted(failed.items())), dict(sorted(models.items()))
