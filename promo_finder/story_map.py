"""Step 2: story map. One LLM call over the whole show's outline.

Input: every episode's logline + continuity notes, plus every speaker label with
sample lines from its first, middle and last episode (each with the shot just before
it, which often names the character).
Output: characters (with the speaker labels that refer to them), story threads, and
key reveals per episode.

The character -> labels mapping becomes the alias map that resolves descriptive
speaker names the parser can't ("THE FORMER GROOM" -> GERALD). A label can be
mapped for specific episodes only, because descriptive labels get reused for
different people (one "BLOND NOBLEMAN" early in a show, a different one later).
Every label is checked against the real speaker list; mistakes go back to the
model to fix, and anything still wrong after that is dropped rather than trusted.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .llm import LLMClient
from .parser import Beat, Show, clean_name

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class LabelUse(BaseModel):
    label: str = Field(description="A label from SPEAKERS, copied exactly")
    episodes: list[int] = Field(
        default_factory=list,
        description="Only if the label means different people in different episodes: the episodes where "
        "it is this character. Leave [] when it is this character in every episode it appears in.",
    )


class Character(BaseModel):
    name: str = Field(description="The short name the script uses for them, usually their label, e.g. 'Juliana'")
    speaker_labels: list[LabelUse] = Field(description="Every SPEAKERS label that is this character")
    role: str = Field(description="Archetype and relationship in a few words, e.g. 'heroine, jilted bride'")
    importance: Literal["lead", "major", "minor"]

    @field_validator("speaker_labels", mode="before")
    @classmethod
    def _strings_to_objects(cls, v):
        # Models sometimes return plain label strings; accept them as "every episode".
        if isinstance(v, list):
            return [{"label": x} if isinstance(x, str) else x for x in v]
        return v


class Thread(BaseModel):
    id: str = Field(description="Short kebab-case id, e.g. 'secret-identity'")
    title: str
    summary: str = Field(description="One or two sentences")
    episodes: list[int]
    title_premise: bool = Field(description="True if the show title refers to this thread")


class EpisodeNote(BaseModel):
    episode: int
    threads: list[str] = Field(description="Ids of the threads this episode advances")
    key_reveals: list[str] = Field(description="Reveals, twists or status reversals in this episode; [] if none")


class StoryMap(BaseModel):
    premise: str = Field(description="The show's core hook in one sentence")
    characters: list[Character]
    threads: list[Thread]
    episodes: list[EpisodeNote]


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM = """You are a story editor for short vertical microdramas. You turn a series outline into a precise \
story map that later steps use to pick promo clips. Use only what the material supports; never invent events \
or names."""

TASK = """SHOW TITLE: {title}

Build the story map.
1. premise: the show's core hook in one sentence.
2. characters: every recurring or plot-relevant character, with every SPEAKERS label that refers to them.
   - Descriptive labels (e.g. "THE FORMER GROOM", "LADY IN GOLD GOWN") often stand for a named character. \
Use the sample lines and the shot before each line to resolve them.
   - A descriptive label can mean different people in different episodes. Samples come from a label's first, \
middle and last episode: compare them. If the label changes person, list it under each character with the \
episodes where it is that character.
   - Leave generic labels you can't pin down (e.g. "GUARD") unassigned.
   - In any one episode, a label belongs to at most one character. Copy labels exactly as written.
   - name: the short name the script itself uses (usually their label, e.g. "Juliana"). Don't add surnames, \
titles or secret identities the labels don't use.
3. threads: 4-12 story threads (arcs) that run across episodes. A title often names two halves, a situation \
and a twist; set title_premise=true on the threads for both halves.
4. episodes: exactly one entry per episode number below, with the thread ids it advances and its key reveals.

EPISODES
{episodes}

SPEAKERS (label | lines | episodes | sample lines from its first, middle and last episode, each with the shot \
just before it)
{speakers}"""


def ep_ranges(nums: list[int]) -> str:
    """[1, 2, 3, 5, 7, 8] -> '1-3, 5, 7-8'"""
    out, nums = [], sorted(set(nums))
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}" if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


def _shot_before(beats: list[Beat], idx: int, lookback: int = 4) -> str | None:
    for k in range(idx - 1, max(-1, idx - 1 - lookback), -1):
        if beats[k].type in ("camera", "action"):
            return beats[k].text
    return None


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def speaker_samples(show: Show) -> dict[str, list[str]]:
    """Per speaker: one line from its first, middle and last episode (two lines if it's in one episode),
    each with the shot just before it."""
    where: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for ep in show.episodes:
        for i, b in enumerate(ep.beats):
            if b.type == "dialogue" and b.speaker:
                where[b.speaker][ep.episode].append(i)
    samples: dict[str, list[str]] = {}
    for speaker, per_ep in where.items():
        eps = sorted(per_ep)
        picks = sorted({eps[0], eps[len(eps) // 2], eps[-1]})
        per = 2 if len(picks) == 1 else 1
        lines = []
        for n in picks:
            ep = show.episode(n)
            for i in per_ep[n][:per]:
                b = ep.beats[i]
                tags = ", ".join(b.emotion)
                line = f'ep{n}: "{_clip(b.text, 110)}"' + (f" ({tags})" if tags else "")
                shot = _shot_before(ep.beats, i)
                if shot:
                    line += f"  [shot before: {_clip(shot, 130)}]"
                lines.append(line)
        samples[speaker] = lines
    return samples


def build_prompt(show: Show, title: str | None = None) -> str:
    episodes = "\n".join(
        f"[{ep.episode}] LOGLINE: {ep.logline}\n    CONTINUITY: {ep.continuity}" for ep in show.episodes
    )
    samples = speaker_samples(show)
    speakers = []
    for name, row in show.speakers.items():
        speakers.append(f"{name} | {row['lines']} lines | eps {ep_ranges(row['episodes'])}")
        speakers += [f"    {s}" for s in samples.get(name, [])]
    return TASK.format(
        title=title or "(unknown: infer the premise from the episodes)",
        episodes=episodes,
        speakers="\n".join(speakers),
    )


# --------------------------------------------------------------------------- #
# Checks + aliases
# --------------------------------------------------------------------------- #


def label_episodes(show: Show) -> dict[str, list[int]]:
    """{speaker label -> episodes it speaks in}, from the parsed show (before aliases)."""
    return {name: row["episodes"] for name, row in show.speakers.items()}


def check_story_map(sm: StoryMap, label_eps: dict[str, list[int]], episodes: list[int]) -> list[str]:
    """Semantic rules the schema can't express. Returned errors are fed back to the model."""
    errs: list[str] = []
    names = [c.name.strip().lower() for c in sm.characters]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        errs.append(f"character {dup!r} is listed more than once; merge the entries")

    owner: dict[tuple[str, int], str] = {}
    clashes: set[tuple[str, str, str]] = set()
    for c in sm.characters:
        for use in c.speaker_labels:
            key = clean_name(use.label)
            if key not in label_eps:
                errs.append(f"character {c.name!r}: speaker label {use.label!r} is not in SPEAKERS (copy labels exactly)")
                continue
            valid = set(label_eps[key])
            bad = sorted(set(use.episodes) - valid)
            if bad:
                errs.append(
                    f"character {c.name!r}: label {key!r} does not speak in episodes {bad} "
                    f"(it speaks in {ep_ranges(label_eps[key])})"
                )
            for n in sorted(set(use.episodes or valid) & valid):
                prev = owner.setdefault((key, n), c.name)
                if prev != c.name:
                    clashes.add((key, prev, c.name))
    for key, a, b in sorted(clashes):
        errs.append(
            f"label {key!r} is given to both {a!r} and {b!r} in the same episode(s); "
            "if it changes person, give each character only its own episodes"
        )

    valid_eps = set(episodes)
    thread_ids = [t.id for t in sm.threads]
    for dup in sorted({t for t in thread_ids if thread_ids.count(t) > 1}):
        errs.append(f"thread id {dup!r} is used more than once")
    for t in sm.threads:
        bad = sorted(set(t.episodes) - valid_eps)
        if bad:
            errs.append(f"thread {t.id!r} lists episodes that don't exist: {bad}")
    if sm.threads and not any(t.title_premise for t in sm.threads):
        errs.append("no thread has title_premise=true; mark the thread(s) the show title refers to")

    got = [e.episode for e in sm.episodes]
    missing = sorted(valid_eps - set(got))
    extra = sorted(set(got) - valid_eps)
    dups = sorted({n for n in got if got.count(n) > 1})
    if missing:
        errs.append(f"episodes missing an entry: {ep_ranges(missing)}")
    if extra:
        errs.append(f"entries for episodes that don't exist: {extra}")
    if dups:
        errs.append(f"episodes with more than one entry: {dups}")
    known = set(thread_ids)
    for e in sm.episodes:
        bad = sorted(set(e.threads) - known)
        if bad:
            errs.append(f"episode {e.episode} uses unknown thread ids {bad}")
    return errs


def canonical_name(c: Character, label_eps: dict[str, list[int]]) -> str:
    """Ground the display name in the script: the longest of the character's own labels
    that appears inside the model's name ('Duke Tom Baker' -> 'TOM').
    Stops the model inventing surnames or leaking a secret identity into speaker names."""
    name = clean_name(c.name)
    own = {clean_name(u.label) for u in c.speaker_labels} & set(label_eps)
    inside = [lab for lab in own if re.search(rf"(?<![\w']){re.escape(lab)}(?![\w'])", name)]
    if inside:
        return max(inside, key=lambda lab: (len(lab), len(label_eps[lab])))
    return name


def aliases_from(sm: StoryMap, label_eps: dict[str, list[int]]) -> dict[str, str]:
    """Alias map for parser.normalise_speakers. Keys are 'LABEL' (every episode) or
    'LABEL@N' (episode N only). Unknown labels are dropped; first owner wins a clash."""
    out: dict[str, str] = {}
    taken: set[tuple[str, int]] = set()
    for c in sm.characters:
        canon = canonical_name(c, label_eps)
        for use in c.speaker_labels:
            key = clean_name(use.label)
            if key not in label_eps:
                continue
            all_eps = label_eps[key]
            eps = [n for n in (use.episodes or all_eps) if n in all_eps and (key, n) not in taken]
            taken.update((key, n) for n in eps)
            if not eps or key == canon:
                continue
            if set(eps) == set(all_eps):
                out[key] = canon
            else:
                out.update({f"{key}@{n}": canon for n in eps})
    return out


def describe_aliases(aliases: dict[str, str]) -> list[str]:
    """Readable lines, folding 'LABEL@N' keys into episode ranges."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key, canon in aliases.items():
        label, _, ep = key.partition("@")
        grouped[(label, canon)].append(int(ep) if ep else -1)
    lines = []
    for (label, canon), eps in sorted(grouped.items(), key=lambda kv: (kv[0][0], min(kv[1]))):
        where = "" if eps == [-1] else f" (eps {ep_ranges([e for e in eps if e >= 0])})"
        lines.append(f"{label}{where} -> {canon}")
    return lines


def build_story_map(show: Show, client: LLMClient, title: str | None = None) -> tuple[StoryMap, dict[str, str]]:
    """Run step 2. Returns the story map and the alias map (not yet applied to `show`)."""
    label_eps = label_episodes(show)
    episodes = [ep.episode for ep in show.episodes]
    sm = client.complete_json(
        step="story_map",
        system=SYSTEM,
        prompt=build_prompt(show, title),
        schema=StoryMap,
        check=lambda m: check_story_map(m, label_eps, episodes),
        on_check_fail="accept",  # a few bad labels shouldn't sink the run; aliases_from drops them
    )
    return sm, aliases_from(sm, label_eps)
