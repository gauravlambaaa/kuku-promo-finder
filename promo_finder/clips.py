"""Step 5: turn (hook beat, cliffhanger beat) into a clip with real cut points, and check it. Pure code.

Cut rules:
  - start on the hook beat. If someone is mid-line at that moment, back up to the start of
    that line, so we never open mid-sentence.
  - end right after the cliffhanger beat. If the line continues in the next beat ("..."), or
    another line is still being spoken at that moment, extend to the end of it, so we never
    cut mid-line.
  - every cut is a beat edge (start or end of a script block), so it maps to a real moment.
Checks:
  - hard errors (clip rejected): beats missing / out of order, length outside 20-60s,
    hook payoff more than 3s after the clip starts.
  - soft flags (kept, shown to the editor): edges that had to move, a hook that opens
    mid-sentence, a long hook line whose payoff may land late.
The same function is used inside the LLM fix loop (step 3), so the model gets the
exact errors and can pick different beats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parser import Beat, Episode, fmt_ts

MIN_S, MAX_S = 20.0, 60.0
HOOK_WINDOW_S = 3.0
LONG_HOOK_LINE_S = 4.5
EPS = 0.05


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    start_beat: str
    end_beat: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "start": fmt_ts(self.start),
            "end": fmt_ts(self.end),
            "start_s": self.start,
            "end_s": self.end,
            "start_beat": self.start_beat,
            "end_beat": self.end_beat,
        }


@dataclass
class Cut:
    segment: Segment | None
    errors: list[str] = field(default_factory=list)  # hard: clip is rejected
    flags: list[str] = field(default_factory=list)  # soft: kept, shown to the editor
    hook_offset: float = 0.0

    @property
    def ok(self) -> bool:
        return self.segment is not None and not self.errors


def normalise_id(beat_id: str, episode: int) -> str:
    """Accept 'B023' or 'e1-b23' style IDs from the model; return 'E001-B023'."""
    s = beat_id.strip().upper()
    if s.startswith("B"):
        s = f"E{episode:03d}-{s}"
    if "-" in s:
        e, b = s.split("-", 1)
        try:
            return f"E{int(e.lstrip('E')):03d}-B{int(b.lstrip('B')):03d}"
        except ValueError:
            return s
    return s


def cut_clip(ep: Episode, hook_id: str, cliff_id: str) -> Cut:
    idx = {b.beat_id: i for i, b in enumerate(ep.beats)}
    hook_id, cliff_id = normalise_id(hook_id, ep.episode), normalise_id(cliff_id, ep.episode)
    errors = [f"beat {x} is not in episode {ep.episode}" for x in (hook_id, cliff_id) if x not in idx]
    if errors:
        return Cut(None, errors)
    hook, cliff = ep.beats[idx[hook_id]], ep.beats[idx[cliff_id]]
    if cliff.end <= hook.start:
        return Cut(None, [f"cliffhanger {cliff_id} must come after hook {hook_id}"])

    lines = [b for b in ep.beats if b.type == "dialogue"]
    flags: list[str] = []

    # ---- start ------------------------------------------------------------- #
    start, start_beat = hook.start, hook.beat_id
    if hook.type == "dialogue" and hook.continues:
        flags.append(f"hook {hook_id} continues an earlier line, so the clip opens mid-sentence")
    for b in lines:
        if b is not hook and b.start < start - EPS and b.end > start + EPS:
            flags.append(f"start moved back {start - b.start:.1f}s so {b.beat_id} isn't cut mid-line")
            start, start_beat = b.start, b.beat_id

    # ---- end --------------------------------------------------------------- #
    end, end_beat = cliff.end, cliff.beat_id
    changed = True
    while changed:
        changed = False
        for b in lines:
            straddles = b.start < end - EPS and b.end > end + EPS
            continues_line = (
                b.continues and b.start >= end - EPS and b.start - end <= 1.0
                and (cliff.type != "dialogue" or b.speaker == cliff.speaker)
            )
            if straddles or continues_line:
                why = "isn't cut mid-line" if straddles else "finishes the sentence"
                flags.append(f"end moved {b.end - end:.1f}s later so {b.beat_id} {why}")
                end, end_beat = b.end, b.beat_id
                changed = True

    # ---- checks ------------------------------------------------------------ #
    errors = []
    hook_offset = round(hook.start - start, 2)
    if hook_offset > HOOK_WINDOW_S:
        errors.append(f"hook {hook_id} starts {hook_offset:.1f}s into the clip; it must land within {HOOK_WINDOW_S:.0f}s")
    duration = end - start
    if duration < MIN_S:
        errors.append(
            f"clip {hook_id} -> {cliff_id} is {duration:.1f}s; promos must be at least {MIN_S:.0f}s, "
            "so choose an earlier hook or a later cliffhanger"
        )
    if duration > MAX_S:
        errors.append(
            f"clip {hook_id} -> {cliff_id} is {duration:.1f}s; promos must be at most {MAX_S:.0f}s, "
            "so choose a later hook or an earlier cliffhanger"
        )
    if hook.type == "dialogue" and hook.end - hook.start > LONG_HOOK_LINE_S and hook_offset + (hook.end - hook.start) > HOOK_WINDOW_S + 2:
        flags.append(f"hook line runs {hook.end - hook.start:.1f}s, so its payoff may land after the first 3s")

    seg = Segment(ep.episode, start, end, start_beat, end_beat)
    return Cut(seg, errors, flags, hook_offset)


def clip_beats(ep: Episode, seg: Segment, *, with_camera: bool = True) -> list[Beat]:
    """Beats inside a segment, in order (for judging/reporting)."""
    keep = ("dialogue", "action", "sfx") + (("camera",) if with_camera else ())
    return [b for b in ep.beats if b.type in keep and b.start >= seg.start - EPS and b.end <= seg.end + EPS]
