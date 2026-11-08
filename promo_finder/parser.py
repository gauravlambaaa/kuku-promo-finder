"""Step 1 — parse screenplay .txt files into timed beats. Pure code, no LLM.

Expected input (one file per episode)::

    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL – DAY
    Scene start: 00:00.00
    Scene end: 02:18.00
    **CAMERA:**                          <- block type, or a speaker
    Start: (00:00.00) | End: (00:04.80)
    <one or more lines of text>
    **MATRIARCH (O.S.) (Cold, Commanding):**
    Start: (00:11.30) | End: (00:13.10)
    "Stop that thief!"
    --- SCRIPT END ---

Design choices:
- Lenient. Anything unexpected becomes a warning on the episode, never an
  exception, so a new show with slightly different formatting still parses.
- Beats are sorted by start time (stable, so file order breaks ties) before IDs
  are assigned. The scripts are *mostly* chronological but not always, and later
  steps treat "beat A .. beat B" as a contiguous time range.
- Speaker names are normalised in two layers:
    1. deterministic, show-wide: strip delivery tags (O.S./V.O.), uppercase,
       and fold a titled form into the bare name only when the bare name is also
       used as a speaker ("LADY JULIANA" -> "JULIANA"). Never folds when two
       titled forms compete for the same bare name ("LORD X" vs "LADY X").
    2. optional alias map (from the story-map step, or a hand-written file) for
       descriptive names regex can't resolve ("THE FORMER GROOM" -> "GERALD").
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Beat:
    beat_id: str  # "E001-B023": unique across the show, ordered by start time
    episode: int
    scene: int | None
    start: float  # seconds from episode start
    end: float
    type: str  # dialogue | camera | action | sfx | music | transition
    speaker: str | None  # canonical speaker (dialogue only)
    speaker_raw: str | None  # name as written, minus tags
    delivery: str | None  # "O.S." | "V.O." | None
    emotion: list[str]  # free-text intensity tags from the header
    continues: bool  # dialogue that continues the previous line — don't cut before it
    text: str
    line: int  # 1-based line of the block header in the source file


@dataclass
class Scene:
    scene: int
    heading: str = ""
    start: float | None = None
    end: float | None = None


@dataclass
class Episode:
    episode: int
    episode_id: str | None  # platform id from the filename, if present
    source_file: str
    logline: str
    continuity: str
    duration: float
    scenes: list[Scene]
    beats: list[Beat]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Show:
    source_dir: str
    episodes: list[Episode]
    # canonical speaker -> {"variants": [...], "lines": int, "episodes": [...]}
    speakers: dict[str, dict]
    warnings: list[str] = field(default_factory=list)  # show-level (cross-episode)

    def episode(self, n: int) -> Episode:
        for ep in self.episodes:
            if ep.episode == n:
                return ep
        raise KeyError(f"episode {n} not found")

    def beat(self, beat_id: str) -> Beat:
        """Look up a beat by ID — this is how LLM-returned IDs become timestamps."""
        ep = self.episode(int(beat_id[1:4]))
        for b in ep.beats:
            if b.beat_id == beat_id:
                return b
        raise KeyError(f"beat {beat_id} not found")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #

_TS = r"\d+(?::\d{1,2}){1,2}(?:\.\d+)?"  # MM:SS.ss, also H:MM:SS(.ss) and MM:SS

SCRIPT_START_RE = re.compile(r"^-{2,}\s*SCRIPT\s+START\s*-{2,}$", re.I)
SCRIPT_END_RE = re.compile(r"^-{2,}\s*SCRIPT\s+END\s*-{2,}$", re.I)
META_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*:?$", re.I)
SCENE_HEADING_RE = re.compile(r"^Scene\s+heading\s*:\s*(.*)$", re.I)
SCENE_START_RE = re.compile(rf"^Scene\s+start\s*:\s*\(?\s*({_TS})", re.I)
SCENE_END_RE = re.compile(rf"^Scene\s+end\s*:\s*\(?\s*({_TS})", re.I)
SKIP_RE = re.compile(r"^Actions?\s+and\s+dialogue\s*:?$", re.I)
HEADER_RE = re.compile(r"^\*\*(.+?)\*\*\s*:?$")
TIMING_RE = re.compile(
    rf"^Start\s*:\s*\(?\s*({_TS})\s*\)?\s*\|\s*End\s*:\s*\(?\s*({_TS})\s*\)?$", re.I
)
FILENAME_RE = re.compile(r"episode[-_ ]?(\d+)(?:.*?episode\s*(\d+))?", re.I)

# Block headers that are not dialogue. Any other header is a speaker.
NON_DIALOGUE_TYPES = {
    "CAMERA": "camera",
    "ACTION": "action",
    "SOUND FX": "sfx",
    "SFX": "sfx",
    "MUSIC": "music",
    "TRANSITION": "transition",
}

DELIVERY_RE = [
    (re.compile(r"^(O\.?\s?S\.?|OFF[- ]?SCREEN|OFF[- ]?CAMERA|O\.?\s?C\.?)$", re.I), "O.S."),
    (re.compile(r"^(V\.?\s?O\.?|VOICE[- ]?OVER)$", re.I), "V.O."),
]
# "Continuing", "Continues", "Voice Continuing", "CONT'D", "Continuing taunt" (-> "taunt").
# The \b keeps "Contemptuous" / "Contemplative" out.
CONTINUES_RE = re.compile(r"^(?:voice\s+)?(?:cont(?:inu(?:ing|es|ed))?|cont'?d)\b\.?\s*(.*)$", re.I)

# Titles folded into the bare name when the bare name is also a speaker.
TITLES = (
    "THE", "LADY", "LORD", "SIR", "DAME", "MR", "MRS", "MS", "MISS", "MASTER", "MISTRESS",
    "DUKE", "DUCHESS", "PRINCE", "PRINCESS", "KING", "QUEEN", "COUNT", "COUNTESS",
    "BARON", "BARONESS", "CAPTAIN", "DR",
)
TITLE_RE = re.compile(rf"^(?:{'|'.join(TITLES)})\.?\s+(.+)$")

QUOTES = "\"'“”‘’"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def parse_ts(s: str) -> float:
    """'01:23.45' -> 83.45 seconds. Also accepts 'H:MM:SS.ss' and 'MM:SS'."""
    parts = s.strip().split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return round(hours * 3600 + minutes * 60 + seconds, 3)


def fmt_ts(seconds: float) -> str:
    """83.45 -> '01:23.45' (the script's own MM:SS.ss format)."""
    minutes, centis = divmod(round(seconds * 100), 6000)
    return f"{minutes:02d}:{centis / 100:05.2f}"


def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip(" .:-").upper()


def parse_header(header: str) -> tuple[str, str | None, list[str], bool]:
    """Split a block header into (name, delivery, emotion tags, continues).

    'MATRIARCH (O.S.) (Firm):'           -> ('MATRIARCH', 'O.S.', ['Firm'], False)
    'GUARD (O.S. / Low, Threatening):'   -> ('GUARD', 'O.S.', ['Low', 'Threatening'], False)
    'THE PRIEST (Off-Screen, Booming):'  -> ('THE PRIEST', 'O.S.', ['Booming'], False)
    'CLARISSA (O.S.) (Continuing taunt)' -> ('CLARISSA', 'O.S.', ['taunt'], True)
    """
    header = header.strip().rstrip(":").strip()
    name = clean_name(header.split("(", 1)[0])
    delivery: str | None = None
    emotion: list[str] = []
    continues = False
    for group in re.findall(r"\(([^)]*)\)", header):
        for part in group.split("/"):
            for tok in part.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                matched = next((label for rx, label in DELIVERY_RE if rx.match(tok)), None)
                if matched:
                    delivery = matched
                    continue
                m = CONTINUES_RE.match(tok)
                if m:
                    continues = True
                    if m.group(1):
                        emotion.append(m.group(1).strip())
                    continue
                emotion.append(tok)
    return name, delivery, emotion, continues


def strip_outer_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] in QUOTES and text[-1] in QUOTES:
        return text[1:-1].strip()
    return text


# --------------------------------------------------------------------------- #
# Episode parser
# --------------------------------------------------------------------------- #


@dataclass
class _RawBlock:
    header: str
    line: int
    start: float
    end: float
    text: str
    scene: int | None


def parse_episode(path: str | Path) -> Episode:
    """Parse one episode file. Speakers are cleaned but not yet show-normalised."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    warnings: list[str] = []
    meta: dict[str, str] = {}
    scenes: list[Scene] = []
    blocks: list[_RawBlock] = []

    state = "meta"  # meta -> script -> done
    last_meta: str | None = None
    cur_scene: Scene | None = None
    i = 0
    n = len(lines)

    def is_block_boundary(s: str) -> bool:
        return bool(HEADER_RE.match(s) or SCENE_RE.match(s) or SCRIPT_END_RE.match(s))

    while i < n:
        s = lines[i].strip()

        if SCRIPT_START_RE.match(s):
            state = "script"
            i += 1
            continue
        if SCRIPT_END_RE.match(s):
            state = "done"
            i += 1
            continue

        if state == "meta":
            # Tolerate a file with no SCRIPT START marker.
            if SCENE_RE.match(s) or HEADER_RE.match(s):
                state = "script"
                warnings.append(f"line {i + 1}: script content before '--- SCRIPT START ---'")
            else:
                m = META_RE.match(s)
                if m:
                    last_meta = m.group(1).upper()
                    meta[last_meta] = m.group(2).strip()
                elif s and last_meta:  # wrapped meta value
                    meta[last_meta] += " " + s
                i += 1
                continue

        if state == "done":
            if s:
                warnings.append(f"line {i + 1}: text after '--- SCRIPT END ---' ignored")
            i += 1
            continue

        if not s or SKIP_RE.match(s):
            i += 1
            continue

        if m := SCENE_RE.match(s):
            cur_scene = Scene(scene=int(m.group(1)))
            scenes.append(cur_scene)
            i += 1
            continue
        if m := SCENE_HEADING_RE.match(s):
            if cur_scene:
                cur_scene.heading = m.group(1).strip()
            i += 1
            continue
        if m := SCENE_START_RE.match(s):
            if cur_scene:
                cur_scene.start = parse_ts(m.group(1))
            i += 1
            continue
        if m := SCENE_END_RE.match(s):
            if cur_scene:
                cur_scene.end = parse_ts(m.group(1))
            i += 1
            continue

        if m := HEADER_RE.match(s):
            header, header_line = m.group(1), i + 1
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            tm = TIMING_RE.match(lines[i].strip()) if i < n else None
            if not tm:
                warnings.append(f"line {header_line}: block {header!r} has no Start/End line; skipped")
                continue
            start, end = parse_ts(tm.group(1)), parse_ts(tm.group(2))
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            body: list[str] = []
            while i < n:
                t = lines[i].strip()
                if not t or is_block_boundary(t):
                    break
                body.append(t)
                i += 1
            blocks.append(
                _RawBlock(
                    header=header,
                    line=header_line,
                    start=start,
                    end=end,
                    text=" ".join(body),
                    scene=cur_scene.scene if cur_scene else None,
                )
            )
            continue

        warnings.append(f"line {i + 1}: unrecognised line {s[:60]!r}")
        i += 1

    # ---- episode number ---------------------------------------------------- #
    fm = FILENAME_RE.search(path.stem)
    file_num = int(fm.group(1)) if fm else None
    episode_id = fm.group(2) if fm and fm.group(2) else None
    hm = re.search(r"(\d+)", meta.get("EPISODE NUMBER", ""))
    header_num = int(hm.group(1)) if hm else None
    if header_num is not None and file_num is not None and header_num != file_num:
        warnings.append(f"episode number {header_num} in header != {file_num} in filename; using header")
    ep_num = header_num if header_num is not None else file_num
    if ep_num is None:
        raise ValueError(f"{path.name}: no episode number in header or filename")

    # ---- blocks -> beats (sorted by start, stable) ------------------------- #
    beats: list[Beat] = []
    for blk in sorted(blocks, key=lambda b: b.start):
        if blk.end < blk.start:
            warnings.append(
                f"line {blk.line}: end {fmt_ts(blk.end)} before start {fmt_ts(blk.start)}; end clamped to start"
            )
            blk.end = blk.start
        name, delivery, emotion, continues = parse_header(blk.header)
        btype = NON_DIALOGUE_TYPES.get(name)
        if btype is None and "(" not in blk.header:
            warnings.append(f"line {blk.line}: header {blk.header!r} has no tags; treated as a speaker")
        text = blk.text
        if btype is None:
            text = strip_outer_quotes(text)
            if text.startswith(("...", "…")):
                continues = True
        if not text:
            warnings.append(f"line {blk.line}: block {blk.header!r} has no text")
        beats.append(
            Beat(
                beat_id=f"E{ep_num:03d}-B{len(beats) + 1:03d}",
                episode=ep_num,
                scene=blk.scene,
                start=blk.start,
                end=blk.end,
                type=btype or "dialogue",
                speaker=None if btype else name,
                speaker_raw=None if btype else name,
                delivery=delivery,
                emotion=emotion,
                continues=continues and btype is None,
                text=text,
                line=blk.line,
            )
        )

    # ---- sanity checks ------------------------------------------------------ #
    if not scenes:
        warnings.append("no SCENE markers found")
    if not beats:
        warnings.append("no beats found")
    elif not any(b.type == "dialogue" for b in beats):
        warnings.append("no dialogue found")
    for sc in scenes:
        if sc.start is None or sc.end is None:
            warnings.append(f"scene {sc.scene}: missing start/end time")
    scene_by_num = {sc.scene: sc for sc in scenes}
    tol = 0.5
    for b in beats:
        sc = scene_by_num.get(b.scene)
        if sc and sc.start is not None and sc.end is not None:
            if b.start < sc.start - tol or b.end > sc.end + tol:
                warnings.append(
                    f"{b.beat_id} (line {b.line}): {fmt_ts(b.start)}-{fmt_ts(b.end)} "
                    f"outside scene {sc.scene} ({fmt_ts(sc.start)}-{fmt_ts(sc.end)})"
                )

    ends = [b.end for b in beats] + [sc.end for sc in scenes if sc.end is not None]
    return Episode(
        episode=ep_num,
        episode_id=episode_id,
        source_file=path.name,
        logline=meta.get("EPISODE LOGLINE", ""),
        continuity=meta.get("CONTINUITY NOTES", ""),
        duration=max(ends) if ends else 0.0,
        scenes=scenes,
        beats=beats,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Show-level: speaker normalisation
# --------------------------------------------------------------------------- #


def build_title_map(names: set[str]) -> dict[str, str]:
    """Fold 'LADY JULIANA' -> 'JULIANA' when 'JULIANA' is itself a speaker.

    Skipped when two titled forms share a bare name ('LORD ASHCROFT' and
    'LADY ASHCROFT'), since those are probably different people.
    """
    variants: dict[str, list[str]] = defaultdict(list)
    for name in names:
        m = TITLE_RE.match(name)
        if m and m.group(1) in names:
            variants[m.group(1)].append(name)
    return {v[0]: bare for bare, v in variants.items() if len(v) == 1}


def normalise_speakers(show: Show, aliases: dict[str, str] | None = None) -> None:
    """(Re)compute canonical speaker names and the show's speaker table, in place.

    Safe to call again later with an alias map from the story-map step. Alias keys
    are 'LABEL' (every episode) or 'LABEL@N' (episode N only, for labels reused for
    different people). Episode-specific keys win.
    """
    aliases = {clean_name(k): clean_name(v) for k, v in (aliases or {}).items()}
    raw_names = {b.speaker_raw for ep in show.episodes for b in ep.beats if b.speaker_raw}
    title_map = build_title_map(raw_names)

    table: dict[str, dict] = {}
    for ep in show.episodes:
        for b in ep.beats:
            if not b.speaker_raw:
                continue
            name = title_map.get(b.speaker_raw, b.speaker_raw)
            for key in (f"{b.speaker_raw}@{ep.episode}", f"{name}@{ep.episode}", b.speaker_raw, name):
                if key in aliases:
                    name = aliases[key]
                    break
            b.speaker = name
            row = table.setdefault(name, {"variants": set(), "lines": 0, "episodes": set()})
            row["variants"].add(b.speaker_raw)
            row["lines"] += 1
            row["episodes"].add(ep.episode)

    show.speakers = {
        name: {
            "variants": sorted(row["variants"]),
            "lines": row["lines"],
            "episodes": sorted(row["episodes"]),
        }
        for name, row in sorted(table.items(), key=lambda kv: -kv[1]["lines"])
    }


def parse_show(input_dir: str | Path, aliases: dict[str, str] | None = None) -> Show:
    """Parse every *.txt episode in a folder into a Show, ordered by episode number."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt episode files in {input_dir}")
    episodes = sorted((parse_episode(f) for f in files), key=lambda e: e.episode)

    show_warnings: list[str] = []
    counts = Counter(ep.episode for ep in episodes)
    for num, c in counts.items():
        if c > 1:
            show_warnings.append(f"episode {num} appears in {c} files")
    nums = sorted(counts)
    missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
    if missing:
        show_warnings.append(f"missing episodes: {missing}")

    show = Show(source_dir=str(input_dir), episodes=episodes, speakers={}, warnings=show_warnings)
    normalise_speakers(show, aliases)
    return show


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def format_beat(b: Beat) -> str:
    """One compact line per beat — for eyeballing, and the base of LLM prompts."""
    span = f"[{fmt_ts(b.start)}-{fmt_ts(b.end)}]"
    if b.type == "dialogue":
        tags = ", ".join(([b.delivery] if b.delivery else []) + b.emotion)
        who = f"{b.speaker}" + (f" ({tags})" if tags else "")
        return f'{b.beat_id} {span} {who}: "{b.text}"'
    return f"{b.beat_id} {span} {b.type.upper()}: {b.text}"


def summary(show: Show) -> str:
    out = []
    out.append(f"{'ep':>4} {'beats':>5} {'dlg':>4} {'cam':>4} {'act':>4} {'sfx':>4} {'scn':>3} {'dur':>8}  warn")
    for ep in show.episodes:
        c = Counter(b.type for b in ep.beats)
        out.append(
            f"{ep.episode:>4} {len(ep.beats):>5} {c['dialogue']:>4} {c['camera']:>4} {c['action']:>4} "
            f"{c['sfx']:>4} {len(ep.scenes):>3} {fmt_ts(ep.duration):>8}  {len(ep.warnings) or ''}"
        )
    total = sum(len(ep.beats) for ep in show.episodes)
    dur = sum(ep.duration for ep in show.episodes)
    out.append(f"\n{len(show.episodes)} episodes, {total} beats, {dur / 60:.1f} min of runtime")

    merged = {k: v for k, v in show.speakers.items() if len(v["variants"]) > 1 or v["variants"] != [k]}
    out.append(f"{len(show.speakers)} speakers; merged name variants:")
    for name, row in merged.items():
        out.append(f"  {name:<22} <- {', '.join(row['variants'])}")

    warns = [(ep.episode, w) for ep in show.episodes for w in ep.warnings]
    out.append(f"\n{len(warns) + len(show.warnings)} warnings")
    for w in show.warnings:
        out.append(f"  show: {w}")
    for num, w in warns:
        out.append(f"  ep {num}: {w}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parse screenplay episodes into timed beats (no LLM).")
    ap.add_argument("input", help="folder of episode .txt files")
    ap.add_argument("--out", default="outputs/raw/beats.json", help="where to write the parsed JSON")
    ap.add_argument("--aliases", help="optional JSON file mapping speaker name -> canonical name")
    ap.add_argument("--show", type=int, metavar="EP", help="print one episode's beats and exit")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    aliases = json.loads(Path(args.aliases).read_text(encoding="utf-8")) if args.aliases else None
    show = parse_show(args.input, aliases)

    if args.show is not None:
        ep = show.episode(args.show)
        print(f"Episode {ep.episode} ({ep.source_file}), {fmt_ts(ep.duration)}")
        print(f"LOGLINE: {ep.logline}\n")
        for b in ep.beats:
            print(format_beat(b))
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(show.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(summary(show))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
