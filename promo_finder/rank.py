"""Step 7: validate every candidate clip, rank the survivors, and write the report. Pure code.

Score = weighted rubric scores from step 3, minus a small penalty per cut flag.
Ranking keeps the list varied: a clip that overlaps a higher-ranked one is dropped,
and each episode gets at most `per_episode` promos.
Weak picks are flagged for the human editor rather than silently dropped.
"""
from __future__ import annotations

from collections import Counter

from .candidates import EpisodeCandidates
from .clips import Segment, clip_beats, cut_clip, normalise_id
from .parser import Beat, Show

WEIGHTS = {"hook": 0.25, "cliffhanger": 0.25, "standalone": 0.20, "intensity": 0.10, "premise": 0.10, "pacing": 0.10}
FLAG_PENALTY = 0.3
MAX_SPEAKERS = 4


def _line(b: Beat) -> str:
    if b.type == "dialogue":
        return f'{b.speaker}: "{b.text}"'
    return f"{b.type.upper()}: {b.text}"


def _review_notes(show: Show, rec: dict, seg: Segment) -> list[str]:
    """Reasons an editor should double-check this pick."""
    s = rec["scores"]
    notes = []
    if s["standalone"] <= 5:
        notes.append(f"low standalone score ({s['standalone']}): a new viewer may not follow it")
    if s["hook"] <= 5:
        notes.append(f"weak hook ({s['hook']})")
    if s["cliffhanger"] <= 5:
        notes.append(f"weak cliffhanger ({s['cliffhanger']})")
    speakers = {b.speaker for b in clip_beats(show.episode(seg.episode), seg) if b.type == "dialogue"}
    if len(speakers) > MAX_SPEAKERS:
        notes.append(f"{len(speakers)} speaking characters; may be hard to follow cold")
    return notes


def build_promos(
    show: Show, cands: dict[int, EpisodeCandidates], models: dict[int, str] | None = None
) -> tuple[list[dict], list[dict]]:
    """Cut + validate every candidate clip. Returns (accepted, rejected)."""
    accepted, rejected = [], []
    for n, c in cands.items():
        ep = show.episode(n)
        for i, clip in enumerate(c.clips, 1):
            hook_id = normalise_id(clip.hook_beat, n)
            cliff_id = normalise_id(clip.cliffhanger_beat, n)
            rec = {
                "episode": n,
                "candidate": i,
                "model": (models or {}).get(n),
                "hook_beat": hook_id,
                "cliffhanger_beat": cliff_id,
                "hook": clip.hook,
                "cliffhanger": clip.cliffhanger,
                "why": clip.why,
                "scores": clip.scores.model_dump(),
            }
            cut = cut_clip(ep, hook_id, cliff_id)
            if not cut.ok:
                rec["errors"] = cut.errors
                rejected.append(rec)
                continue
            seg = cut.segment
            rec.update(
                segments=[seg.to_dict()],
                duration_s=seg.duration,
                hook_offset_s=cut.hook_offset,
                hook_line=_line(show.beat(hook_id)),
                cliffhanger_line=_line(show.beat(cliff_id)),
                episode_ends_on_cliffhanger=c.ends_on_cliffhanger,
                cut_flags=cut.flags,
            )
            rec["review"] = _review_notes(show, rec, seg)
            weighted = sum(WEIGHTS[k] * v for k, v in rec["scores"].items())
            rec["score"] = round(weighted - FLAG_PENALTY * len(cut.flags), 2)
            accepted.append(rec)
    return accepted, rejected


def _overlap(a: dict, b: dict) -> float:
    sa, sb = a["segments"][0], b["segments"][0]
    if sa["episode"] != sb["episode"]:
        return 0.0
    inter = min(sa["end_s"], sb["end_s"]) - max(sa["start_s"], sb["start_s"])
    return max(0.0, inter) / min(a["duration_s"], b["duration_s"])


def _rank_reason(r: dict) -> str:
    s = r["scores"]
    strong = [f"{k} {v}" for k, v in sorted(s.items(), key=lambda kv: -kv[1]) if v >= 8][:3]
    weak = [f"{k} {v}" for k, v in s.items() if v <= 5]
    parts = []
    if strong:
        parts.append("strongest on " + ", ".join(strong))
    if s["premise"] >= 8:
        parts.append("sells the title premise")
    if weak:
        parts.append("held back by " + ", ".join(weak))
    if r["cut_flags"]:
        parts.append(f"{len(r['cut_flags'])} cut adjustment(s)")
    return f"score {r['score']:.2f}: " + ("; ".join(parts) or "solid but unremarkable across the rubric")


MIN_CALIBRATION_GROUP = 5


def calibrate(accepted: list[dict]) -> dict[str, float]:
    """When episodes were scored by different models (free-tier fallback), shift each model's
    scores so its mean matches the overall mean. Models with too few clips are left alone.
    Assumes each model's episodes are of similar underlying quality. Returns {model: shift}."""
    by_model: dict[str, list[float]] = {}
    for r in accepted:
        r["score_raw"] = r["score"]
        by_model.setdefault(r.get("model") or "?", []).append(r["score"])
    if len(by_model) < 2:
        return {}
    overall = sum(r["score"] for r in accepted) / len(accepted)
    shifts = {m: round(overall - sum(s) / len(s), 2) for m, s in by_model.items() if len(s) >= MIN_CALIBRATION_GROUP}
    for r in accepted:
        r["score"] = round(r["score_raw"] + shifts.get(r.get("model") or "?", 0.0), 2)
    return shifts


def rank(accepted: list[dict], per_episode: int = 2) -> tuple[list[dict], list[dict], dict[str, float]]:
    """Calibrate across models, order by score, drop overlaps, cap promos per episode.
    Returns (ranked, dropped, calibration shifts)."""
    shifts = calibrate(accepted)
    ranked, dropped = [], []
    per_ep: Counter = Counter()
    for r in sorted(accepted, key=lambda r: (-r["score"], r["episode"], r["candidate"])):
        if any(_overlap(r, k) > 0.5 for k in ranked):
            dropped.append({**r, "dropped": "overlaps a higher-ranked clip"})
        elif per_ep[r["episode"]] >= per_episode:
            dropped.append({**r, "dropped": f"episode already has {per_episode} promos"})
        else:
            per_ep[r["episode"]] += 1
            ranked.append(r)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        r["id"] = f"P{i:02d}"
        r["rank_reason"] = _rank_reason(r)
    return ranked, dropped, shifts


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _cell(s: str, n: int = 70) -> str:
    s = s.replace("|", "/").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def render_markdown(
    *, title: str | None, model: str, ranked: list[dict], dropped: list[dict], rejected: list[dict],
    cands: dict[int, EpisodeCandidates], failed: dict[int, str], top: int = 20,
    shifts: dict[str, float] | None = None,
) -> str:
    n_cand = sum(len(c.clips) for c in cands.values())
    out = [f"# Promo picks: {title or 'untitled show'}", ""]
    out.append(
        f"Models: {model}. {len(cands)} episodes analysed, {n_cand} candidate clips, "
        f"{n_cand - len(rejected)} passed validation, {len(rejected)} rejected, "
        f"{len(ranked)} ranked ({len(dropped)} dropped as overlaps or per-episode cap)."
    )
    if shifts:
        out += ["", "Scores came from more than one model (free-tier fallback), and the models score at different "
                "levels, so each model's scores were shifted to a common mean before ranking: "
                + ", ".join(f"{m} {s:+.2f}" for m, s in shifts.items())
                + ". This assumes each model's episodes are of similar quality. Raw scores are in promos.json."]
    out += ["", "Timestamps are MM:SS.ss from the start of the episode. Each promo is one continuous segment.", ""]

    out += [f"## Top {min(top, len(ranked))} promos (post in this order)", "",
            "| # | Ep | Segment | Len | Hook | Cliffhanger | Score |", "|---|---|---|---|---|---|---|"]
    for r in ranked[:top]:
        seg = r["segments"][0]
        out.append(
            f"| {r['rank']} | {r['episode']} | {seg['start']}–{seg['end']} | {r['duration_s']:.0f}s "
            f"| {_cell(r['hook'])} | {_cell(r['cliffhanger'])} | {r['score']:.2f}{' ⚠' if r['review'] else ''} |"
        )

    out += ["", "## Details", ""]
    for r in ranked[:top]:
        seg = r["segments"][0]
        s = r["scores"]
        out += [
            f"### {r['id']} · Episode {r['episode']} · {seg['start']}–{seg['end']} ({r['duration_s']:.1f}s)",
            f"- **Segments:** ep {seg['episode']} {seg['start']}–{seg['end']} (beats {seg['start_beat']} → {seg['end_beat']})",
            f"- **Hook** ({r['hook_beat']}, {r['hook_offset_s']:.1f}s in): {r['hook']}  \n  `{_cell(r['hook_line'], 160)}`",
            f"- **Cliffhanger** ({r['cliffhanger_beat']}): {r['cliffhanger']}  \n  `{_cell(r['cliffhanger_line'], 160)}`",
            f"- **Why it works:** {r['why']}",
            f"- **Why this rank:** {r['rank_reason']}",
            f"- **Scores:** " + " · ".join(f"{k} {v}" for k, v in s.items()),
        ]
        if r["cut_flags"]:
            out.append("- **Cut adjustments:** " + "; ".join(r["cut_flags"]))
        if r["review"]:
            out.append("- **⚠ Review:** " + "; ".join(r["review"]))
        out.append("")

    out += ["## Quality check: where the tool may have picked badly", ""]
    flagged = [r for r in ranked if r["review"] or r["cut_flags"]]
    out.append(f"**Flagged ranked picks ({len(flagged)}):**")
    out += [f"- {r['id']} (ep {r['episode']}): " + "; ".join(r["review"] + r["cut_flags"]) for r in flagged] or ["- none"]
    out += ["", f"**Rejected by the validator ({len(rejected)}):** the model proposed these, but they break a hard rule."]
    out += [f"- ep {r['episode']} clip {r['candidate']} ({r['hook_beat']} → {r['cliffhanger_beat']}): "
            + "; ".join(r["errors"]) for r in rejected] or ["- none"]
    resolved = [n for n, c in cands.items() if not c.ends_on_cliffhanger]
    out += ["", f"**Episodes the model says do not end on a cliffhanger ({len(resolved)}):** "
            + (", ".join(map(str, resolved)) or "none")]
    if failed:
        out += ["", f"**Episodes with no candidates (LLM failure) ({len(failed)}):**"]
        out += [f"- ep {n}: {_cell(e, 200)}" for n, e in failed.items()]

    if len(ranked) > top:
        out += ["", f"## Promos {top + 1}–{len(ranked)}", "", "| # | Ep | Segment | Len | Hook | Score |", "|---|---|---|---|---|---|"]
        for r in ranked[top:]:
            seg = r["segments"][0]
            out.append(f"| {r['rank']} | {r['episode']} | {seg['start']}–{seg['end']} | {r['duration_s']:.0f}s "
                       f"| {_cell(r['hook'])} | {r['score']:.2f} |")
    return "\n".join(out) + "\n"
