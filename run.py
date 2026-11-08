"""Promo finder pipeline: python run.py --input <episodes folder> --out <output folder>

Steps:
  1. parse        screenplay .txt -> timed beats (code)
  2. story map    one LLM call: characters + speaker aliases, story threads, reveals per episode
  3. candidates   one LLM call per episode (parallel): hooks, cliffhangers, best clips, rubric scores
  5. validate     cut every clip on beat edges; 20-60s, hook within 3s, no mid-line cuts (code)
  7. rank+report  weighted score, de-duplicated, flagged weak picks (code)
  (4. cross-episode stitching and 6. standalone judge are not built yet)

Outputs (in --out; they contain script text, so keep them out of public repos):
  promos.md        the ranked report a human editor reads
  promos.json      ranked promos + dropped + rejected, machine-readable
  candidates.json  raw step-3 output per episode
  story_map.json   story map + alias map
  beats.json       parsed show, speakers normalised
  usage.jsonl      one line per LLM call: tokens, latency, cache hit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from promo_finder.candidates import find_all_candidates
from promo_finder.llm import LLMClient, LLMError, load_config
from promo_finder.parser import normalise_speakers, parse_show
from promo_finder.rank import build_promos, rank, render_markdown
from promo_finder.story_map import build_prompt, build_story_map, describe_aliases, ep_ranges


def parse_episode_list(s: str | None) -> list[int] | None:
    """'1-3,7' -> [1, 2, 3, 7]"""
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find promo clips in a folder of microdrama screenplays.")
    ap.add_argument("--input", required=True, help="folder of episode .txt files")
    ap.add_argument("--out", default="outputs/raw", help="output folder (default: outputs/raw, gitignored)")
    ap.add_argument("--title", help="show title, if known (helps rank the title premise)")
    ap.add_argument("--episodes", help="only run step 3 on these episodes, e.g. '1-3,7' (default: all)")
    ap.add_argument("--top", type=int, default=20, help="promos shown in full in promos.md")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="parse and print the story-map prompt size; no LLM calls")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # 1. parse
    show = parse_show(args.input)
    n_beats = sum(len(e.beats) for e in show.episodes)
    n_warn = sum(len(e.warnings) for e in show.episodes) + len(show.warnings)
    print(f"[1] parsed {len(show.episodes)} episodes, {n_beats} beats, {len(show.speakers)} speakers, {n_warn} warnings")

    if args.dry_run:
        prompt = build_prompt(show, args.title)
        print(f"[2] story-map prompt: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens). Not sent (--dry-run).")
        (out / "story_map_prompt.txt").write_text(prompt, encoding="utf-8")
        return 0

    cfg = load_config(args.config)
    usage_log = out / "usage.jsonl"
    usage_log.unlink(missing_ok=True)
    client = LLMClient(cfg, usage_log=usage_log)

    try:
        # 2. story map
        print(f"[2] story map with {client.describe()} ...", flush=True)
        sm, aliases = build_story_map(show, client, args.title)
        normalise_speakers(show, aliases)
        print(f"    {len(sm.characters)} characters, {len(sm.threads)} threads, {len(aliases)} alias rules, "
              f"{len(show.speakers)} speakers after merging")
        for line in describe_aliases(aliases):
            print(f"      {line}")
        write_json(out / "story_map.json", {
            "model": client.model_for("story_map"), "provider": cfg.provider, "title": args.title,
            "aliases": aliases, "story_map": sm.model_dump()})
        write_json(out / "beats.json", show.to_dict())

        # 3. per-episode candidates
        episodes = parse_episode_list(args.episodes)
        n = len(episodes) if episodes else len(show.episodes)
        print(f"[3] candidates for {n} episodes ({cfg.concurrency} parallel, {cfg.rpm} rpm) ...", flush=True)
        cands, failed, models = find_all_candidates(show, sm, client, args.title, episodes)
    except LLMError as e:
        print(f"\nLLM error: {e}", file=sys.stderr)
        return 1
    write_json(out / "candidates.json", {
        "models": models,
        "episodes": {n: c.model_dump() for n, c in cands.items()}, "failed": failed})
    by_model: dict[str, list[int]] = {}
    for n, m in models.items():
        by_model.setdefault(m, []).append(n)
    model_note = "; ".join(f"{cfg.provider}:{m} (eps {ep_ranges(e)})" for m, e in by_model.items()) or "none"

    # 5. validate + 7. rank
    accepted, rejected = build_promos(show, cands, models)
    ranked, dropped, shifts = rank(accepted)
    print(f"[5] {len(accepted)} clips passed validation, {len(rejected)} rejected")
    print(f"[7] {len(ranked)} promos ranked, {len(dropped)} dropped (overlap / per-episode cap)")
    write_json(out / "promos.json", {
        "title": args.title, "models": models, "score_calibration": shifts,
        "ranked": ranked, "dropped": dropped, "rejected": rejected, "failed_episodes": failed})
    (out / "promos.md").write_text(render_markdown(
        title=args.title, model=model_note, ranked=ranked, dropped=dropped, rejected=rejected,
        cands=cands, failed=failed, top=args.top, shifts=shifts), encoding="utf-8")

    for w in client.warnings:
        print(f"    warning: {w}")
    print("\nLLM usage:\n" + client.usage_summary())
    print(f"\nwrote {out / 'promos.md'} and promos.json, candidates.json, story_map.json, beats.json, usage.jsonl "
          f"in {time.monotonic() - t0:.0f}s")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
