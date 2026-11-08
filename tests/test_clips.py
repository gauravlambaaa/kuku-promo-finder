"""Clip cutting/validation (step 5), candidate checks (step 3) and ranking (step 7). Offline."""
import json
import tempfile
import unittest
from pathlib import Path

from promo_finder.candidates import EpisodeCandidates, check_candidates, find_candidates
from promo_finder.clips import cut_clip, normalise_id
from promo_finder.llm import LLMClient, LLMConfig
from promo_finder.parser import Beat, Episode, Show
from promo_finder.rank import build_promos, rank
from promo_finder.story_map import StoryMap
from tests.test_llm import FakeTransport, ok


def make_episode(spec, n=1) -> Episode:
    """spec: list of (type, start, end, speaker, text, continues)."""
    beats = []
    for i, (t, s, e, spk, text, cont) in enumerate(spec, 1):
        beats.append(Beat(beat_id=f"E{n:03d}-B{i:03d}", episode=n, scene=1, start=s, end=e, type=t,
                          speaker=spk, speaker_raw=spk, delivery=None, emotion=[], continues=cont, text=text, line=i))
    return Episode(episode=n, episode_id=None, source_file="x.txt", logline="A test.", continuity="",
                   duration=max(b.end for b in beats), scenes=[], beats=beats)


EP = make_episode([
    ("camera", 0.0, 4.0, None, "Wide shot of the hall.", False),            # B001
    ("dialogue", 1.0, 5.0, "ANNA", "You stole my ring!", False),              # B002
    ("action", 4.0, 6.0, None, "Anna slaps him.", False),                     # B003
    ("dialogue", 6.5, 9.0, "TOM", "I never...", False),                       # B004
    ("dialogue", 9.2, 11.0, "TOM", "...touched it.", True),                   # B005  continuation
    ("dialogue", 24.0, 27.0, "ANNA", "Then who did?", False),                 # B006
    ("dialogue", 26.5, 29.0, "TOM", "Your sister.", False),                   # B007  overlaps B006
    ("camera", 29.0, 30.0, None, "Cut to black.", False),                     # B008
    ("dialogue", 70.0, 72.0, "ANNA", "Much later.", False),                   # B009
])


class TestCutClip(unittest.TestCase):
    def test_clean_clip(self):
        cut = cut_clip(EP, "E001-B002", "E001-B008")
        self.assertTrue(cut.ok, cut.errors)
        self.assertEqual((cut.segment.start, cut.segment.end, cut.segment.duration), (1.0, 30.0, 29.0))
        self.assertEqual(cut.hook_offset, 0.0)
        self.assertEqual(cut.flags, [])

    def test_end_extends_over_overlapping_line(self):
        cut = cut_clip(EP, "E001-B002", "E001-B006")  # B007 starts at 26.5, before B006 ends at 27.0
        self.assertTrue(cut.ok, cut.errors)
        self.assertEqual((cut.segment.end, cut.segment.end_beat), (29.0, "E001-B007"))
        self.assertIn("isn't cut mid-line", cut.flags[0])

    def test_end_extends_to_finish_sentence(self):
        cut = cut_clip(EP, "E001-B001", "E001-B004")  # "I never..." continues in B005
        self.assertEqual(cut.segment.end, 11.0)
        self.assertTrue(any("finishes the sentence" in f for f in cut.flags))
        self.assertTrue(any("at least 20s" in e for e in cut.errors))  # and it is far too short

    def test_start_backs_up_out_of_a_line(self):
        cut = cut_clip(EP, "E001-B003", "E001-B008")  # B002 (1.0-5.0) is mid-line at 4.0
        self.assertTrue(cut.ok, cut.errors)
        self.assertEqual((cut.segment.start, cut.hook_offset), (1.0, 3.0))
        self.assertIn("start moved back", cut.flags[0])

    def test_hard_errors(self):
        self.assertIn("at most 60s", cut_clip(EP, "E001-B002", "E001-B009").errors[0])
        self.assertIn("must come after", cut_clip(EP, "E001-B008", "E001-B002").errors[0])
        self.assertIn("not in episode", cut_clip(EP, "E001-B002", "E001-B099").errors[0])

    def test_hook_opening_mid_sentence_is_flagged(self):
        cut = cut_clip(EP, "E001-B005", "E001-B008")
        self.assertTrue(any("mid-sentence" in f for f in cut.flags))

    def test_normalise_id(self):
        self.assertEqual(normalise_id("B023", 7), "E007-B023")
        self.assertEqual(normalise_id("e7-b23", 7), "E007-B023")
        self.assertEqual(normalise_id("E007-B023", 7), "E007-B023")


def clip(hook_id, cliff_id, **scores):
    s = {"hook": 7, "cliffhanger": 7, "standalone": 7, "intensity": 7, "premise": 5, "pacing": 7, **scores}
    return {"hook_beat": hook_id, "cliffhanger_beat": cliff_id, "hook": "h", "cliffhanger": "c", "why": "w", "scores": s}


def candidates(*clips):
    return {"ends_on_cliffhanger": True, "hooks": [], "cliffhangers": [], "clips": list(clips)}


class TestCandidatesAndRank(unittest.TestCase):
    def setUp(self):
        self.show = Show(source_dir="x", episodes=[EP], speakers={})
        self.sm = StoryMap.model_validate({"premise": "p", "characters": [], "threads": [], "episodes": []})

    def test_check_reports_bad_clips(self):
        c = EpisodeCandidates.model_validate(candidates(clip("E001-B002", "E001-B009"), clip("B002", "B008")))
        errs = check_candidates(c, EP)
        self.assertEqual(len(errs), 1)
        self.assertTrue(errs[0].startswith("clip 1:") and "at most 60s" in errs[0])

    def test_fix_loop_gets_validator_errors(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = LLMConfig(provider="fake", model="m", base_url="https://x/v1", api_key_env=None, cache_dir=tmp.name)
        t = FakeTransport(ok(json.dumps(candidates(clip("E001-B002", "E001-B009")))),
                          ok(json.dumps(candidates(clip("E001-B002", "E001-B008")))))
        result = find_candidates(self.show, self.sm, EP, LLMClient(cfg, transport=t), "Title")
        self.assertEqual(result.clips[0].cliffhanger_beat, "E001-B008")
        self.assertIn("at most 60s", t.requests[1]["body"]["messages"][-1]["content"])
        self.assertIn('E001-B002 [00:01.00-00:05.00] ANNA: "You stole my ring!"', t.requests[0]["body"]["messages"][1]["content"])

    def test_rank_validates_scores_and_dedups(self):
        c = EpisodeCandidates.model_validate(candidates(
            clip("E001-B002", "E001-B008", hook=9, cliffhanger=9),    # best
            clip("E001-B001", "E001-B008"),                           # overlaps the best -> dropped
            clip("E001-B002", "E001-B009"),                           # too long -> rejected
        ))
        accepted, rejected = build_promos(self.show, {1: c})
        self.assertEqual(len(accepted), 2)
        self.assertEqual(rejected[0]["candidate"], 3)
        ranked, dropped, shifts = rank(accepted)
        self.assertEqual(shifts, {})  # one model -> no calibration
        self.assertEqual([r["candidate"] for r in ranked], [1])
        self.assertEqual(ranked[0]["id"], "P01")
        self.assertEqual(ranked[0]["segments"][0]["start"], "00:01.00")
        self.assertIn("strongest on hook 9, cliffhanger 9", ranked[0]["rank_reason"])
        self.assertEqual(dropped[0]["dropped"], "overlaps a higher-ranked clip")


class TestCalibration(unittest.TestCase):
    def test_shifts_each_model_to_the_common_mean(self):
        from promo_finder.rank import calibrate
        recs = [{"model": "lite", "score": 9.0} for _ in range(5)] + [{"model": "flash", "score": 8.0} for _ in range(5)]
        recs.append({"model": "tiny", "score": 7.0})  # too few clips to calibrate
        shifts = calibrate(recs)
        self.assertEqual(set(shifts), {"lite", "flash"})
        self.assertEqual(recs[0]["score"], round(9.0 + shifts["lite"], 2))
        self.assertAlmostEqual(recs[0]["score"] - recs[5]["score"], 0.0, places=2)  # the 1.0 gap was model bias
        self.assertEqual((recs[-1]["score"], recs[-1]["score_raw"]), (7.0, 7.0))


if __name__ == "__main__":
    unittest.main()
