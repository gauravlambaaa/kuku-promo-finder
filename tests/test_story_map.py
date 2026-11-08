"""Story-map tests, offline. Uses the synthetic show from test_parser."""
import json
import tempfile
import unittest
from pathlib import Path

from promo_finder.llm import LLMClient, LLMConfig
from promo_finder.parser import normalise_speakers, parse_show
from promo_finder.story_map import (
    Character,
    StoryMap,
    aliases_from,
    build_prompt,
    build_story_map,
    canonical_name,
    check_story_map,
    describe_aliases,
    ep_ranges,
    label_episodes,
)
from tests.test_llm import FakeTransport, ok
from tests.test_parser import EP1, EP2

# Episode 3 reuses the descriptive label "THE STRANGER" for a different person.
EP3 = """EPISODE NUMBER: Episode 3
EPISODE LOGLINE: Lord Ashby returns in disguise.
CONTINUITY NOTES: Follows episode 2.
--- SCRIPT START ---
SCENE 01
Scene heading: EXT. ROAD – NIGHT
Scene start: 00:00.00
Scene end: 00:05.00
**CAMERA:**
Start: (00:00.00) | End: (00:02.00)
Close-up on Lord Ashby, hooded.
**THE STRANGER (Low):**
Start: (00:00.50) | End: (00:02.00)
"Nobody recognises me."
--- SCRIPT END ---
"""

GOOD = {
    "premise": "A falsely accused baker fights a cruel noblewoman.",
    "characters": [
        {"name": "Marigold", "speaker_labels": ["MARIGOLD"], "role": "villain", "importance": "lead"},
        {"name": "Tom", "speaker_labels": ["TOM", "THE STRANGER"], "role": "accused baker", "importance": "lead"},
    ],
    "threads": [
        {"id": "theft", "title": "The theft", "summary": "Tom is framed.", "episodes": [1, 2], "title_premise": True},
    ],
    "episodes": [
        {"episode": 1, "threads": ["theft"], "key_reveals": ["Tom is accused"]},
        {"episode": 2, "threads": ["theft"], "key_reveals": []},
    ],
}


def copy(d):
    return json.loads(json.dumps(d))


class TestStoryMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "show"
        self.dir.mkdir()
        (self.dir / "episode-001-Episode 9001.txt").write_text(EP1, encoding="utf-8")
        (self.dir / "episode-002-Episode 9002.txt").write_text(EP2, encoding="utf-8")
        self.show = parse_show(self.dir)
        self.label_eps = label_episodes(self.show)

    def tearDown(self):
        self.tmp.cleanup()

    def add_ep3(self):
        (self.dir / "episode-003-Episode 9003.txt").write_text(EP3, encoding="utf-8")
        self.show = parse_show(self.dir)
        self.label_eps = label_episodes(self.show)

    def test_ep_ranges(self):
        self.assertEqual(ep_ranges([5, 1, 2, 3, 7, 8]), "1-3, 5, 7-8")
        self.assertEqual(ep_ranges([4]), "4")

    def test_plain_string_labels_are_accepted(self):
        c = Character.model_validate({"name": "Tom", "speaker_labels": ["TOM"], "role": "x", "importance": "lead"})
        self.assertEqual((c.speaker_labels[0].label, c.speaker_labels[0].episodes), ("TOM", []))

    def test_prompt_has_episodes_speakers_and_shot_context(self):
        p = build_prompt(self.show, "The Baker's Revenge")
        self.assertIn("SHOW TITLE: The Baker's Revenge", p)
        self.assertIn("[2] LOGLINE: Marigold gloats.", p)
        self.assertIn("THE STRANGER | 1 lines | eps 2", p)
        self.assertIn("[shot before: Close-up on Tom.]", p)  # the clue that names the speaker

    def test_prompt_samples_first_and_last_episode(self):
        self.add_ep3()
        p = build_prompt(self.show)
        block = p.split("THE STRANGER | 2 lines | eps 2-3")[1].split("\n")[1:3]
        self.assertTrue(block[0].strip().startswith('ep2: "Fools."'))
        self.assertTrue(block[1].strip().startswith('ep3: "Nobody recognises me."'))
        self.assertIn("Close-up on Lord Ashby, hooded.", block[1])

    def test_check_passes_good_map(self):
        self.assertEqual(check_story_map(StoryMap.model_validate(GOOD), self.label_eps, [1, 2]), [])

    def test_check_catches_semantic_errors(self):
        bad = copy(GOOD)
        bad["characters"][0]["speaker_labels"] = ["MARIGOLD", "LADY MARIGOLDE", {"label": "TOM", "episodes": [2]}]
        bad["characters"][1]["speaker_labels"].append("MARIGOLD")  # double-assigned
        bad["threads"][0]["title_premise"] = False
        bad["episodes"] = [{"episode": 1, "threads": ["nope"], "key_reveals": []}]
        errs = "\n".join(check_story_map(StoryMap.model_validate(bad), self.label_eps, [1, 2]))
        self.assertIn("'LADY MARIGOLDE' is not in SPEAKERS", errs)
        self.assertIn("label 'TOM' does not speak in episodes [2]", errs)
        self.assertIn("'MARIGOLD' is given to both 'Marigold' and 'Tom'", errs)
        self.assertIn("no thread has title_premise", errs)
        self.assertIn("episodes missing an entry: 2", errs)
        self.assertIn("unknown thread ids ['nope']", errs)

    def test_canonical_name_is_grounded_in_labels(self):
        def char(name, labels):
            return Character.model_validate({"name": name, "speaker_labels": labels, "role": "x", "importance": "lead"})

        self.assertEqual(canonical_name(char("Duke Tom Baker", ["TOM"]), self.label_eps), "TOM")
        self.assertEqual(canonical_name(char("Lady Marigold", ["MARIGOLD"]), self.label_eps), "MARIGOLD")
        self.assertEqual(canonical_name(char("Lord Ashby", ["LORD ASHBY"]), self.label_eps), "LORD ASHBY")
        # no own label inside the name -> keep the model's name
        self.assertEqual(canonical_name(char("The Baker", ["TOM"]), self.label_eps), "THE BAKER")

    def test_aliases_drop_unknown_and_duplicate_labels(self):
        bad = copy(GOOD)
        bad["characters"][0]["speaker_labels"] += ["GHOST", "THE STRANGER"]
        a = aliases_from(StoryMap.model_validate(bad), self.label_eps)
        self.assertEqual(a, {"THE STRANGER": "MARIGOLD"})  # first owner wins, GHOST dropped, no-ops skipped

    def test_episode_scoped_aliases(self):
        self.add_ep3()
        sm = copy(GOOD)
        sm["characters"][1]["speaker_labels"] = ["TOM", {"label": "THE STRANGER", "episodes": [2]}]
        sm["characters"].append({"name": "Lord Ashby", "role": "x", "importance": "minor",
                                 "speaker_labels": ["LORD ASHBY", {"label": "THE STRANGER", "episodes": [3]}]})
        sm["episodes"].append({"episode": 3, "threads": ["theft"], "key_reveals": []})
        sm = StoryMap.model_validate(sm)
        self.assertEqual(check_story_map(sm, self.label_eps, [1, 2, 3]), [])
        aliases = aliases_from(sm, self.label_eps)
        self.assertEqual(aliases, {"THE STRANGER@2": "TOM", "THE STRANGER@3": "LORD ASHBY"})
        self.assertEqual(describe_aliases(aliases),
                         ["THE STRANGER (eps 2) -> TOM", "THE STRANGER (eps 3) -> LORD ASHBY"])
        normalise_speakers(self.show, aliases)
        self.assertEqual(self.show.beat("E002-B005").speaker, "TOM")
        self.assertEqual(self.show.beat("E003-B002").speaker, "LORD ASHBY")
        self.assertNotIn("THE STRANGER", self.show.speakers)

    def test_end_to_end_with_fake_llm(self):
        cfg = LLMConfig(provider="fake", model="m", base_url="https://x/v1", api_key_env=None,
                        cache_dir=str(Path(self.tmp.name) / "cache"))
        bad = copy(GOOD)
        bad["episodes"] = bad["episodes"][:1]
        t = FakeTransport(ok(json.dumps(bad)), ok(json.dumps(GOOD)))
        client = LLMClient(cfg, transport=t)
        sm, aliases = build_story_map(self.show, client, "The Baker's Revenge")
        self.assertEqual(len(t.requests), 2)  # first reply missed episode 2 -> one fix round
        self.assertIn("episodes missing an entry: 2", t.requests[1]["body"]["messages"][-1]["content"])
        self.assertEqual(aliases, {"THE STRANGER": "TOM"})
        normalise_speakers(self.show, aliases)
        self.assertEqual(self.show.beat("E002-B005").speaker, "TOM")
        self.assertEqual(self.show.speakers["TOM"]["episodes"], [1, 2])


if __name__ == "__main__":
    unittest.main()
