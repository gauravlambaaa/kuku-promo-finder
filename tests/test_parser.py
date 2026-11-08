"""Parser tests. Fixtures are synthetic (the real input pack is confidential and gitignored).

Run:  python -m unittest discover tests      (or: pytest)
"""
import tempfile
import unittest
from pathlib import Path

from promo_finder.parser import (
    build_title_map,
    fmt_ts,
    format_beat,
    parse_episode,
    parse_header,
    parse_show,
    parse_ts,
)

DATA = Path(__file__).resolve().parent.parent / "data"

EP1 = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A baker is accused of theft at the village fair.
CONTINUITY NOTES: First episode.

--- SCRIPT START ---

SCENE 01
Scene heading: EXT. VILLAGE FAIR – DAY
Scene start: 00:00.00
Scene end: 00:20.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Wide shot of the fair.

**LADY MARIGOLD (Cold, Commanding):**
Start: (00:01.00) | End: (00:02.50)
"Stop that thief!"

**TOM (O.S. / Low, Threatening):**
Start: (00:04.00) | End: (00:06.00)
"I took nothing..."

**CAMERA:**
Start: (00:03.00) | End: (00:06.00)
Close-up on Tom.

**TOM (Continuing):**
Start: (00:06.10) | End: (00:07.00)
"...and you know it."

SCENE 02
SCENE HEADING: INT. JAIL – NIGHT
SCENE START: 00:20.00
SCENE END: 00:30.00

**ACTION:**
Start: (00:20.00) | End: (00:22.00)
Tom is thrown into a cell.

**ACTION:**
Start: (00:23.00)
Missing its end time.

**SOUND FX:**
Start: (00:28.00) | End: (00:29.50)
Ominous drone swells to a cliffhanger peak.

--- SCRIPT END ---
"""

EP2 = """EPISODE NUMBER: Episode 2
EPISODE LOGLINE: Marigold gloats.
CONTINUITY NOTES: Follows episode 1.
--- SCRIPT START ---
SCENE 01
Scene heading: INT. MANOR – DAY
Scene start: 00:00.00
Scene end: 00:10.00
**MARIGOLD (Smug):**
Start: (00:00.50) | End: (00:02.00)
"He will hang."
**LORD ASHBY (Stern):**
Start: (00:02.00) | End: (00:04.00)
"Quiet."
**LADY ASHBY (Sharp):**
Start: (00:04.00) | End: (00:06.00)
"Let her speak."
**ASHBY (Tired):**
Start: (00:06.00) | End: (00:08.00)
"Enough, both of you."
**THE STRANGER (Contemptuous):**
Start: (00:08.00) | End: (00:09.00)
"Fools."
--- SCRIPT END ---
"""


class TestHelpers(unittest.TestCase):
    def test_parse_ts(self):
        self.assertEqual(parse_ts("00:00.00"), 0.0)
        self.assertEqual(parse_ts("01:23.45"), 83.45)
        self.assertEqual(parse_ts("02:18"), 138.0)
        self.assertEqual(parse_ts("1:00:01.5"), 3601.5)

    def test_fmt_ts_roundtrip(self):
        for s in ["00:00.00", "00:04.80", "01:23.45", "02:59.99", "12:00.00"]:
            self.assertEqual(fmt_ts(parse_ts(s)), s)

    def test_header_plain(self):
        self.assertEqual(parse_header("MATRIARCH (Cold, Commanding):"),
                         ("MATRIARCH", None, ["Cold", "Commanding"], False))

    def test_header_delivery_variants(self):
        self.assertEqual(parse_header("MATRIARCH (O.S.) (Firm):"), ("MATRIARCH", "O.S.", ["Firm"], False))
        self.assertEqual(parse_header("GUARD (O.S. / Low, Threatening):"),
                         ("GUARD", "O.S.", ["Low", "Threatening"], False))
        self.assertEqual(parse_header("THE PRIEST (Off-Screen, Booming):"),
                         ("THE PRIEST", "O.S.", ["Booming"], False))
        self.assertEqual(parse_header("JULIANA (V.O.):"), ("JULIANA", "V.O.", [], False))

    def test_header_continuing(self):
        self.assertEqual(parse_header("JULIANA (Continuing):"), ("JULIANA", None, [], True))
        self.assertEqual(parse_header("X (Voice Continuing):"), ("X", None, [], True))
        self.assertEqual(parse_header("X (O.S. / Continues):"), ("X", "O.S.", [], True))
        self.assertEqual(parse_header("CLARISSA (O.S.) (Continuing taunt):"),
                         ("CLARISSA", "O.S.", ["taunt"], True))

    def test_header_cont_prefix_words_are_emotions(self):
        # 'Contemptuous' / 'Contemplative' must not be read as 'Continuing'
        _, _, emo, cont = parse_header("X (Contemptuous, Contemplative):")
        self.assertEqual(emo, ["Contemptuous", "Contemplative"])
        self.assertFalse(cont)

    def test_title_map(self):
        names = {"JULIANA", "LADY JULIANA", "THE PRIEST", "PRIEST",
                 "LORD ASHBY", "LADY ASHBY", "ASHBY", "THE FORMER GROOM", "LORD"}
        m = build_title_map(names)
        self.assertEqual(m["LADY JULIANA"], "JULIANA")
        self.assertEqual(m["THE PRIEST"], "PRIEST")
        # two titles compete for ASHBY -> leave both alone
        self.assertNotIn("LORD ASHBY", m)
        self.assertNotIn("LADY ASHBY", m)
        # no bare 'FORMER GROOM' speaker -> unchanged
        self.assertNotIn("THE FORMER GROOM", m)


class TestEpisode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "episode-001-Episode 9001.txt").write_text(EP1, encoding="utf-8")
        (self.dir / "episode-002-Episode 9002.txt").write_text(EP2, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_meta_and_scenes(self):
        ep = parse_episode(self.dir / "episode-001-Episode 9001.txt")
        self.assertEqual(ep.episode, 1)
        self.assertEqual(ep.episode_id, "9001")
        self.assertIn("baker", ep.logline)
        self.assertEqual([s.scene for s in ep.scenes], [1, 2])
        # upper-case SCENE HEADING / START / END variant
        self.assertEqual(ep.scenes[1].heading, "INT. JAIL – NIGHT")
        self.assertEqual((ep.scenes[1].start, ep.scenes[1].end), (20.0, 30.0))
        self.assertEqual(ep.duration, 30.0)

    def test_beats_sorted_and_ids_sequential(self):
        ep = parse_episode(self.dir / "episode-001-Episode 9001.txt")
        starts = [b.start for b in ep.beats]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual([b.beat_id for b in ep.beats],
                         [f"E001-B{i:03d}" for i in range(1, len(ep.beats) + 1)])
        # the CAMERA written after TOM's line (03.00) sorts before it (04.00)
        types = [(b.type, b.start) for b in ep.beats[:4]]
        self.assertEqual(types, [("camera", 0.0), ("dialogue", 1.0), ("camera", 3.0), ("dialogue", 4.0)])

    def test_dialogue_fields(self):
        ep = parse_episode(self.dir / "episode-001-Episode 9001.txt")
        tom = [b for b in ep.beats if b.speaker_raw == "TOM"]
        self.assertEqual(tom[0].text, "I took nothing...")  # outer quotes stripped
        self.assertEqual(tom[0].delivery, "O.S.")
        self.assertEqual(tom[0].emotion, ["Low", "Threatening"])
        self.assertFalse(tom[0].continues)
        self.assertTrue(tom[1].continues)
        self.assertEqual(tom[0].scene, 1)

    def test_non_dialogue_types(self):
        ep = parse_episode(self.dir / "episode-001-Episode 9001.txt")
        sfx = [b for b in ep.beats if b.type == "sfx"]
        self.assertEqual(len(sfx), 1)
        self.assertIsNone(sfx[0].speaker)
        self.assertEqual(sfx[0].scene, 2)
        self.assertIn("cliffhanger", sfx[0].text)

    def test_bad_block_is_warning_not_crash(self):
        ep = parse_episode(self.dir / "episode-001-Episode 9001.txt")
        self.assertTrue(any("no Start/End" in w for w in ep.warnings))
        self.assertFalse(any("Missing its end time" in b.text for b in ep.beats))

    def test_show_normalises_and_looks_up(self):
        show = parse_show(self.dir)
        self.assertEqual([e.episode for e in show.episodes], [1, 2])
        # LADY MARIGOLD (ep 1) folds into bare MARIGOLD (ep 2)
        self.assertEqual(show.speakers["MARIGOLD"]["variants"], ["LADY MARIGOLD", "MARIGOLD"])
        self.assertEqual(show.speakers["MARIGOLD"]["episodes"], [1, 2])
        self.assertIn("LORD ASHBY", show.speakers)
        self.assertIn("LADY ASHBY", show.speakers)
        b = show.beat("E002-B001")
        self.assertEqual((b.speaker, b.start, b.end), ("MARIGOLD", 0.5, 2.0))
        self.assertTrue(format_beat(b).startswith('E002-B001 [00:00.50-00:02.00] MARIGOLD (Smug): "He will'))

    def test_alias_map(self):
        show = parse_show(self.dir, aliases={"The Stranger": "Tom"})
        self.assertEqual(show.beat("E002-B005").speaker, "TOM")
        self.assertIn("THE STRANGER", show.speakers["TOM"]["variants"])


@unittest.skipUnless(DATA.is_dir() and any(DATA.glob("*.txt")), "input pack not present")
class TestRealPack(unittest.TestCase):
    def test_full_pack_parses_cleanly(self):
        show = parse_show(DATA)
        self.assertEqual(len(show.episodes), 50)
        self.assertEqual([e.episode for e in show.episodes], list(range(1, 51)))
        # every **HEADER** block in the files became exactly one beat
        n_headers = sum(
            sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.startswith("**"))
            for f in DATA.glob("*.txt")
        )
        self.assertEqual(sum(len(e.beats) for e in show.episodes), n_headers)
        self.assertEqual([w for e in show.episodes for w in e.warnings], [])
        self.assertEqual(show.warnings, [])
        self.assertIn("LADY JULIANA", show.speakers["JULIANA"]["variants"])
        for e in show.episodes:
            for b in e.beats:
                self.assertLessEqual(b.start, b.end, b.beat_id)
                self.assertTrue(b.text, b.beat_id)


if __name__ == "__main__":
    unittest.main()
