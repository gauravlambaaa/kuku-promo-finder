"""LLM adapter tests, fully offline: a fake transport stands in for the HTTP call."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import BaseModel

from promo_finder.llm import LLMClient, LLMConfig, LLMError, extract_json, load_config


class Pick(BaseModel):
    beat_id: str
    score: int


def ok(content: str, prompt_tokens=100, completion_tokens=20) -> tuple[int, str, dict]:
    body = {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
    return 200, json.dumps(body), {}


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": headers, "body": json.loads(json.dumps(body))})
        return self.responses.pop(0)


class ClientTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = LLMConfig(provider="fake", model="m-1", base_url="https://llm.example/v1",
                             api_key_env="FAKE_KEY", rpm=None, cache_dir=self.tmp.name)
        os.environ["FAKE_KEY"] = "sk-test"

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("FAKE_KEY", None)

    def client(self, *responses, **cfg_overrides):
        for k, v in cfg_overrides.items():
            setattr(self.cfg, k, v)
        t = FakeTransport(*responses)
        c = LLMClient(self.cfg, usage_log=Path(self.tmp.name) / "usage.jsonl", transport=t)
        c._sleep = lambda s: None
        return c, t

    def ask(self, c, prompt="pick one", **kw):
        return c.complete_json(step="test", system="sys", prompt=prompt, schema=Pick, **kw)


class TestExtractJson(unittest.TestCase):
    def test_plain_fenced_and_wrapped(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here it is: {"a": {"b": 2}} Hope that helps.'), {"a": {"b": 2}})

    def test_rejects_non_objects(self):
        with self.assertRaises(ValueError):
            extract_json("no json here")
        with self.assertRaises(ValueError):
            extract_json("[1, 2]")


class TestCompleteJson(ClientTestCase):
    def test_happy_path_request_shape(self):
        c, t = self.client(ok('{"beat_id": "E001-B003", "score": 9}'))
        self.assertEqual(self.ask(c), Pick(beat_id="E001-B003", score=9))
        req = t.requests[0]
        self.assertEqual(req["url"], "https://llm.example/v1/chat/completions")
        self.assertEqual(req["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(req["body"]["model"], "m-1")
        self.assertEqual(req["body"]["response_format"], {"type": "json_object"})
        self.assertIn('"beat_id"', req["body"]["messages"][0]["content"])  # schema is in the prompt

    def test_self_correction_on_bad_json_then_schema_error(self):
        c, t = self.client(ok("oops, not json"), ok('{"beat_id": "E001-B003"}'), ok('{"beat_id": "E001-B003", "score": 7}'))
        self.assertEqual(self.ask(c).score, 7)
        self.assertEqual(len(t.requests), 3)
        last = t.requests[2]["body"]["messages"]
        self.assertEqual([m["role"] for m in last], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertIn("no JSON object", last[3]["content"])
        self.assertIn("score: Field required", last[5]["content"])

    def test_semantic_check_feeds_back(self):
        c, t = self.client(ok('{"beat_id": "E009-B999", "score": 7}'), ok('{"beat_id": "E001-B003", "score": 7}'))
        check = lambda p: [] if p.beat_id == "E001-B003" else [f"unknown beat id {p.beat_id}"]
        self.assertEqual(self.ask(c, check=check).beat_id, "E001-B003")
        self.assertIn("unknown beat id E009-B999", t.requests[1]["body"]["messages"][-1]["content"])

    def test_gives_up_after_fix_attempts(self):
        c, _ = self.client(ok("bad"), ok("bad"), ok("bad"), fix_attempts=2)
        with self.assertRaises(LLMError):
            self.ask(c)

    def test_accept_mode_returns_last_valid_with_warnings(self):
        c, _ = self.client(*[ok('{"beat_id": "X", "score": 1}')] * 3, fix_attempts=2)
        result = self.ask(c, check=lambda p: ["still wrong"], on_check_fail="accept")
        self.assertEqual(result.beat_id, "X")
        self.assertEqual(c.warnings, ["test: still wrong"])

    def test_cache_makes_reruns_free(self):
        c1, t1 = self.client(ok('{"beat_id": "A", "score": 1}'))
        self.ask(c1)
        c2, t2 = self.client()  # no responses queued: any HTTP call would fail
        self.assertEqual(self.ask(c2).beat_id, "A")
        self.assertEqual(t2.requests, [])
        self.assertEqual(c2.totals["test"]["cached"], 1)
        self.assertEqual(c2.totals["test"]["prompt_tokens"], 100)  # cached calls still report tokens

    def test_cache_key_depends_on_model(self):
        c1, _ = self.client(ok('{"beat_id": "A", "score": 1}'))
        self.ask(c1)
        c2, t2 = self.client(ok('{"beat_id": "B", "score": 2}'), model="m-2")
        self.assertEqual(self.ask(c2).beat_id, "B")
        self.assertEqual(len(t2.requests), 1)

    def test_usage_log(self):
        c, _ = self.client(ok('{"beat_id": "A", "score": 1}', prompt_tokens=1234, completion_tokens=56))
        self.ask(c)
        rec = json.loads((Path(self.tmp.name) / "usage.jsonl").read_text().splitlines()[0])
        self.assertEqual((rec["step"], rec["prompt_tokens"], rec["completion_tokens"], rec["cached"]),
                         ("test", 1234, 56, False))


class TestHttp(ClientTestCase):
    def test_retries_429_then_succeeds(self):
        c, t = self.client(
            (429, '{"error": {"message": "Resource exhausted. Please retry in 2.5s."}}', {}),
            (503, "overloaded", {"Retry-After": "1"}),
            ok('{"beat_id": "A", "score": 1}'),
        )
        sleeps = []
        c._sleep = sleeps.append
        self.assertEqual(self.ask(c).beat_id, "A")
        self.assertEqual(sleeps, [3.5, 2.0])

    def test_daily_quota_fails_fast(self):
        c, t = self.client((429, '{"error": {"message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}', {}))
        with self.assertRaisesRegex(LLMError, "daily free-tier quota"):
            self.ask(c)
        self.assertEqual(len(t.requests), 1)

    def test_falls_back_to_next_model_when_daily_quota_is_gone(self):
        per_day = (429, '{"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}', {})
        c, t = self.client(per_day, ok('{"beat_id": "A", "score": 1}'), ok('{"beat_id": "B", "score": 2}'),
                           fallback_models=["m-2"])
        self.assertEqual(self.ask(c).beat_id, "A")
        self.assertEqual([r["body"]["model"] for r in t.requests], ["m-1", "m-2"])
        self.assertEqual(c.last_model(), "m-2")
        self.ask(c, prompt="second")  # m-1 is remembered as exhausted
        self.assertEqual(t.requests[2]["body"]["model"], "m-2")
        self.assertIn("m-1: daily free-tier quota used up; falling back to m-2", c.warnings)

    def test_cache_from_primary_model_is_reused_before_fallback(self):
        c1, _ = self.client(ok('{"beat_id": "A", "score": 1}'))
        self.ask(c1)  # cached under m-1
        c2, t2 = self.client(fallback_models=["m-2"])
        self.assertEqual(self.ask(c2).beat_id, "A")
        self.assertEqual(t2.requests, [])

    def test_auth_error_not_retried(self):
        c, t = self.client((401, '{"error": "invalid api key"}', {}))
        with self.assertRaisesRegex(LLMError, "HTTP 401"):
            self.ask(c)
        self.assertEqual(len(t.requests), 1)

    def test_falls_back_when_response_format_unsupported(self):
        c, t = self.client(
            (400, '{"error": {"message": "Invalid value for response_format"}}', {}),
            ok('{"beat_id": "A", "score": 1}'),
            ok('{"beat_id": "B", "score": 2}'),
        )
        self.assertEqual(self.ask(c).beat_id, "A")
        self.assertNotIn("response_format", t.requests[1]["body"])
        self.ask(c, prompt="second call")  # remembered for the rest of the run
        self.assertNotIn("response_format", t.requests[2]["body"])

    def test_missing_api_key(self):
        os.environ.pop("FAKE_KEY")
        c, _ = self.client()
        with self.assertRaisesRegex(LLMError, "FAKE_KEY is not set"):
            self.ask(c)


class TestConfig(unittest.TestCase):
    def write(self, text):
        d = tempfile.mkdtemp()
        p = Path(d) / "config.yaml"
        p.write_text(text, encoding="utf-8")
        return p

    CFG = """
llm:
  provider: gemini
  model: gemini-x
  step_models: {judge: gemini-lite}
  rpm: 5
providers:
  gemini: {base_url: "https://g.example/openai/", api_key_env: GEMINI_API_KEY}
  groq: {base_url: "https://groq.example/v1", api_key_env: GROQ_API_KEY}
"""

    def test_load(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("PROMO_PROVIDER", "PROMO_MODEL", "PROMO_BASE_URL", "PROMO_API_KEY_ENV"):
                os.environ.pop(k, None)
            cfg = load_config(self.write(self.CFG))
        self.assertEqual((cfg.provider, cfg.model, cfg.base_url, cfg.api_key_env, cfg.rpm),
                         ("gemini", "gemini-x", "https://g.example/openai/", "GEMINI_API_KEY", 5))
        self.assertEqual(cfg.step_models, {"judge": "gemini-lite"})

    def test_env_swaps_provider_and_drops_step_models(self):
        with mock.patch.dict(os.environ, {"PROMO_PROVIDER": "groq", "PROMO_MODEL": "llama-x"}):
            cfg = load_config(self.write(self.CFG))
        self.assertEqual((cfg.provider, cfg.model, cfg.base_url, cfg.api_key_env),
                         ("groq", "llama-x", "https://groq.example/v1", "GROQ_API_KEY"))
        self.assertEqual(cfg.step_models, {})

    def test_unknown_provider_errors(self):
        with mock.patch.dict(os.environ, {"PROMO_PROVIDER": "nope"}):
            with self.assertRaisesRegex(LLMError, "no base_url"):
                load_config(self.write(self.CFG))

    def test_repo_config_loads(self):
        cfg = load_config()
        self.assertTrue(cfg.base_url.startswith("http"))


if __name__ == "__main__":
    unittest.main()
