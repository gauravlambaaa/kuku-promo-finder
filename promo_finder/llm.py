"""The single adapter for every LLM call. Provider and model come from config.yaml / env.

Every supported provider speaks the OpenAI chat-completions protocol (Gemini through
its OpenAI-compatible endpoint, Groq, Ollama, OpenAI, OpenRouter, most in-house
gateways), so swapping models is a config change, not a code change. Plain stdlib
HTTP, no vendor SDK.

`LLMClient.complete_json()` turns a prompt into a validated pydantic object:
  - the JSON schema is embedded in the prompt, so it works even without native JSON mode
  - self-correction loop: schema errors and semantic `check` errors go back to the model
  - on-disk cache keyed on the full request: re-runs are free and reproducible
  - client-side throttle + backoff on 429/5xx, for free tiers
  - per-call token usage appended to a JSONL log, totals per step in `client.totals`
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "kuku-promo-finder/0.1"

JSON_INSTRUCTIONS = (
    "Respond with a single JSON object and nothing else: no prose, no markdown fences. "
    "It must validate against this JSON Schema:\n{schema}"
)
FIX_PROMPT = (
    "Your previous reply was rejected:\n{errors}\n\n"
    "Return the complete corrected JSON object only."
)


class LLMError(RuntimeError):
    pass


class _ResponseFormatUnsupported(Exception):
    pass


class DailyQuotaExhausted(LLMError):
    pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str | None = None
    step_models: dict[str, str] = field(default_factory=dict)
    fallback_models: list[str] = field(default_factory=list)  # tried in order when a model's daily quota runs out
    temperature: float = 0.2
    json_mode: Literal["json_object", "json_schema", "off"] = "json_object"
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    fix_attempts: int = 2
    http_retries: int = 5
    timeout_s: float = 300
    rpm: float | None = None
    concurrency: int = 4
    use_cache: bool = True
    cache_dir: str = ".cache/llm"


def load_dotenv(path: str | Path = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; never overrides variables already set."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path = ROOT / "config.yaml") -> LLMConfig:
    load_dotenv()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    llm = dict(raw.get("llm") or {})
    env_override = any(os.environ.get(k) for k in ("PROMO_PROVIDER", "PROMO_MODEL"))

    provider = os.environ.get("PROMO_PROVIDER") or llm.pop("provider", "gemini")
    llm.pop("provider", None)
    preset = (raw.get("providers") or {}).get(provider) or {}
    model = os.environ.get("PROMO_MODEL") or llm.pop("model", None)
    llm.pop("model", None)
    base_url = os.environ.get("PROMO_BASE_URL") or preset.get("base_url")
    api_key_env = os.environ.get("PROMO_API_KEY_ENV") or preset.get("api_key_env")
    if not model:
        raise LLMError("no model set (llm.model in config.yaml, or PROMO_MODEL)")
    if not base_url:
        raise LLMError(f"provider {provider!r} has no base_url (add it under providers: in config.yaml, or set PROMO_BASE_URL)")
    if env_override:
        llm["step_models"] = {}  # an env override means "this one model for everything"
        llm["fallback_models"] = []
    llm["step_models"] = llm.get("step_models") or {}
    llm["fallback_models"] = llm.get("fallback_models") or []
    known = set(LLMConfig.__dataclass_fields__)
    unknown = set(llm) - known
    if unknown:
        raise LLMError(f"unknown llm settings in config.yaml: {sorted(unknown)}")
    return LLMConfig(provider=provider, model=model, base_url=base_url, api_key_env=api_key_env, **llm)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating fences and stray prose."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("reply contains no JSON object")
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"reply is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ValueError("reply must be a JSON object, not a " + type(obj).__name__)
    return obj


def format_validation_error(e: ValidationError, limit: int = 30) -> list[str]:
    out = []
    for err in e.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        out.append(f"{loc}: {err['msg']}")
    return out


def _urllib_transport(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, str, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers or {})
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return 0, f"network error: {e}", {}


def _retry_delay(text: str, headers: dict, attempt: int) -> float:
    ra = {k.lower(): v for k, v in headers.items()}.get("retry-after")
    if ra:
        try:
            return float(ra) + 1
        except ValueError:
            pass
    m = re.search(r'retry(?:Delay"?\s*:\s*"|\s+in\s+)(\d+(?:\.\d+)?)s', text, re.I)
    if m:
        return float(m.group(1)) + 1
    return min(60.0, 2.0 * 2**attempt) + random.uniform(0, 1)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

Transport = Callable[[str, dict, dict, float], tuple[int, str, dict]]


class LLMClient:
    def __init__(self, cfg: LLMConfig, usage_log: str | Path | None = None, transport: Transport | None = None):
        self.cfg = cfg
        self.usage_log = Path(usage_log) if usage_log else None
        self.cache_dir = Path(cfg.cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = ROOT / self.cache_dir
        self.totals: dict[str, Counter] = defaultdict(Counter)
        self.warnings: list[str] = []
        self._transport = transport or _urllib_transport
        self._sleep = time.sleep
        self._lock = threading.Lock()  # complete_json may be called from several threads
        self._next_slot = 0.0
        self._response_format_ok = True
        self._exhausted: set[str] = set()  # models whose daily quota ran out this run
        self._local = threading.local()

    def model_for(self, step: str) -> str:
        return self.cfg.step_models.get(step, self.cfg.model)

    def last_model(self) -> str | None:
        """Model that answered this thread's most recent call (may be a fallback)."""
        return getattr(self._local, "model", None)

    def describe(self) -> str:
        steps = ", ".join(f"{k}={v}" for k, v in self.cfg.step_models.items())
        fb = f" then {', '.join(self.cfg.fallback_models)}" if self.cfg.fallback_models else ""
        return f"{self.cfg.provider}:{self.cfg.model}" + (f" ({steps})" if steps else "") + fb

    # ---- public ------------------------------------------------------------ #

    def complete_json(
        self,
        *,
        step: str,
        system: str,
        prompt: str,
        schema: type[T],
        check: Callable[[T], list[str]] | None = None,
        on_check_fail: Literal["raise", "accept"] = "raise",
    ) -> T:
        """Prompt -> validated `schema` instance, with up to cfg.fix_attempts self-corrections.

        `check` adds semantic rules on top of the schema (e.g. "IDs must exist"); its
        errors are fed back like schema errors. With on_check_fail="accept", a reply
        that passes the schema but still fails `check` after all fixes is returned
        and the leftover errors go to `self.warnings`, for the caller to clean up.
        """
        model = self.model_for(step)
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        messages = [
            {"role": "system", "content": system.strip() + "\n\n" + JSON_INSTRUCTIONS.format(schema=schema_json)},
            {"role": "user", "content": prompt},
        ]
        errors: list[str] = []
        last_valid: T | None = None
        for attempt in range(self.cfg.fix_attempts + 1):
            text, finish = self._chat(step, model, messages, schema, attempt)
            obj = None
            try:
                obj = schema.model_validate(extract_json(text))
                last_valid = obj
                errors = check(obj) if check else []
            except ValidationError as e:
                errors = format_validation_error(e)
            except ValueError as e:
                errors = [str(e)]
            if finish == "length":
                errors.insert(0, "your reply was cut off at the output token limit; be more concise")
            if not errors and obj is not None:
                return obj
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": FIX_PROMPT.format(errors="\n".join(f"- {e}" for e in errors[:30]))},
            ]
        if on_check_fail == "accept" and last_valid is not None:
            self.warnings += [f"{step}: {e}" for e in errors]
            return last_valid
        raise LLMError(f"{step}: reply still invalid after {self.cfg.fix_attempts} fixes: {errors[:5]}")

    def usage_summary(self) -> str:
        rows = [f"{'step':<14}{'calls':>6}{'cached':>7}{'prompt':>9}{'output':>8}{'total':>9}{'secs':>7}"]
        grand = Counter()
        for step, c in self.totals.items():
            grand.update(c)
            rows.append(
                f"{step:<14}{c['calls']:>6}{c['cached']:>7}{c['prompt_tokens']:>9}"
                f"{c['completion_tokens']:>8}{c['total_tokens']:>9}{c['latency_s']:>7.0f}"
            )
        if len(self.totals) > 1:
            rows.append(
                f"{'TOTAL':<14}{grand['calls']:>6}{grand['cached']:>7}{grand['prompt_tokens']:>9}"
                f"{grand['completion_tokens']:>8}{grand['total_tokens']:>9}{grand['latency_s']:>7.0f}"
            )
        return "\n".join(rows)

    # ---- internals --------------------------------------------------------- #

    def _body(self, model: str, messages: list[dict], schema: type[BaseModel]) -> dict:
        body: dict = {"model": model, "messages": messages, "temperature": self.cfg.temperature}
        if self._response_format_ok:
            if self.cfg.json_mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            elif self.cfg.json_mode == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                }
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        if self.cfg.max_output_tokens:
            body["max_tokens"] = self.cfg.max_output_tokens
        return body

    def _cache_key(self, model: str, messages: list[dict]) -> str:
        logical = {
            "provider": self.cfg.provider,
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "json_mode": self.cfg.json_mode,
            "reasoning_effort": self.cfg.reasoning_effort,
            "max_output_tokens": self.cfg.max_output_tokens,
        }
        return hashlib.sha256(json.dumps(logical, sort_keys=True).encode("utf-8")).hexdigest()[:32]

    def _chat(self, step: str, model: str, messages: list[dict], schema: type[BaseModel], attempt: int) -> tuple[str, str | None]:
        chain = [model] + [m for m in self.cfg.fallback_models if m != model]
        # 1. any model in the chain that already answered this exact request (re-runs are free)
        if self.cfg.use_cache:
            for m in chain:
                path = self.cache_dir / f"{self._cache_key(m, messages)}.json"
                if path.is_file():
                    rec = json.loads(path.read_text(encoding="utf-8"))
                    self._record(step, m, rec.get("usage") or {}, cached=True, attempt=attempt, latency=0.0)
                    self._local.model = m
                    return rec["text"], rec.get("finish_reason")
        # 2. otherwise the first model whose daily quota isn't used up
        for m in chain:
            if m in self._exhausted:
                continue
            try:
                return self._call_model(step, m, messages, schema, attempt)
            except DailyQuotaExhausted:
                with self._lock:
                    if m not in self._exhausted:
                        self._exhausted.add(m)
                        nxt = next((x for x in chain if x not in self._exhausted), None)
                        msg = f"{m}: daily free-tier quota used up" + (f"; falling back to {nxt}" if nxt else "")
                        self.warnings.append(msg)
                        print(f"  [llm] {msg}", flush=True)
        raise DailyQuotaExhausted(
            f"daily free-tier quota used up for every configured model ({', '.join(chain)}). Finished calls are "
            "cached; re-run after the quota resets, or add fallback_models / switch provider in config.yaml."
        )

    def _call_model(self, step: str, model: str, messages: list[dict], schema: type[BaseModel], attempt: int) -> tuple[str, str | None]:
        path = self.cache_dir / f"{self._cache_key(model, messages)}.json"
        try:
            resp, latency = self._request(self._body(model, messages, schema))
        except _ResponseFormatUnsupported:
            self._response_format_ok = False
            self.warnings.append(f"{self.cfg.provider}:{model} rejected response_format; using prompt-only JSON")
            resp, latency = self._request(self._body(model, messages, schema))

        try:
            choice = resp["choices"][0]
            text = choice["message"].get("content") or ""
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {json.dumps(resp)[:500]}") from None
        usage = resp.get("usage") or {}

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"provider": self.cfg.provider, "model": model, "step": step, "text": text,
                 "finish_reason": finish, "usage": usage, "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
        self._record(step, model, usage, cached=False, attempt=attempt, latency=latency)
        self._local.model = model
        return text, finish

    def _request(self, body: dict) -> tuple[dict, float]:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env)
            if not key:
                raise LLMError(
                    f"{self.cfg.api_key_env} is not set. Put it in .env (see .env.example) "
                    f"to use provider {self.cfg.provider!r}."
                )
            headers["Authorization"] = f"Bearer {key}"

        for attempt in range(self.cfg.http_retries + 1):
            self._throttle()
            t0 = time.monotonic()
            status, text, resp_headers = self._transport(url, headers, body, self.cfg.timeout_s)
            latency = time.monotonic() - t0
            if status == 200:
                try:
                    return json.loads(text), latency
                except json.JSONDecodeError:
                    raise LLMError(f"provider returned non-JSON body: {text[:300]}") from None
            if status == 400 and "response_format" in body and re.search(r"response_format|json_object|json_schema|mime", text, re.I):
                raise _ResponseFormatUnsupported(text[:300])
            if status == 429 and re.search(r"per\s?day|PerDay|daily", text, re.I):
                raise DailyQuotaExhausted(f"daily free-tier quota exhausted for {body['model']}: {text[:300]}")
            if status in (0, 429, 500, 502, 503, 504) and attempt < self.cfg.http_retries:
                delay = _retry_delay(text, resp_headers, attempt)
                print(f"  [llm] HTTP {status or 'network error'}; retrying in {delay:.0f}s "
                      f"({attempt + 1}/{self.cfg.http_retries})", flush=True)
                self._sleep(delay)
                continue
            raise LLMError(f"HTTP {status} from {self.cfg.provider}: {text[:500]}")
        raise LLMError("unreachable")

    def _throttle(self) -> None:
        """Reserve the next request slot (thread-safe), then sleep until it."""
        if not self.cfg.rpm:
            return
        with self._lock:
            slot = max(time.monotonic(), self._next_slot)
            self._next_slot = slot + 60.0 / self.cfg.rpm
        wait = slot - time.monotonic()
        if wait > 0:
            self._sleep(wait)

    def _record(self, step: str, model: str, usage: dict, *, cached: bool, attempt: int, latency: float) -> None:
        details = usage.get("completion_tokens_details") or {}
        rec = {
            "step": step,
            "model": model,
            "cached": cached,
            "attempt": attempt,
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "reasoning_tokens": details.get("reasoning_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
            "latency_s": round(latency, 2),
        }
        # Some providers (Gemini's OpenAI layer) count hidden reasoning only in total_tokens.
        hidden = rec["total_tokens"] - rec["prompt_tokens"] - rec["completion_tokens"]
        if not rec["reasoning_tokens"] and hidden > 0:
            rec["reasoning_tokens"] = hidden
        with self._lock:
            t = self.totals[step]
            t["calls"] += 1
            t["cached"] += int(cached)
            for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "latency_s"):
                t[k] += rec[k]
            if self.usage_log:
                self.usage_log.parent.mkdir(parents=True, exist_ok=True)
                with self.usage_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")
