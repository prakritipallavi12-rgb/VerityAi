"""
utils/llm.py - The ONLY place in the project that talks to an LLM.

We use the `openai` library, but it can talk to any provider with the same format
(Groq, OpenAI, Gemini, OpenRouter for free Llama, Ollama for local Llama).
A model is written as "Provider::model-id", e.g. "Groq::openai/gpt-oss-20b", so the
generator and the judge can come from different companies.

Three jobs besides sending prompts:
  1. Strict JSON  - ask_json() asks for JSON mode, checks required keys, retries with the error.
  2. Rate limits  - a limiter per model waits before a call would break the provider's
                    requests/tokens-per-minute limit (it learns the real limits from responses).
  3. Usage        - every call records tokens, seconds and estimated cost, so runs can show
                    "tokens used" and "seconds per question" (see usage_scope / role).
"""

import json
import re
import threading
import time
from collections import deque
from contextlib import contextmanager

import requests
from openai import AuthenticationError, BadRequestError, NotFoundError, OpenAI

import config

_clients = {}
_lock = threading.Lock()
_local = threading.local()  # per-thread: current usage log + role ("generator", "judge"...)

# Model names containing these words are not chat models (speech, safety filters...)
_NOT_CHAT = ("whisper", "orpheus", "guard", "tts", "embed", "audio", "image", "moderation", "allam")


# ======================================================================
# Providers and models
# ======================================================================
_ollama_checked = {"time": 0.0, "up": False}


def _ollama_running():
    """Ollama has no key - it is 'available' when its local server answers. Checked at most every 30 s."""
    if time.time() - _ollama_checked["time"] > 30:
        base = config.PROVIDERS["Ollama"]["base_url"].rsplit("/v1", 1)[0]
        try:
            _ollama_checked["up"] = requests.get(f"{base}/api/tags", timeout=0.5).ok
        except requests.RequestException:
            _ollama_checked["up"] = False
        _ollama_checked["time"] = time.time()
    return _ollama_checked["up"]


def _has_key(provider):
    p = config.PROVIDERS.get(provider, {})
    if p.get("local"):
        return _ollama_running()
    key = p.get("key")
    return bool(key) and not key.startswith("paste-your")


def available_providers():
    """Providers that can be used right now (key in .env, or local server running)."""
    return [name for name in config.PROVIDERS if _has_key(name)]


def get_client(provider=None):
    """Create one client per provider and reuse it."""
    provider = provider or config.DEFAULT_PROVIDER
    with _lock:
        if provider not in _clients:
            if not _has_key(provider):
                raise RuntimeError(f"{provider} is not available. Add its key to .env (see .env.example)"
                                   + (" or start Ollama." if provider == "Ollama" else "."))
            p = config.PROVIDERS[provider]
            # max_retries: the library itself waits and retries on rate limits (429) and timeouts
            _clients[provider] = OpenAI(api_key=p["key"], base_url=p["base_url"], max_retries=6, timeout=120)
        return _clients[provider]


def parse_model(spec=None):
    """'Groq::openai/gpt-oss-20b' -> ('Groq', 'openai/gpt-oss-20b')."""
    if spec and "::" in spec:
        provider, model = spec.split("::", 1)
        return provider, model
    return config.DEFAULT_PROVIDER, spec or config.LLM_MODEL


def model_label(spec):
    provider, model = parse_model(spec)
    return f"{provider} · {model}"


def list_models(provider):
    """Chat models this provider offers to our key (falls back to a fixed list)."""
    try:
        ids = sorted(m.id.replace("models/", "") for m in get_client(provider).models.list().data)
        chat = [i for i in ids if not any(word in i.lower() for word in _NOT_CHAT)]
        if provider == "OpenRouter":  # hundreds of models - keep the free Llama ones
            chat = [i for i in chat if i.endswith(":free") and "llama" in i.lower()]
        if provider == "Gemini":  # many specialised models - keep the general flash / pro chat models
            # (gemini-2.x is listed but closed to new users, so it is left out)
            chat = [i for i in chat if re.fullmatch(r"gemini-[\d.]+-(flash|pro)(-lite)?(-preview)?", i)
                    and not i.startswith("gemini-2")]
        if config.PROVIDERS[provider].get("local"):
            return chat  # only models that are actually downloaded (e.g. after `ollama pull llama3.2`)
        return chat or config.FALLBACK_MODELS.get(provider, [])
    except Exception:
        return [] if config.PROVIDERS.get(provider, {}).get("local") else config.FALLBACK_MODELS.get(provider, [])


# ======================================================================
# Rate limiter: never send more than the provider allows per minute
# ======================================================================
class RateLimiter:
    """Sliding 60-second window of requests and tokens for ONE model."""

    def __init__(self, rpm, tpm):
        self.rpm, self.tpm = rpm, tpm
        self.events = deque()  # (time, tokens)
        self.lock = threading.Lock()

    def _trim(self, now):
        while self.events and now - self.events[0][0] > 60:
            self.events.popleft()

    def acquire(self, tokens):
        """Block until a call of about `tokens` tokens fits in the window, then reserve it."""
        tokens = min(tokens, self.tpm)
        while True:
            with self.lock:
                now = time.time()
                self._trim(now)
                used = sum(t for _, t in self.events)
                if len(self.events) < self.rpm and used + tokens <= self.tpm:
                    self.events.append((now, tokens))
                    return
                wait = 60 - (now - self.events[0][0]) + 0.05 if self.events else 0.5
            time.sleep(max(0.05, min(wait, 1.0)))

    def correct(self, estimated, actual):
        """Replace our guess with the real token count once the reply arrives."""
        with self.lock:
            self.events.append((time.time(), actual - estimated))

    def learn(self, headers):
        """Providers send their real limits in response headers - use them (paid tiers get faster)."""
        try:
            tpm = int(headers.get("x-ratelimit-limit-tokens") or 0)
            if tpm > 0:
                self.tpm = tpm
        except ValueError:
            pass


_limiters = {}


def _limiter(provider, model_id):
    key = f"{provider}::{model_id}"
    with _lock:
        if key not in _limiters:
            rpm, tpm = config.RATE_LIMITS.get(provider, (60, 100000))
            _limiters[key] = RateLimiter(rpm, tpm)
        return _limiters[key]


# ======================================================================
# Usage tracking: tokens, seconds and cost of every call
# ======================================================================
@contextmanager
def usage_scope():
    """Collect every LLM call made by this thread inside the `with` block into a list."""
    previous = getattr(_local, "log", None)
    _local.log = []
    try:
        yield _local.log
    finally:
        _local.log = previous


@contextmanager
def role(name):
    """Tag calls inside the block, e.g. role('judge'), so usage can be split by stage."""
    previous = getattr(_local, "role", None)
    _local.role = name
    try:
        yield
    finally:
        _local.role = previous


def estimate_cost(provider, model_id, prompt_tokens, completion_tokens):
    if provider in config.FREE_PROVIDERS or model_id.endswith(":free"):
        return 0.0
    price_in, price_out = config.PRICES.get(model_id, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def _record(provider, model_id, usage, seconds):
    log = getattr(_local, "log", None)
    if log is None:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    log.append({"role": getattr(_local, "role", None) or "other", "model": f"{provider}::{model_id}",
                "prompt_tokens": prompt, "completion_tokens": completion, "seconds": round(seconds, 3),
                "cost": estimate_cost(provider, model_id, prompt, completion)})


# ======================================================================
# Calls
# ======================================================================
def _complete(prompt, system, model, json_mode, max_tokens=None):
    """One raw API call (rate-limited and tracked). Returns the reply text."""
    provider, model_id = parse_model(model)
    kwargs = {
        "model": model_id,
        "temperature": config.LLM_TEMPERATURE,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}  # forces the model to output JSON
    if "gpt-oss" in model_id:
        kwargs["reasoning_effort"] = "low"  # these models "think" first; low = faster and fewer tokens

    client = get_client(provider)
    limiter = _limiter(provider, model_id)
    estimate = (len(system) + len(prompt)) // 3 + (max_tokens or 400)
    limiter.acquire(estimate)
    start = time.time()
    try:
        raw = client.chat.completions.with_raw_response.create(**kwargs)
    except BadRequestError as e:
        # Some models reject an option (temperature, JSON mode, reasoning). Drop it and try once more.
        text = str(e).lower()
        for option in ("temperature", "response_format", "reasoning_effort"):
            if option in text:
                kwargs.pop(option, None)
        raw = client.chat.completions.with_raw_response.create(**kwargs)
    response = raw.parse()
    limiter.learn(raw.headers)
    usage = response.usage
    limiter.correct(estimate, getattr(usage, "total_tokens", estimate) or estimate)
    _record(provider, model_id, usage, time.time() - start)

    reply = response.choices[0].message.content or ""
    # Some models write their thinking inside <think>...</think> - remove it
    return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()


class DailyLimitError(RuntimeError):
    """The provider's daily token/request quota is used up - retrying today will not help."""


def _check_daily_limit(error, model):
    text = str(error).lower()
    # Groq says "rate limit ... tokens per day"; Gemini says "quota ... requests per day limit exceeded"
    if "per day" in text and any(w in text for w in ("rate limit", "rate_limit", "quota", "resource_exhausted")):
        raise DailyLimitError(f"{model_label(model)} has used its free daily limit. "
                              "Pick another model (or the local judge), or try again tomorrow.") from error


def ask(prompt, system="You are a helpful assistant.", model=None, max_tokens=None):
    """Send a prompt to the LLM and return its reply as plain text."""
    last_error = None
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            return _complete(prompt, system, model, json_mode=False, max_tokens=max_tokens)
        except (AuthenticationError, NotFoundError):
            raise  # wrong key or unknown model - retrying will not help
        except Exception as e:
            _check_daily_limit(e, model)
            last_error = e
            print(f"[llm] attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)  # wait a bit longer each time
    raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} attempts: {last_error}")


def extract_json(text):
    """Pull a JSON object out of a reply, even if the model wrapped it in ``` fences or extra words."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def ask_json(prompt, system="You are a helpful assistant. Reply with valid JSON only.",
             model=None, required_keys=(), max_tokens=None):
    """
    Send a prompt and return the reply as a Python dict.

    LLMs sometimes return broken JSON - this is the #1 crash in RAG projects. So we:
      1. ask for JSON mode
      2. parse it (also handles ``` fences and extra text around the JSON)
      3. check every required key is present
      4. if anything fails, retry and tell the model exactly what was wrong
    """
    last_error = None
    current_prompt = prompt
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            reply = _complete(current_prompt, system, model, json_mode=True, max_tokens=max_tokens)
            data = extract_json(reply)
            if not isinstance(data, dict):
                raise ValueError("reply is not a JSON object")
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"missing keys: {missing}")
            return data
        except (AuthenticationError, NotFoundError):
            raise
        except Exception as e:  # bad JSON, missing keys, or an API error
            _check_daily_limit(e, model)
            last_error = e
            print(f"[llm] JSON attempt {attempt} failed: {e}")
            current_prompt = (
                f"{prompt}\n\nYour previous reply was rejected ({e}). "
                f"Reply with ONLY one valid JSON object containing the keys: {list(required_keys)}."
            )
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not get valid JSON after {config.LLM_MAX_RETRIES} attempts: {last_error}")
