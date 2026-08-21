"""Backend behaviour: budget enforcement, caching, and request construction.

The budget tests exist because the ceiling was genuinely broken: five
concurrent specialists all read the same under-budget count before any of them
finished, and a ceiling of two let six calls through. That is exactly the kind
of defect a single-threaded test would never see, so these run real threads.

The Anthropic tests use a stub client. That cannot prove the server accepts the
request, but it does prove the request we build matches the shape the installed
SDK declares, and that we parse the response — including a refusal — correctly.
"""

from __future__ import annotations

import concurrent.futures as futures
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from analyst import Settings
from analyst.config import PROVIDER_ANTHROPIC, PROVIDER_MOCK, PROVIDER_OPENAI
from analyst.llm import (
    BudgetExceeded,
    Completion,
    MockBackend,
    extract_json,
    make_backend,
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class BudgetCeilingTests(unittest.TestCase):
    def test_ceiling_holds_under_concurrency(self):
        """The regression test for the check-then-act race.

        Ten threads race for three slots. Reading the count and then deciding
        lets all ten through; reserving inside the lock does not.
        """
        backend = MockBackend(Settings(backend=PROVIDER_MOCK, max_llm_calls=3, use_cache=False))
        granted = []
        lock = threading.Lock()
        barrier = threading.Barrier(10)

        def attempt() -> None:
            barrier.wait()  # maximise the overlap
            try:
                backend._reserve_call()
            except BudgetExceeded:
                return
            with lock:
                granted.append(1)

        with futures.ThreadPoolExecutor(max_workers=10) as pool:
            list(futures.as_completed([pool.submit(attempt) for _ in range(10)]))

        self.assertEqual(len(granted), 3, "more calls were admitted than the ceiling allows")

    def test_ceiling_raises_once_exhausted(self):
        backend = MockBackend(Settings(backend=PROVIDER_MOCK, max_llm_calls=2, use_cache=False))
        backend._reserve_call()
        backend._reserve_call()
        with self.assertRaises(BudgetExceeded):
            backend._reserve_call()

    def test_cost_ceiling_trips(self):
        backend = MockBackend(Settings(backend=PROVIDER_MOCK, max_cost_usd=1.0, use_cache=False))
        backend.usage.cost_usd = 1.5
        with self.assertRaises(BudgetExceeded):
            backend._reserve_call()


class ModelResolutionTests(unittest.TestCase):
    """Switching backend without naming a model must not send an
    Anthropic model id to OpenAI — that produced a 404 on every agent."""

    def test_model_defaults_follow_the_backend(self):
        self.assertEqual(Settings(backend=PROVIDER_OPENAI).model, "gpt-5.6-luna")
        self.assertEqual(Settings(backend=PROVIDER_ANTHROPIC).model, "claude-opus-5")

    def test_explicit_model_is_respected(self):
        self.assertEqual(Settings(backend=PROVIDER_OPENAI, model="gpt-4.1").model, "gpt-4.1")


class JsonExtractionTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced_block(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_object_embedded_in_prose(self):
        self.assertEqual(extract_json('Sure!\n{"a": 1}\nHope that helps.'), {"a": 1})

    def test_empty_output_is_an_error(self):
        with self.assertRaises(Exception):
            extract_json("")


class CacheTests(unittest.TestCase):
    def test_identical_prompts_hit_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("analyst.llm.CACHE_DIR", __import__("pathlib").Path(tmp)):
                settings = Settings(backend=PROVIDER_MOCK, use_cache=True)
                backend = MockBackend(settings)
                key = "abc123"
                backend._cache_put(key, Completion(text='{"a": 1}', model="m", input_tokens=5, output_tokens=6))
                hit = backend._cache_get(key)
                self.assertIsNotNone(hit)
                self.assertTrue(hit.cached)
                self.assertEqual(hit.text, '{"a": 1}')

    def test_cache_disabled_returns_nothing(self):
        backend = MockBackend(Settings(backend=PROVIDER_MOCK, use_cache=False))
        self.assertIsNone(backend._cache_get("whatever"))


def _stub_anthropic_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=11, output_tokens=22),
    )


class AnthropicRequestTests(unittest.TestCase):
    """Exercises the Anthropic path without spending credit.

    Everything except the server's own acceptance of the payload is covered:
    the kwargs we build, and how we read the response back.
    """

    def _backend(self, response):
        settings = Settings(backend=PROVIDER_ANTHROPIC, model="claude-opus-5", use_cache=False)
        with mock.patch("analyst.config.load_api_key", return_value="sk-ant-test"), \
             mock.patch("anthropic.Anthropic") as client_cls:
            client_cls.return_value.messages.create.return_value = response
            backend = make_backend(settings)
        return backend, backend.client.messages.create

    def test_request_matches_the_installed_sdk_shape(self):
        backend, create = self._backend(_stub_anthropic_response('{"answer": "ok"}'))
        backend.complete(role="finance", prompt="p", system="s", schema=SCHEMA)

        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertEqual(kwargs["system"], "s")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "p"}])
        self.assertIn("max_tokens", kwargs)

        # output_config carries effort and the JSON schema — the two keys the
        # installed SDK's OutputConfigParam declares.
        oc = kwargs["output_config"]
        self.assertIn("effort", oc)
        self.assertEqual(oc["format"], {"type": "json_schema", "schema": SCHEMA})

        from anthropic.types.output_config_param import OutputConfigParam
        import typing

        self.assertTrue(set(oc).issubset(set(typing.get_type_hints(OutputConfigParam))))

    def test_response_is_parsed_and_accounted(self):
        backend, _ = self._backend(_stub_anthropic_response('{"answer": "ok"}'))
        completion = backend.complete(role="finance", prompt="p", schema=SCHEMA)
        self.assertEqual(extract_json(completion.text), {"answer": "ok"})
        self.assertEqual((completion.input_tokens, completion.output_tokens), (11, 22))
        self.assertEqual(backend.usage.calls, 1)
        self.assertGreater(backend.usage.cost_usd, 0)

    def test_refusal_yields_empty_text_not_a_crash(self):
        backend, _ = self._backend(_stub_anthropic_response("nope", stop_reason="refusal"))
        completion = backend.complete(role="finance", prompt="p", schema=SCHEMA)
        self.assertEqual(completion.text, "")

    def test_missing_key_is_a_clear_error(self):
        with mock.patch("analyst.llm.load_api_key", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                make_backend(Settings(backend=PROVIDER_ANTHROPIC))
        self.assertIn("No Anthropic API key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()