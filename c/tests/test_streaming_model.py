"""Opt-in, read-only acceptance against an explicitly supplied running gateway.

Set COLI_STREAM_BASE_URL and COLI_STREAM_MODEL_ID (and COLI_API_KEY when needed).
The owner must configure CTX=128 and a generation ceiling of at least 64 tokens.
This suite never launches processes or changes model files.
"""
import json
import os
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class StreamingModelTest(unittest.TestCase):
    PROMPT = "What is the capital of France? Reply briefly."

    @classmethod
    def setUpClass(cls):
        base = os.environ.get("COLI_STREAM_BASE_URL")
        model = os.environ.get("COLI_STREAM_MODEL_ID")
        if not base and not model:
            raise unittest.SkipTest("Set COLI_STREAM_BASE_URL and COLI_STREAM_MODEL_ID for real-model acceptance.")
        if not base or not model:
            missing = "COLI_STREAM_BASE_URL" if not base else "COLI_STREAM_MODEL_ID"
            raise RuntimeError("Real-model acceptance requires " + missing + ".")
        cls.base = base.rstrip("/").removesuffix("/v1")
        cls.model = model
        cls.key = os.environ.get("COLI_API_KEY", "")

    def request(self, path, body=None, timeout=120):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        payload = None if body is None else json.dumps(body).encode()
        return urlopen(Request(self.base + path, data=payload, headers=headers), timeout=timeout)

    def chat_body(self, stream=False, prompt=None, maximum=16):
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt or self.PROMPT}],
                "temperature": 0, "top_p": 1, "max_tokens": maximum,
                "enable_thinking": False, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def events(self, response):
        """Parse complete named/data SSE events, with a finite transaction deadline."""
        deadline = time.monotonic() + 120
        data, name = [], "message"
        for raw in response:
            if time.monotonic() >= deadline:
                self.fail("The response exceeded the streaming transaction deadline.")
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if data:
                    yield name, "\n".join(data)
                data, name = [], "message"
                continue
            field, sep, value = line.partition(":")
            if not sep:
                value = ""
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                name = value
        self.assertFalse(data, "The stream ended inside an SSE event.")

    def openai_stream(self, path, body, legacy=False):
        chunks, usage, finish, done = [], None, None, 0
        with self.request(path, body) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))
            for _, data in self.events(response):
                if data == "[DONE]":
                    done += 1
                    continue
                self.assertEqual(done, 0, "An event arrived after [DONE].")
                event = json.loads(data)
                self.assertNotIn("error", event)
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    self.assertFalse(delta.get("tool_calls"), "Unexpected tool output.")
                    chunks.append(choice.get("text", "") if legacy else delta.get("content", ""))
                    if choice.get("finish_reason") is not None:
                        self.assertIsNone(finish, "Duplicate finish choice.")
                        finish = choice["finish_reason"]
        self.assertEqual(done, 1)
        self.assertIsNotNone(finish)
        self.assertIsNotNone(usage)
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])
        return "".join(chunks), usage, finish

    def test_01_chat_json_sse_json_sse(self):
        with self.request("/v1/chat/completions", self.chat_body()) as response:
            baseline = json.load(response)
        answer = baseline["choices"][0]["message"]["content"]
        self.assertIn("Paris", answer)
        for streaming in (True, False, True):
            if streaming:
                text, usage, finish = self.openai_stream("/v1/chat/completions", self.chat_body(True))
                self.assertEqual(text, answer)
                self.assertEqual(usage, baseline["usage"])
                self.assertEqual(finish, baseline["choices"][0]["finish_reason"])
            else:
                with self.request("/v1/chat/completions", self.chat_body()) as response:
                    payload = json.load(response)
                self.assertEqual(payload["choices"][0]["message"]["content"], answer)
                self.assertEqual(payload["usage"], baseline["usage"])

    def test_02_legacy_completion(self):
        body = {"model": self.model, "prompt": "The capital of France is",
                "temperature": 0, "top_p": 1, "max_tokens": 16}
        with self.request("/v1/completions", body) as response:
            baseline = json.load(response)
        self.assertIn("Paris", baseline["choices"][0]["text"])
        text, usage, finish = self.openai_stream("/v1/completions", dict(
            body, stream=True, stream_options={"include_usage": True}), legacy=True)
        self.assertEqual(text, baseline["choices"][0]["text"])
        self.assertEqual(usage, baseline["usage"])
        self.assertEqual(finish, baseline["choices"][0]["finish_reason"])

    def test_03_anthropic_json_and_stream(self):
        body = {"model": self.model, "max_tokens": 16, "temperature": 0, "top_p": 1,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": self.PROMPT}]}
        with self.request("/v1/messages", body) as response:
            baseline = json.load(response)
        answer = "".join(block["text"] for block in baseline["content"] if block["type"] == "text")
        self.assertIn("Paris", answer)
        text, usage, names, stop_reason = [], {}, [], None
        with self.request("/v1/messages", dict(body, stream=True)) as response:
            for name, data in self.events(response):
                event = json.loads(data)
                self.assertEqual(name, event["type"])
                self.assertNotEqual(name, "error")
                self.assertNotIn("message_stop", names, "An event arrived after message_stop.")
                names.append(name)
                if name == "message_start":
                    usage.update(event["message"]["usage"])
                elif name == "content_block_delta" and event["delta"]["type"] == "text_delta":
                    text.append(event["delta"]["text"])
                elif name == "message_delta":
                    usage.update(event["usage"])
                    stop_reason = event["delta"]["stop_reason"]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        self.assertEqual(names.count("message_stop"), 1)
        self.assertEqual("".join(text), answer)
        self.assertEqual(usage, baseline["usage"])
        self.assertGreater(usage["input_tokens"], 0)
        self.assertGreater(usage["output_tokens"], 0)
        self.assertEqual(stop_reason, baseline["stop_reason"])

    def test_04_context_rejection_before_commit(self):
        prompt = "outside context " * 300
        bodies = [("/v1/chat/completions", self.chat_body(True, prompt)),
                  ("/v1/messages", {"model": self.model, "stream": True, "max_tokens": 16,
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": prompt}]})]
        for path, body in bodies:
            with self.subTest(path=path):
                with self.assertRaises(HTTPError) as caught:
                    self.request(path, body)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn("application/json", response.headers.get("Content-Type", ""))
                    payload = json.load(response)
                self.assertIn("context", payload["error"]["message"].lower())

    def test_05_disconnect_drains_and_next_request_succeeds(self):
        body = self.chat_body(True,
            "Count the integers from 1 to 100, separated by commas. Do not stop early.", 64)
        saw_content = False
        with self.request("/v1/chat/completions", body) as response:
            for _, data in self.events(response):
                self.assertNotEqual(data, "[DONE]", "Counting ended before a mid-generation disconnect.")
                event = json.loads(data)
                self.assertNotIn("error", event)
                if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                    with self.request("/health", timeout=5) as health:
                        active = json.load(health)["scheduler"]["active"]
                    self.assertGreater(active, 0, "Counting already finished; cancellation was not exercised.")
                    saw_content = True
                    break
        self.assertTrue(saw_content)
        deadline = time.monotonic() + 30
        while True:
            with self.request("/health", timeout=5) as health:
                active = json.load(health)["scheduler"]["active"]
            if not active:
                break
            self.assertLess(time.monotonic(), deadline, "Cancellation did not release the scheduler within 30 seconds.")
            time.sleep(0.05)
        text, _, _ = self.openai_stream("/v1/chat/completions", self.chat_body(True))
        self.assertIn("Paris", text)
