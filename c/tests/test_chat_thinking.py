"""#979: `coli chat` must SHOW the model's thinking, and be able to hide it.

The server streams thinking as `reasoning_content` deltas; chat_attached's
loop used to drop every non-content delta (its comment listed `reasoning`
among the discards), so `coli run` showed reasoning while `coli chat`
silently hid it. These tests drive the real chat_attached under a pty
against a stdlib SSE server:

  1. default: the reasoning text is visible, inside the thinking box;
  2. COLI_SHOW_THINK=0: the reasoning is absent, the answer still arrives;
  3. both ways: the assistant history carries `content` only -- the second
     request's messages must not contain the scratchpad.
"""
import contextlib
import http.client
import io
import importlib.machinery
import importlib.util
import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest

try:
    import pty                      # POSIX-only: no termios/pty on Windows.
    HAVE_PTY = True
except ImportError:                 # The TUI path under test needs a real
    HAVE_PTY = False                # terminal; Linux and macOS runners cover it.
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

CLI = Path(__file__).resolve().parent.parent / "coli"

SEEN_MESSAGES = []

# The child is spawned with fork+EXEC, never bare fork: this test process is
# multi-threaded (the fake SSE server runs in a thread), and pty.fork() from a
# multi-threaded process inherits half-held runtime locks -- Python itself
# warns "use of forkpty() may lead to deadlocks in the child", and on the
# macOS runner it did exactly that, reliably, until the job timeout.
BOOTSTRAP = """
import importlib.machinery, importlib.util, os, sys
from types import SimpleNamespace
loader = importlib.machinery.SourceFileLoader("coli_t", os.environ["COLI_TEST_CLI"])
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
sys.argv = ["coli"]
try:
    loader.exec_module(module)
except SystemExit:
    pass
module.chat_attached(SimpleNamespace(ngen=32, api_key=None),
                     os.environ["COLI_TEST_BASE"], "fake-model")
"""


class FakeSSE(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN_MESSAGES.append(body["messages"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def event(delta):
            payload = {"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")

        event({"role": "assistant"})
        scenario = getattr(self.server, "scenario", None)
        if len(SEEN_MESSAGES) == 1 and scenario:
            event({"reasoning_content": "PRIVATE"})
            event({"content": "```python\nPARTIAL"})
            if scenario == "error":
                self.wfile.write(b'data:{"error":{"message":"native failed"}}\n\n')
            return
        # Both channels may share a delta. Exercise assembled, no-space SSE
        # fields over a real terminal, without putting reasoning in history.
        payload = json.dumps({"choices": [{"delta": {
            "reasoning_content": "SCRATCHPAD-ALPHA SCRATCHPAD-BETA",
            "content": "ANSWER-GAMMA"}}]})
        self.wfile.write(b": heartbeat\r\nevent: message\r\n")
        self.wfile.write(b"data:" + payload[:12].encode() + b"\r\n")
        self.wfile.write(b"data:" + payload[12:].encode() + b"\r\n\r\n")
        self.wfile.write(b"data: [DONE]\n\n")


def run_chat(base, extra_env):
    """Drive chat_attached through a pty slave: two prompts, then quit."""
    master, slave = pty.openpty()
    env = dict(os.environ, COLI_TEST_CLI=str(CLI), COLI_TEST_BASE=base,
               TERM="xterm", **extra_env)
    child = subprocess.Popen([sys.executable, "-c", BOOTSTRAP],
                             stdin=slave, stdout=slave, stderr=slave,
                             env=env, start_new_session=True, close_fds=True)
    os.close(slave)
    fd = master

    # Lockstep, not fire-and-forget: writing lines before the child reaches
    # input() is timing-dependent -- on macOS, libedit's initialization at the
    # first input() flushes the tty buffer and eats pre-written lines (the
    # readline activation from #928), while Linux's GNU readline preserves
    # them. Wait for each marker before typing, like a human would, with a
    # strictly-advancing cursor so an old prompt can never satisfy a new wait.
    output = b""
    pos = 0
    deadline = time.time() + 30

    def read_until(needle):
        nonlocal output, pos
        while time.time() < deadline:
            found = output.find(needle, pos)
            if found >= 0:
                pos = found + len(needle)
                return True
            if not select.select([fd], [], [], max(0, deadline-time.time()))[0]:
                return False
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return False
            if not chunk:
                return False
            output += chunk
        return False

    prompt = "\u203a".encode()           # the › of a fresh input box
    if read_until(prompt):
        os.write(fd, b"first question\n")
    read_until("◆ colibri".encode())     # skip the echoed/redrawn ›
    if read_until(prompt):
        os.write(fd, b"second question\n")
    read_until("◆ colibri".encode())
    if read_until(prompt):
        os.write(fd, b":q\n")
    while time.time() < deadline:
        if not select.select([fd], [], [], max(0, deadline-time.time()))[0]:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    os.close(master)
    if child.returncode != 0:
        raise AssertionError("chat child failed:\n" + output.decode("utf-8", "replace"))
    return output.decode("utf-8", "replace")


@unittest.skipUnless(HAVE_PTY, "pty is POSIX-only; covered on Linux/macOS")
class ChatThinkingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSE)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        SEEN_MESSAGES.clear()
        self.server.scenario = None

    def test_thinking_is_shown_by_default(self):
        out = run_chat(self.base, {"COLI_SHOW_THINK": "1"})
        self.assertIn("SCRATCHPAD-ALPHA", out, "reasoning deltas must render")
        self.assertIn("thinking", out, "the thinking box must announce itself")
        self.assertIn("ANSWER-GAMMA", out)

    def test_toggle_hides_thinking(self):
        out = run_chat(self.base, {"COLI_SHOW_THINK": "0"})
        self.assertNotIn("SCRATCHPAD-ALPHA", out,
                         "COLI_SHOW_THINK=0 must suppress the scratchpad")
        self.assertIn("ANSWER-GAMMA", out, "the answer must still stream")

    def test_history_carries_content_only(self):
        run_chat(self.base, {"COLI_SHOW_THINK": "1"})
        self.assertGreaterEqual(len(SEEN_MESSAGES), 2, "need the second request")
        second = SEEN_MESSAGES[1]
        assistant = [m for m in second if m["role"] == "assistant"]
        self.assertTrue(assistant)
        self.assertEqual(assistant[0]["content"], "ANSWER-GAMMA")
        self.assertNotIn("SCRATCHPAD", json.dumps(second),
                         "the scratchpad must never re-enter the context")

    def test_error_and_truncation_recover_with_partial_history(self):
        for scenario, diagnostic in (
                ("error", "native failed"),
                ("truncated", "The response stream ended before [DONE].")):
            with self.subTest(scenario=scenario):
                SEEN_MESSAGES.clear()
                self.server.scenario = scenario
                out = run_chat(self.base, {"COLI_SHOW_THINK": "1"})
                self.assertIn(diagnostic, out)
                self.assertNotIn("Traceback", out)
                self.assertIn("ANSWER-GAMMA", out)
                self.assertEqual(out.count("tok ·"), 1)
                self.assertEqual(SEEN_MESSAGES[1], [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "```python\nPARTIAL"},
                    {"role": "user", "content": "second question"}])


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_chat_test", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "argv", ["coli"]):
        loader.exec_module(module)
    return module


def sse(delta):
    return ("data: " + json.dumps({"choices": [{"delta": delta}]},
                                ensure_ascii=False) + "\n\n").encode()


class StreamResponse(io.BytesIO):
    """HTTP reader shape, with observable reads and transport failures."""
    def __init__(self, chunks, stay_open=False):
        self.chunks = list(chunks)
        self.stay_open = stay_open
        self.reads = 0
        self.was_closed = False
        super().__init__(b"".join(c for c in chunks if isinstance(c, bytes)))

    def read1(self, size=-1):
        self.reads += 1
        if self.chunks:
            chunk = self.chunks.pop(0)
            if isinstance(chunk, BaseException):
                raise chunk
            return chunk
        if self.stay_open:
            raise AssertionError("consumer read beyond DONE on an open source")
        return b""

    def close(self):
        self.was_closed = True
        super().close()


class ChatStreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def run_chat(self, responses, show_think="1", prompts=None):
        seen = []
        output = io.StringIO()
        real_md = self.cli.MDStream
        renderers = []

        def make_md(*args):
            md = real_md(*args)
            md.close = mock.Mock(wraps=md.close)
            renderers.append(md)
            return md

        def open_request(req):
            seen.append(json.loads(req.data)["messages"])
            response = responses[len(seen)-1]
            if isinstance(response, BaseException):
                raise response
            return response

        if prompts is None:
            prompts = ["question "+str(i) for i in range(len(responses))]
        with mock.patch.object(self.cli, "TTY", False), \
                mock.patch.object(self.cli, "MDStream", side_effect=make_md), \
                mock.patch.object(self.cli, "Spinner") as spinner, \
                mock.patch("urllib.request.urlopen", side_effect=open_request), \
                mock.patch("builtins.input", side_effect=prompts + [":q"]), \
                mock.patch.dict(os.environ, {"COLI_SHOW_THINK": show_think}), \
                contextlib.redirect_stdout(output):
            self.cli.chat_attached(SimpleNamespace(ngen=32, api_key=None),
                                   "http://127.0.0.1:1", "fake")
        for md in renderers:
            md.close.assert_called_once()
        self.assertGreaterEqual(spinner.return_value.stop.call_count,
                                len(responses))
        for response in responses:
            if isinstance(response, StreamResponse):
                self.assertTrue(response.was_closed)
        return output.getvalue(), seen

    def test_combined_channels_visible_and_hidden(self):
        for visible in ("0", "1"):
            with self.subTest(visible=visible):
                out, seen = self.run_chat([
                    StreamResponse([sse({"reasoning_content": "PRIVATE",
                                         "content": "ANSWER"}),
                                    b"data:[DONE]\n\n"]),
                    StreamResponse([sse({"content": "NEXT"}),
                                    b"data:[DONE]\n\n"])], visible)
                self.assertEqual(out.count("ANSWER"), 1)
                self.assertEqual("PRIVATE" in out, visible == "1")
                self.assertEqual(seen[1][1],
                                 {"role": "assistant", "content": "ANSWER"})
                self.assertNotIn("PRIVATE", json.dumps(seen))

    def test_done_does_not_wait_for_eof_or_consume_later_events(self):
        response = StreamResponse([sse({"content": "ANSWER"}) +
            b"data:[DONE]\n\n" + sse({"content": "FORBIDDEN"})], stay_open=True)
        out, _ = self.run_chat([response])
        self.assertIn("ANSWER", out)
        self.assertNotIn("FORBIDDEN", out)
        self.assertEqual(response.reads, 1)

    def test_errors_keep_only_answer_history_and_recover(self):
        cases = [
            ([b'data:{"error":{"message":"native failed"}}\n\n'],
             "native failed"),
            ([b'data:{"error":{}}\n\n'],
             "The server reported a streaming error."),
            ([b"data:{broken}\n\n"],
             "The server returned an invalid streaming event."),
            ([], "The response stream ended before [DONE]."),
            ([b"data:[DONE]"], "The response stream ended before [DONE]."),
            ([sse({"tool_calls": [{"index": 0}]})],
             "This text chat does not support tool calls."),
            ([http.client.IncompleteRead(b"fragment", 10)],
             "The response stream ended before [DONE]."),
            ([http.client.RemoteDisconnected("peer closed")],
             "The response stream ended before [DONE]."),
            ([OSError("connection reset")], "connection reset"),
            ([KeyboardInterrupt()], "interrupted"),
        ]
        for chunks, diagnostic in cases:
            for partial in (False, True):
                with self.subTest(diagnostic=diagnostic, partial=partial):
                    initial = [sse({"reasoning_content": "PRIVATE"})]
                    if partial:
                        initial.append(sse({"content": "```python\npartial"}))
                    out, seen = self.run_chat([
                        StreamResponse(initial + chunks),
                        StreamResponse([sse({"content": "RECOVERED"}),
                                        b"data:[DONE]\n\n"])])
                    self.assertIn(diagnostic, out)
                    self.assertEqual(out.count("tok ·"), 1,
                                     "only recovery has a success footer")
                    self.assertEqual(out.count("┌ thinking"), 1)
                    self.assertIn("└─", out)
                    expected = ([{"role": "user", "content": "question 0"},
                                 {"role": "assistant",
                                  "content": "```python\npartial"}]
                                if partial else [])
                    self.assertEqual(seen[1], expected + [
                        {"role": "user", "content": "question 1"}])

    def test_empty_completion_is_diagnostic_and_not_history(self):
        for delta in ({}, {"reasoning_content": "PRIVATE"}):
            with self.subTest(delta=delta):
                out, seen = self.run_chat([
                    StreamResponse([sse(delta), b"data:[DONE]\n\n"]),
                    StreamResponse([sse({"content": "RECOVERED"}),
                                    b"data:[DONE]\n\n"])])
                self.assertIn("The server returned no answer content.", out)
                self.assertEqual(out.count("tok ·"), 1)
                self.assertEqual(seen[1],
                                 [{"role": "user", "content": "question 1"}])

    def test_http_error_preserves_status_and_detail_and_cleanup(self):
        import urllib.error
        rejected = urllib.error.HTTPError("http://localhost", 400, "bad", {},
            io.BytesIO(b'{"error":{"message":"context exceeded"}}'))
        out, seen = self.run_chat([rejected,
            StreamResponse([sse({"content": "RECOVERED"}), b"data:[DONE]\n\n"])])
        self.assertIn("HTTP 400 — context exceeded", out)
        self.assertEqual(seen[1], [{"role": "user", "content": "question 1"}])

    def test_usage_only_frame_preserves_exact_footer(self):
        usage = b'data:{"choices":[],"usage":{"completion_tokens":37}}\n\n'
        out, _ = self.run_chat([
            StreamResponse([sse({"content": "ANSWER"}), usage,
                            b"data:[DONE]\n\n"])])
        self.assertIn("37 tok ·", out)
        self.assertNotIn("~37 tok", out)

    def test_image_user_turn_survives_only_with_partial_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory, "shot.png")
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            for partial in (False, True):
                with self.subTest(partial=partial):
                    chunks = [sse({"content": "PARTIAL"})] if partial else []
                    chunks.append(OSError("connection reset"))
                    out, seen = self.run_chat([
                        StreamResponse(chunks),
                        StreamResponse([sse({"content": "RECOVERED"}),
                                        b"data:[DONE]\n\n"])],
                        prompts=["describe " + str(image), "next"])
                    self.assertIn("connection reset", out)
                    original_user = seen[0][0]
                    self.assertEqual(original_user["content"][0],
                                     {"type": "text", "text": "describe"})
                    url = original_user["content"][1]["image_url"]["url"]
                    self.assertTrue(url.startswith("data:image/png;base64,"))
                    expected = ([original_user,
                                 {"role": "assistant", "content": "PARTIAL"}]
                                if partial else [])
                    self.assertEqual(seen[1], expected + [
                        {"role": "user", "content": "next"}])

    def test_sse_fields_keep_one_optional_space_and_require_delimiters(self):
        response = StreamResponse([
            b":ignored\ndata:\ndata:  leading\ndata\n\n",
            b"data:unterminated\n"])
        self.assertEqual(list(self.cli._sse_data_events(response)),
                         ["\n leading\n"])

    def test_sse_framing_every_byte_boundary(self):
        for ending in ("\n", "\r\n", "\r"):
            wire = (":comment"+ending+"event: message"+ending+
                    'data:{"choices":'+ending+
                    'data:[{"delta":{"content":"café 🦜"}}]}'+ending+ending+
                    "data:[DONE]"+ending+ending).encode()
            for split in range(1, len(wire)):
                with self.subTest(ending=repr(ending), split=split):
                    response = StreamResponse([wire[:split], wire[split:]])
                    self.assertEqual(list(self.cli._sse_data_events(response)),
                        ['{"choices":\n[{"delta":{"content":"café 🦜"}}]}',
                         "[DONE]"])

    def test_real_http_done_on_open_transport(self):
        release = threading.Event()
        sent = threading.Event()

        class OpenSSE(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(sse({"content": "LIVE"}) +
                                 b"data:[DONE]\n\n")
                self.wfile.flush()
                sent.set()
                release.wait(5)
                self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), OpenSSE)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        output = io.StringIO()
        finished = threading.Event()
        failures = []

        def chat():
            try:
                with mock.patch.object(self.cli, "TTY", False), \
                        mock.patch("builtins.input", side_effect=["question", ":q"]), \
                        contextlib.redirect_stdout(output):
                    self.cli.chat_attached(SimpleNamespace(ngen=16, api_key=None),
                        f"http://127.0.0.1:{server.server_address[1]}", "fake")
            except BaseException as error:
                failures.append(error)
            finally:
                finished.set()

        worker = threading.Thread(target=chat)
        worker.start()
        try:
            self.assertTrue(sent.wait(3))
            self.assertTrue(finished.wait(2), "DONE must not wait for HTTP EOF")
            self.assertFalse(failures)
            self.assertIn("LIVE", output.getvalue())
        finally:
            release.set()
            worker.join(6)
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
