import { afterEach, describe, expect, it, vi } from "vitest"

import { askBrio, extractSSE, getHealth, getProfile, serverEndpoint, streamChat } from "./api"

afterEach(() => vi.unstubAllGlobals())

describe("extractSSE", () => {
  it("keeps an incomplete frame for the next network chunk", () => {
    const parsed = extractSSE('data: {"choices":[]}\n\ndata: {"cho')
    expect(parsed.data).toEqual(['{"choices":[]}'])
    expect(parsed.rest).toBe('data: {"cho')
  })

  it("supports CRLF and multiple data frames", () => {
    const parsed = extractSSE("data: one\r\n\r\ndata: two\r\n\r\n")
    expect(parsed.data).toEqual(["one", "two"])
    expect(parsed.rest).toBe("")
  })
})

describe("runtime API", () => {
  it.each([
    ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/health"],
    ["https://example.test/api/v1/", "https://example.test/api/health"],
    ["https://example.test/api", "https://example.test/api/health"],
  ])("resolves the health endpoint outside the OpenAI v1 prefix", (baseUrl, expected) => {
    expect(serverEndpoint(baseUrl, "health")).toBe(expected)
  })

  it("requests health with the configured bearer credential", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "ok", scheduler: { active: true } })))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getHealth("http://localhost:8000/v1/", "secret")).resolves.toMatchObject({ status: "ok" })
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/health", expect.objectContaining({
      headers: expect.objectContaining({ Authorization: "Bearer secret" }),
    }))
  })

  it("requests the profiling history next to the OpenAI v1 prefix", async () => {
    const turn = {
      wall_s: 2.5, prompt_tokens: 7, completion_tokens: 12,
      expert_disk_s: 0.4, expert_wait_s: 0.1, expert_matmul_s: 0.9,
      attention_s: 0.6, lm_head_s: 0.2, forwards: 15,
    }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ seq: 1, turns: [turn] })))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getProfile("http://localhost:8000/v1/")).resolves.toEqual({ seq: 1, turns: [turn] })
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/profile", expect.anything())
  })
})

describe("chat request extensions", () => {
  const completedStream = () => new Response("data: [DONE]\n\n", {
    headers: { "content-type": "text/event-stream" },
  })

  async function requestBody(cacheSlot?: number) {
    const fetchMock = vi.fn().mockResolvedValue(completedStream())
    vi.stubGlobal("fetch", fetchMock)
    await streamChat({
      baseUrl: "http://localhost:8000/v1",
      apiKey: "",
      model: "test-model",
      messages: [],
      temperature: 0,
      maxTokens: 8,
      enableThinking: false,
      cacheSlot,
      signal: new AbortController().signal,
      onDelta: () => undefined,
    })
    return JSON.parse(fetchMock.mock.calls[0][1].body as string) as Record<string, unknown>
  }

  it("omits cache_slot for a generic OpenAI-compatible backend", async () => {
    expect(await requestBody()).not.toHaveProperty("cache_slot")
  })

  it("sends cache_slot zero when colibrì advertises KV slots", async () => {
    expect(await requestBody(0)).toMatchObject({ cache_slot: 0 })
  })

  it("preserves image turns and text-only history serialization", async () => {
    const fetchMock = vi.fn().mockResolvedValue(completedStream())
    vi.stubGlobal("fetch", fetchMock)
    const image = "data:image/png;base64,aW1hZ2U="
    await streamChat({
      baseUrl: "http://localhost/v1", apiKey: "", model: "vision-test",
      messages: [
        { id: "1", role: "user", content: "describe", images: [image] },
        { id: "2", role: "assistant", content: "a picture" },
        { id: "3", role: "user", content: "", images: [image, image] },
        { id: "4", role: "user", content: "continue", images: [] },
      ],
      temperature: 0, maxTokens: 8, enableThinking: false,
      signal: new AbortController().signal, onDelta: () => undefined,
    })
    expect(JSON.parse(fetchMock.mock.calls[0][1].body as string).messages).toEqual([
      { role: "user", content: [{ type: "text", text: "describe" }, { type: "image_url", image_url: { url: image } }] },
      { role: "assistant", content: "a picture" },
      { role: "user", content: [{ type: "image_url", image_url: { url: image } }, { type: "image_url", image_url: { url: image } }] },
      { role: "user", content: "continue" },
    ])
  })
})

describe("askBrio", () => {
  /* The options must travel as a field, never folded into the prompt: keeping
     them out of the text is half of what the mode saves, and a refactor that
     "helpfully" appended them would be invisible in the answer. */
  it("sends the options as data and posts to the brio endpoint", async () => {
    const seen: { url?: string; body?: unknown } = {}
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.url = url
      seen.body = JSON.parse(String(init.body))
      return new Response(JSON.stringify({
        answer: "b", entropy: 0.5, normalize: "mean",
        choices: [{ option: "b", p: 0.7, logprob: -1, mean_logprob: -1, tokens: 1 }],
        usage: { prompt_tokens: 9, completion_tokens: 0, read_tokens: 2, total_tokens: 11 },
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    }))
    const out = await askBrio("http://x/v1", "", "m", "state", "q?", ["a", "b"])
    expect(seen.url).toBe("http://x/v1/brio")
    expect(seen.body).toMatchObject({ model: "m", state: "state", question: "q?", options: ["a", "b"] })
    expect(out.answer).toBe("b")
    expect(out.usage.completion_tokens).toBe(0)
  })

  it("surfaces the server's own message instead of a bare status", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ error: { message: "options must be a non-empty array" } }),
      { status: 400, headers: { "Content-Type": "application/json" } })))
    await expect(askBrio("http://x/v1", "", "m", "s", "q", ["a"]))
      .rejects.toThrow("options must be a non-empty array")
  })
})

describe("streaming transactions", () => {
  const frame = (value: unknown) => `data:${JSON.stringify(value)}\n\n`
  const textFrame = (content: string) => frame({ choices: [{ delta: { content } }] })
  const bytes = (text: string) => new TextEncoder().encode(text)

  function transaction(chunks: Uint8Array[], close = true, controller = new AbortController()) {
    const cancel = vi.fn()
    const body = new ReadableStream<Uint8Array>({
      start(source) {
        for (const chunk of chunks) source.enqueue(chunk)
        if (close) source.close()
      },
      cancel,
    })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, {
      headers: { "x-request-id": "req-1", "x-colibri-queue-wait-ms": "2.5" },
    })))
    const onDelta = vi.fn()
    const options = {
      baseUrl: "http://localhost:8000/v1", apiKey: "", model: "test", messages: [],
      temperature: 0, maxTokens: 8, enableThinking: false, signal: controller.signal, onDelta,
    }
    return { options, onDelta, cancel, body }
  }

  it("assembles multiline fields, comments, and only one optional space", () => {
    expect(extractSSE(": ping\nevent: message\ndata: one\ndata:  two\n\n").data).toEqual(["one\n two"])
    expect(extractSSE("data:\n\n: comment\n\n").data).toEqual([])
  })

  it.each(["\n", "\r\n", "\r"])("handles every split of %j event delimiters", (ending) => {
    const wire = `: ping${ending}data:one${ending}data: two${ending}${ending}data:three${ending}${ending}`
    for (let split = 0; split <= wire.length; split++) {
      const first = extractSSE(wire.slice(0, split))
      const second = extractSSE(first.rest + wire.slice(split), true)
      expect([...first.data, ...second.data]).toEqual(["one\ntwo", "three"])
      expect(second.rest).toBe("")
    }
    expect(extractSSE("data:unfinished\r", true).data).toEqual([])
  })

  it("preserves valid Unicode across every byte boundary and usage after finish", async () => {
    const usage = { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }
    const wire = bytes(textFrame("Paris 🗼 café") + frame({ choices: [{ finish_reason: "stop" }] })
      + frame({ choices: [], usage }) + "data:[DONE]\n\n")
    for (let split = 0; split <= wire.length; split++) {
      const tx = transaction([wire.slice(0, split), wire.slice(split)])
      await expect(streamChat(tx.options)).resolves.toEqual({
        finishReason: "stop", usage, requestId: "req-1", queueWaitMs: 2.5,
      })
      expect(tx.onDelta.mock.calls).toEqual([["Paris 🗼 café"]])
      expect(tx.body.locked).toBe(false)
    }
  })

  it.each(["\n", "\r\n", "\r"])("streams multiline JSON across every %j boundary", async (ending) => {
    const wire = bytes([
      ": keepalive", "event: message", 'data:{"choices":',
      'data:[{"delta":{"reasoning_content":"thought","content":"Paris"}}]}', "",
      "data:[DONE]", "", "",
    ].join(ending))
    for (let split = 0; split <= wire.length; split++) {
      const tx = transaction([wire.slice(0, split), wire.slice(split)])
      await expect(streamChat(tx.options)).resolves.toMatchObject({ finishReason: null })
      expect(tx.onDelta.mock.calls).toEqual([["Paris"]])
      expect(tx.body.locked).toBe(false)
    }
  })

  it("finishes and cancels an open transport at DONE, ignoring later frames", async () => {
    const tx = transaction([bytes(textFrame("Paris") + "data:[DONE]\n\n"
      + textFrame("ignored") + frame({ error: { message: "ignored" } }))], false)
    await streamChat(tx.options)
    expect(tx.onDelta.mock.calls).toEqual([["Paris"]])
    expect(tx.cancel).toHaveBeenCalledTimes(1)
    expect(tx.body.locked).toBe(false)
  }, 1000)

  it.each(["\n", "\r\n", "\r"])("finishes an open transport at a trailing %j DONE boundary", async (ending) => {
    const wire = bytes(textFrame("Paris").replaceAll("\n", ending)
      + `data:[DONE]${ending}${ending}`)
    for (let split = 0; split <= wire.length; split++) {
      const tx = transaction([wire.slice(0, split), wire.slice(split)], false)
      await streamChat(tx.options)
      expect(tx.onDelta.mock.calls).toEqual([["Paris"]])
      expect(tx.cancel).toHaveBeenCalledTimes(1)
      expect(tx.body.locked).toBe(false)
    }
  }, 1000)

  it.each([
    ["", "The response stream ended before [DONE]."],
    [textFrame("partial"), "The response stream ended before [DONE]."],
    ["data:[DONE]\n", "The response stream ended before [DONE]."],
    ["data:bad JSON\n\n", "The server returned an invalid streaming event."],
    [frame(null), "The server returned an invalid streaming event."],
    [frame({ error: { message: "busy" } }), "busy"],
    [textFrame("partial") + frame({ error: {} }), "The server reported a streaming error."],
    [frame({ choices: [{ delta: { tool_calls: [{ id: "call-1" }] } }] }), "This text chat does not support tool calls."],
  ])("rejects invalid transaction %j", async (wire, message) => {
    const tx = transaction([bytes(wire)])
    await expect(streamChat(tx.options)).rejects.toThrow(message)
    expect(tx.body.locked).toBe(false)
    if (wire.startsWith(textFrame("partial"))) expect(tx.onDelta.mock.calls).toEqual([["partial"]])
  })

  it("cancels and unlocks on an in-band error without waiting for EOF", async () => {
    const tx = transaction([bytes(frame({ error: { message: "failed" } }))], false)
    await expect(streamChat(tx.options)).rejects.toThrow("failed")
    expect(tx.cancel).toHaveBeenCalledTimes(1)
    expect(tx.body.locked).toBe(false)
  }, 1000)

  it("does not deliver buffered content or success after abort", async () => {
    const controller = new AbortController()
    const tx = transaction([bytes(textFrame("first") + textFrame("later") + "data:[DONE]\n\n")], false, controller)
    tx.onDelta.mockImplementation(() => controller.abort())
    await expect(streamChat(tx.options)).rejects.toMatchObject({ name: "AbortError" })
    expect(tx.onDelta.mock.calls).toEqual([["first"]])
    expect(tx.cancel).toHaveBeenCalledTimes(1)
    expect(tx.body.locked).toBe(false)
  }, 1000)

  it("rejects an already aborted response before delivering content", async () => {
    const controller = new AbortController()
    controller.abort()
    const tx = transaction([bytes(textFrame("later") + "data:[DONE]\n\n")], false, controller)
    await expect(streamChat(tx.options)).rejects.toMatchObject({ name: "AbortError" })
    expect(tx.onDelta).not.toHaveBeenCalled()
    expect(tx.body.locked).toBe(false)
  }, 1000)

  it("aborts while a read is pending and releases ownership", async () => {
    // ES2022 (the frontend's library target) does not expose Promise.withResolvers.
    let markReading!: () => void
    const reading = new Promise<void>((resolve) => { markReading = resolve })
    const cancel = vi.fn()
    const body = new ReadableStream<Uint8Array>({
      pull() { markReading() },
      cancel,
    })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)))
    const controller = new AbortController()
    const onDelta = vi.fn()
    const result = streamChat({
      baseUrl: "http://localhost/v1", apiKey: "", model: "test", messages: [],
      temperature: 0, maxTokens: 8, enableThinking: false, signal: controller.signal, onDelta,
    })
    await reading
    controller.abort()
    await expect(result).rejects.toMatchObject({ name: "AbortError" })
    expect(onDelta).not.toHaveBeenCalled()
    expect(cancel).toHaveBeenCalledTimes(1)
    expect(body.locked).toBe(false)
  }, 1000)

  it("does not replace a callback failure with a cancellation failure", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(source) { source.enqueue(bytes(textFrame("first"))) },
      cancel() { throw new Error("cleanup failed") },
    })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)))
    await expect(streamChat({
      baseUrl: "http://localhost/v1", apiKey: "", model: "test", messages: [],
      temperature: 0, maxTokens: 8, enableThinking: false, signal: new AbortController().signal,
      onDelta() { throw new Error("consumer failed") },
    })).rejects.toThrow("consumer failed")
    expect(body.locked).toBe(false)
  })

  it("does not wait for a source's cancellation promise after DONE", async () => {
    const cancel = vi.fn(() => new Promise<void>(() => undefined))
    const body = new ReadableStream<Uint8Array>({
      start(source) { source.enqueue(bytes("data:[DONE]\n\n")) },
      cancel,
    })
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body)))
    await expect(streamChat({
      baseUrl: "http://localhost/v1", apiKey: "", model: "test", messages: [],
      temperature: 0, maxTokens: 8, enableThinking: false, signal: new AbortController().signal,
      onDelta: () => undefined,
    })).resolves.toMatchObject({ usage: null })
    expect(cancel).toHaveBeenCalledTimes(1)
    expect(body.locked).toBe(false)
  }, 1000)

  it("accepts reasoning-only or empty completion without inventing visible content", async () => {
    const tx = transaction([bytes(frame({ choices: [{ delta: { reasoning_content: "thought" } }] }) + "data:[DONE]\n\n")])
    await expect(streamChat(tx.options)).resolves.toMatchObject({ usage: null })
    expect(tx.onDelta).not.toHaveBeenCalled()
  })
})
