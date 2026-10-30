export type ChatRole = "system" | "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  /* Data URIs of the pictures attached to this turn. Kept on the message rather
     than on the draft, because the transcript is resent on every later turn and
     the model has to keep seeing what it was shown. */
  images?: string[]
}

interface OpenAIError {
  error?: { message?: string }
}

export interface SchedulerHealth {
  active: boolean | number
  capacity?: number
  queued: number
  max_queue: number
  queue_timeout_seconds: number
  admitted: number
  completed: number
  rejected: number
  timed_out: number
  cancelled: number
}

export interface TiersHealth {
  vram: number
  ram: number
  disk: number
  vram_gb: number
  ram_gb: number
}

export interface HwinfoHealth {
  cores: number
  ram_total_gb: number
  ram_avail_gb: number
  gpus: number
  vram_total_gb: number
  cpu: string
  gpu: string
}

export interface HealthResponse {
  status: string
  scheduler?: SchedulerHealth
  kv_slots?: number
  tiers?: TiersHealth
  hwinfo?: HwinfoHealth
}

export interface ProfileTurn {
  wall_s: number
  prompt_tokens: number
  completion_tokens: number
  expert_disk_s: number
  expert_wait_s: number
  expert_matmul_s: number
  attention_s: number
  lm_head_s: number
  forwards: number
}

export interface ProfileResponse {
  seq: number
  turns: ProfileTurn[]
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface StreamChatResult {
  finishReason: string | null
  usage: TokenUsage | null
  requestId: string | null
  queueWaitMs: number | null
}

export function endpoint(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`
}

export function serverEndpoint(baseUrl: string, path: string) {
  return endpoint(baseUrl.replace(/\/v1\/?$/, ""), path)
}

function headers(apiKey: string) {
  return {
    "Content-Type": "application/json",
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
  }
}

async function responseError(response: Response) {
  const fallback = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as OpenAIError
    return body.error?.message || fallback
  } catch {
    return fallback
  }
}

export async function listModels(baseUrl: string, apiKey: string, signal?: AbortSignal) {
  const response = await fetch(endpoint(baseUrl, "models"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  const body = (await response.json()) as { data?: Array<{ id: string }> }
  return (body.data || []).map((model) => model.id)
}

export async function getHealth(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "health"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as HealthResponse
}

export async function getProfile(baseUrl: string, apiKey = "", signal?: AbortSignal): Promise<ProfileResponse> {
  const response = await fetch(serverEndpoint(baseUrl, "profile"), { headers: headers(apiKey), signal })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as ProfileResponse
}

export function extractSSE(buffer: string, final = false): { data: string[]; rest: string } {
  const data: string[] = []
  let fields: string[] = []
  let eventStart = 0
  let lineStart = 0
  for (let index = 0; index < buffer.length; index++) {
    const character = buffer[index]
    if (character !== "\r" && character !== "\n") continue
    // A CR ending a blank line already completes the event. A following LF
    // is harmless as an empty event; only an unfinished field must wait.
    if (character === "\r" && index === buffer.length - 1 && index !== lineStart && !final) break
    const line = buffer.slice(lineStart, index)
    if (character === "\r" && buffer[index + 1] === "\n") index++
    lineStart = index + 1
    if (line === "") {
      const payload = fields.join("\n")
      if (payload) data.push(payload)
      fields = []
      eventStart = lineStart
    } else if (line === "data" || line.startsWith("data:")) {
      const value = line.slice(5)
      fields.push(value.startsWith(" ") ? value.slice(1) : value)
    }
  }
  return { data, rest: buffer.slice(eventStart) }
}

export interface StreamChatOptions {
  baseUrl: string
  apiKey: string
  model: string
  messages: ChatMessage[]
  temperature: number
  maxTokens: number
  enableThinking: boolean
  cacheSlot?: number
  signal: AbortSignal
  onDelta: (text: string) => void
}

export async function streamChat(options: StreamChatOptions): Promise<StreamChatResult> {
  const response = await fetch(endpoint(options.baseUrl, "chat/completions"), {
    method: "POST",
    headers: headers(options.apiKey),
    signal: options.signal,
    body: JSON.stringify({
      model: options.model,
      /* A turn with pictures goes out in the content-array form the API takes;
         a plain turn stays a string, so a text-only server sees exactly what it
         saw before this existed. */
      messages: options.messages.map(({ role, content, images }) => images?.length
        ? { role, content: [
            ...(content ? [{ type: "text", text: content }] : []),
            ...images.map(url => ({ type: "image_url", image_url: { url } })),
          ] }
        : { role, content }),
      temperature: options.temperature,
      max_completion_tokens: options.maxTokens,
      enable_thinking: options.enableThinking,
      ...(options.cacheSlot === undefined ? {} : { cache_slot: options.cacheSlot }),
      stream: true,
      stream_options: { include_usage: true },
    }),
  })
  if (!response.ok) throw new Error(await responseError(response))
  if (!response.body) throw new Error("The server returned an empty stream.")

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let finishReason: string | null = null
  let usage: TokenUsage | null = null

  let transportEnded = false
  let cancelRequested = false
  const cancelConsumption = () => {
    if (transportEnded || cancelRequested) return
    cancelRequested = true
    // Some sources never settle cancellation; cleanup must not hold completion open.
    void reader.cancel().catch(() => undefined)
  }
  options.signal.addEventListener("abort", cancelConsumption, { once: true })
  try {
    let completed = false
    while (!completed) {
      options.signal.throwIfAborted()
      const { value, done } = await reader.read()
      transportEnded = done
      options.signal.throwIfAborted()
      buffer += decoder.decode(value, { stream: !done })
      const parsed = extractSSE(buffer, done)
      buffer = parsed.rest
      for (const data of parsed.data) {
        options.signal.throwIfAborted()
        if (data === "[DONE]") {
          completed = true
          break
        }
        let event: {
          error?: { message?: string } | null
          choices?: Array<{ delta?: { content?: string; tool_calls?: unknown }; finish_reason?: string | null }>
          usage?: TokenUsage | null
        }
        try {
          event = JSON.parse(data) as typeof event
          if (!event || typeof event !== "object" || Array.isArray(event)) throw new Error()
        } catch {
          throw new Error("The server returned an invalid streaming event.")
        }
        if ("error" in event) throw new Error(event.error?.message || "The server reported a streaming error.")
        const choice = event.choices?.[0]
        if (choice?.delta?.tool_calls) throw new Error("This text chat does not support tool calls.")
        const text = choice?.delta?.content
        if (text) {
          options.signal.throwIfAborted()
          options.onDelta(text)
        }
        if (choice?.finish_reason) finishReason = choice.finish_reason
        if (event.usage) usage = event.usage
      }
      if (done && !completed) throw new Error("The response stream ended before [DONE].")
    }
    options.signal.throwIfAborted()
    const queueWaitHeader = response.headers.get("x-colibri-queue-wait-ms")
    const parsedQueueWait = queueWaitHeader === null ? null : Number(queueWaitHeader)
    return {
      finishReason,
      usage,
      requestId: response.headers.get("x-request-id"),
      queueWaitMs: parsedQueueWait !== null && Number.isFinite(parsedQueueWait) ? parsedQueueWait : null,
    }
  } finally {
    options.signal.removeEventListener("abort", cancelConsumption)
    cancelConsumption()
    reader.releaseLock()
  }
}

/* Modalita brio: il modello non genera, assegna una probabilita a ogni opzione
 * ammessa. Il ciclo (fotografia del prefisso condiviso, una lettura per
 * opzione, normalizzazione per lunghezza) sta nel gateway: qui si manda una
 * richiesta e si riceve una distribuzione. */
export interface BrioChoice {
  option: string
  p: number
  logprob: number
  mean_logprob: number
  tokens: number
}

export interface BrioResponse {
  answer: string
  entropy: number
  normalize: "mean" | "sum"
  choices: BrioChoice[]
  usage: { prompt_tokens: number; completion_tokens: number; read_tokens: number; total_tokens: number }
}

export async function askBrio(
  baseUrl: string,
  apiKey: string,
  model: string,
  state: string,
  question: string,
  options: string[],
  signal?: AbortSignal,
): Promise<BrioResponse> {
  const response = await fetch(endpoint(baseUrl, "brio"), {
    method: "POST",
    headers: headers(apiKey),
    body: JSON.stringify({ model, state, question, options }),
    signal,
  })
  if (!response.ok) throw new Error(await responseError(response))
  return (await response.json()) as BrioResponse
}
