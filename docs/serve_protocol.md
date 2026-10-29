# The serve protocols — engine ⇄ server wire format

The engine speaks two line-oriented protocols over stdin/stdout. Both are plain text
plus byte-counted payload frames; every outbound line is written with a trailing
`fflush`, one line per write. On Windows both ends of the pipe are switched to binary
mode at startup — the CRT's CRLF translation otherwise corrupts the sentinels and
stalls byte-counted reads (#195).

| protocol | entry | selected by | used by |
|---|---|---|---|
| **mux** (continuous batching, up to 16 KV slots) | `run_serve_mux` | `SERVE_BATCH=1` | `openai_server.py`, `coli web` |
| **legacy** (single slot, interactive) | `run_serve` | `SERVE=1` (without `SERVE_BATCH`) | `coli chat` |

This document is the reference for the **mux** protocol; the legacy protocol is
summarized at the end. Line formats below are quoted from the emitting `printf`s in
`glm.c` — if this document and the code disagree, the code wins and this file needs a PR.

## Startup handshake (engine → server)

```
\x01\x01READY\x01\x01
STAT 0 0.00 0.0 <rss_gb>
HWINFO <cores> <ram_total_gb> <ram_avail_gb> <ngpu> <vram_total_gb> <cpu_name>|<gpu_name>
TIERS <vram_experts> <ram_experts> <disk_experts> <vram_gb> <ram_gb>
EMAP <rows> <cols> <hex>
```

The server must not send requests before `READY`. `HWINFO`/`TIERS`/`EMAP` are
telemetry (see below) and may grow — **servers must ignore line kinds they do not
recognize**; that is the protocol's forward-compatibility rule.

## Requests (server → engine)

```
SUBMIT <id> <slot> <bytes> <max_tokens> <temperature> <top_p>\n<payload>\n
STOP <id>\n
CANCEL <id>\n
```

- `id` — non-zero u64, unique among in-flight requests.
- `slot` — KV slot index, `0 … KV_SLOTS-1` (`KV_SLOTS` env, 1–16, default 1). A slot
  holds one conversation's KV; the engine matches the tokenized payload against the
  slot's history and reuses the common prefix (truncate-and-extend), so stateless
  HTTP turns keep their cache.
- `bytes` — exact byte length of `payload` (UTF-8, may contain newlines). The engine
  reads exactly that many bytes after the header line, then one trailing `\n`.
- `payload` — the fully rendered prompt (the server owns the chat template).
- `STOP` ends generation through the normal successful `DONE` path. Statistics,
  usage history, and KV state are persisted; the HTTP gateway uses it after a
  client-provided stop sequence matches.
- `CANCEL` aborts a request after its client disconnects and returns `CANCELLED`.
- EOF on stdin = graceful shutdown: in-flight requests finish first.

Prefill is serial; decode is continuously batched — every active slot contributes
one row per forward.

## Responses (engine → server)

Per request, in order:

```
DATA <id> <n>\n<n bytes of UTF-8>\n        # a decoded token's text; repeated
TOPK <id> 5 <logprob> <hextext> ... ×5     # candidates for the sampled token (SERVE_TOPK=1)
HITS <rows> <cols> <hex>                   # ~every 6 tokens: routed-expert bitmap since last HITS
REPIN <layer> <eid> <old_tier> <gpu>       # live re-pin swap events, as they happen
...
DONE <id> STAT <emitted> <tok_s> <hit_pct> <rss_gb> <prompt_tokens> <length_limited>
```

Errors replace the stream: `ERROR <id> <CODE>` with codes `BAD_FRAME`, `BAD_REQUEST`,
`SLOT_BUSY`, `DUPLICATE_ID`, `EMPTY_PROMPT`, `NOT_FOUND` (CANCEL of unknown id),
`CANCELLED`. A `CANCEL` is acknowledged by `ERROR <id> CANCELLED` after the slot's KV
is persisted.

The serial OLMoE, Inkling, Kimi and DeepSeek V4 servers use the same
request-ID control contract. They emit `ACCEPT <id> <prompt_tokens>` after
validation and before prefill. Context rejection uses
`ERROR <id> CONTEXT_EXCEEDED <used> <limit>`. At existing forward/callback
boundaries, readable controls are drained before more DATA; CANCEL wins over STOP.
STOP finishes with normal DONE and `length_limited=0`; CANCEL emits only its
CANCELLED terminal error. EOF closes admission but finishes accepted work.
OLMoE/Inkling retain FIFO submissions: queued STOP yields an accepted zero-output
completion when its turn arrives; queued CANCEL removes only that request.

SUBMIT framing is distinct from semantic admission. A fully framed rejected
request has its payload and bounded optional extension consumed before rejection,
so prompt bytes cannot become control commands. Invalid lengths, incomplete or
overlong headers, truncated payloads and bad final delimiters are fatal BAD_FRAME:
the stream is not resynchronized. Inkling retains its optional audio extension
and 64 MiB extension ceiling.

Immediately before each `DONE` the engine emits a telemetry block for the finished
turn: `HWINFO`, `PERF`, `ENTROPY`, `GPUS`, `TIERS`, `EMAP`, `HITS` (formats below).
`.coli_usage` is persisted at every turn end, not only at exit.

## Telemetry lines

| line | format | meaning |
|---|---|---|
| `TIERS` | `TIERS <vram> <ram> <disk> <vram_gb> <ram_gb>` | expert count per tier + resident bytes |
| `HWINFO` | `HWINFO <cores> <ram_total> <ram_avail> <ngpu> <vram_total> <cpu>\|<gpu>` | host snapshot (GBs are floats) |
| `EMAP` | `EMAP <rows> <cols> <hex>` | one byte per expert, row-major over `rows×cols` (sparse layers +MTP × experts): `byte = (tier<<6) \| heat` — 2-bit tier (0 disk / 1 RAM / 2 VRAM), 6-bit log₂-bucketed usage heat |
| `HITS` | `HITS <rows> <cols> <hex>` | 1 bit per expert, experts routed since the previous `HITS` |
| `PERF` | `PERF <id> <dt> <t_edisk> <t_ewait> <t_emm> <t_attn> <t_kvb> <t_head>` | this turn's PROFILO deltas, seconds |
| `ENTROPY` | `ENTROPY <h0> <h1> …` | per-sparse-layer routing entropy of the turn, bits |
| `GPUS` | `GPUS <n> (<used_gb> <total_gb> <experts>)×n` | per-device VRAM + resident expert count (CUDA builds) |
| `TOPK` | `TOPK <id> 5 (<logprob> <hextext>)×5` | token text hex-encoded so the line stays line-shaped |
| `REPIN` | `REPIN <layer> <eid> <old_tier> <gpu>` | one line per hot-store swap (`REPIN=n` mode) |

All telemetry is advisory: servers render what they know and skip the rest.

## HTTP surface (`openai_server.py`)

- `POST /v1/chat/completions` — OpenAI-compatible; streaming responses emit one extra
  SSE frame `data: {"colibri": {stats, perf, topk, entropy, gpus, repin}}` immediately
  before `data: [DONE]`; non-streaming responses attach the same object as a
  `"colibri"` field.
- `GET /experts` — the latest `EMAP`/`HITS` state: `{rows, cols, map, hits, seq,
  gpus, entropy, repin}`.
- `GET /*` — static hosting of `web/dist` (SPA fallback, path-traversal-safe), plus
  `experts.json` if published there (the measured expert atlas, #175/#218).

### Streaming lifecycle and built-in consumers

OpenAI chat/completions and Anthropic messages defer SSE success headers until
native acceptance. Older engines without ACCEPT are accepted on their first
DATA/DONE. A live prefill is not cancellation; disconnected peers request native
CANCEL, and the transaction remains owned until its terminal acknowledgement.
Callback failures cancel and drain too, without replacing the original failure.

Before acceptance, errors remain HTTP JSON errors. After acceptance, OpenAI
emits a data event containing an `error` envelope; Anthropic emits a named `error`
event. Neither error path emits a success terminator or a second HTTP status.
Keepalive pumps stop and join on success, failure and disconnect. Anthropic block
indices increase across reopened thinking, text and tool-use blocks; input usage
comes from ACCEPT or, for older engines, authoritative terminal statistics.

The browser and attached terminal assemble SSE data fields per complete event,
including multiline/no-space fields, comments and LF/CRLF/CR line endings. They
require `[DONE]`, stop immediately at that marker even on an open connection,
and explicitly report server errors, invalid JSON, premature EOF and unsupported
tool calls. Partial answer text survives interruption; reasoning is not added to
terminal assistant history. Empty successful replies produce a no-answer
diagnostic rather than a permanent typing indicator.

Opt-in real-model acceptance attaches only to an explicitly supplied gateway:

```sh
cd c
COLI_STREAM_BASE_URL=http://127.0.0.1:18773 \
COLI_STREAM_MODEL_ID=olmoe-colibri \
python3.13 -m unittest -v tests.test_streaming_model
```

Set `COLI_API_KEY` in the environment when authentication is enabled. The suite
expects an owner-configured `CTX=128` and at least a 64-token generation ceiling.
It compares JSON/SSE answers and usage on all three endpoints, checks pre-accept
context rejection, and disconnects mid-generation before checking recovery. It
does not start/stop a gateway or download/change weights. Normal discovery skips
when both streaming variables are unset; setting only one is an error.


## Legacy protocol (`run_serve`, `coli chat`)

Interactive lines are prompts; control frames: `\x02RESET` (clear history),
`\x02MORE` (continue an NGEN-truncated answer), and
`\x02PROMPT <bytes> <max_tokens> <temperature> <top_p> [kv_slot]\n<prompt>\n`
(the pre-mux API mode). Responses are raw text terminated by `\x01\x01END\x01\x01`
plus a `STAT` line. New integrations should use the mux protocol.
