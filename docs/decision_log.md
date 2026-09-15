# Decision Log

Non-obvious decisions made while building this project, and why. Entries are
numbered in the order they were made, not necessarily the order they appear
in the final pipeline.

---

### 1. Brand selection: SpotifyCares

**Decision:** Build the agent for `SpotifyCares` rather than the higher-volume
`AmazonHelp` (169,840 agent messages) or `AppleSupport` (106,860).

**Why:** Spotify is a single product/service, so its recurring customer
issues cluster into a small, coherent intent set (login/account, billing,
subscription tier, playback bugs, ads, device connectivity, content
availability). Amazon spans every retail category it sells (electronics,
groceries, deliveries, returns, digital content, Prime) and Apple spans many
hardware lines (iPhone, Mac, Watch, iCloud) plus software — both would need
either a much larger intent taxonomy or a taxonomy so coarse it stops being
useful for a demo of this scope. SpotifyCares also had a 99.8% in-thread
brand-reply rate on sampled root conversations and 43,265 agent messages /
13,476+ sampled root threads, comfortably enough to build train/golden splits
without touching the full 2.8M-row dataset. See `reports/eda/eda_summary.md`
and `scripts/02_select_brand.py` for the underlying evidence.

**Trade-off:** Spotify's data is skewed toward a handful of chronic complaints
(offline downloads breaking, ads, "why is my playlist gone") so the long tail
of rare intents is thinner than it would be for a broader retailer — the
golden set deliberately oversamples minority intents to compensate (see
decision log entries on golden-set sampling).

---

### 2. Dataset sourced via a verified Hugging Face mirror, not the Kaggle API

**Decision:** Download `twcs.csv` from the `SunidhiSriram/twcs` Hugging Face
dataset repo instead of the Kaggle API.

**Why:** The Kaggle API requires a per-user API token (`kaggle.json`), which
was not available in this environment and downloading it would have required
pausing to get credentials from the user. Before using the mirror, its schema
was byte-verified against the known Kaggle schema (`tweet_id, author_id,
inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`) and
its row count (2,811,774) matches the publicly documented size of the
original Kaggle "Customer Support on Twitter" dataset, giving reasonable
confidence it's a faithful, unmodified copy rather than a derived/cleaned
variant.

**Trade-off:** We cannot cryptographically prove this mirror is byte-identical
to Kaggle's canonical copy. The README documents the exact source URL and how
to instead fetch it via the official Kaggle API if a reviewer wants to
verify against the canonical copy directly.

---

### 3. LLM access via OpenRouter as a single OpenAI-compatible gateway

**Decision:** Call all LLM components (intent classifier, reply generator,
judge) through OpenRouter using the `openai` Python SDK with a custom
`base_url`, rather than calling Anthropic/OpenAI SDKs directly.

**Why:** OpenRouter exposes an OpenAI-compatible `/chat/completions` API in
front of many providers/models, so one SDK dependency (already installed:
`openai`) and one API key can reach any model by name (e.g.
`openai/gpt-4o-mini`, `anthropic/claude-3.5-sonnet`). This also makes it
trivial to use a *different* model for the judge than for generation, which
reduces self-preference bias in the LLM-as-judge evaluation (Phase 11/12).

**Trade-off:** Adds a third-party proxy in the request path (latency,
another point of failure, OpenRouter's own rate limits) instead of talking to
providers directly.
