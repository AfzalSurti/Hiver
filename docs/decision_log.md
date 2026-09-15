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

---

### 4. Intent taxonomy: 9 intents, with a deliberately fine-grained financial cluster

**Decision:** Use 9 intents rather than collapsing to the suggested 6-8,
specifically keeping `PAYMENT_BILLING_ISSUE`, `REFUND_REQUEST`, and
`SUBSCRIPTION_PLAN_MANAGEMENT` as three separate intents instead of one
"billing" bucket.

**Why:** These three have materially different escalation implications -
`REFUND_REQUEST` escalates almost unconditionally (money moving), a plain
`SUBSCRIPTION_PLAN_MANAGEMENT` question usually auto-handles, and
`PAYMENT_BILLING_ISSUE` sits in between depending on whether it's a generic
mechanical failure or a specific-charge dispute. Collapsing them would force
the escalation policy to make this same distinction *within* one intent
using extra logic anyway, so keeping them separate makes the escalation
policy simpler and the confusion matrix more diagnostic. See
`docs/intent_taxonomy.md` "Notes on overlap."

**Trade-off:** More intents means smaller per-class support in golden-set
evaluation (some classes end up thin, e.g. `POSITIVE_FEEDBACK_OR_RESOLVED`
at 4/228), which weakens per-class precision/recall reliability for the
smallest classes.

---

### 5. Escalation ground truth is judged from message content, not from mimicking the brand's actual historical replies

**Decision:** Golden-set `expected_action` labels reflect what a careful
human support lead would decide from the customer's message alone, not what
SpotifyCares' real agents historically did in that thread.

**Why:** During annotation it became clear that the overwhelming majority of
real SpotifyCares replies are a boilerplate "please DM us your account
email" *regardless of issue severity* - it's their standard
privacy/verification practice, not a severity signal. Treating "the brand
asked for a DM" as ground truth for "this needed escalation" would have
produced a policy that escalates almost everything and taught nothing about
genuine risk differentiation. See
`data/golden/ANNOTATION_METHODOLOGY.md` for the full writeup - this was one
of the most consequential things learned while building the golden set.

**Trade-off:** This makes the ground truth a normative judgment call (what
*should* happen) rather than a descriptive one (what *did* happen), which is
inherently harder to audit and is exactly why every escalation label carries
a required, human-readable justification in `escalation_reason` + `notes`.

---

### 6. Golden set is intent-stratified, not traffic-proportional

**Decision:** Golden-set sampling targets up to ~20 examples per intent
(plus deliberate short/ambiguous/multi-signal buckets) rather than sampling
i.i.d. from the eval pool, which would be dominated by whichever 2-3 intents
are most common in real traffic.

**Why:** An i.i.d. sample large enough to get reliable per-class metrics on
the rarest real intent (`CONTENT_CATALOG_AVAILABILITY`, ~0.6% of weak-labeled
train traffic) would need to be enormous. Stratifying trades "reflects true
traffic mix" for "every intent has enough examples to compute a meaningful
F1," which is the right trade for a diagnostic evaluation set.

**Trade-off:** This is exactly the kind of thing that makes a single
headline accuracy/F1 number misleading if reported without context - see
`REPORT.md`'s "What is misleading about my headline number?" section. A
production deployment would see a very different (much more skewed) intent
mix than the golden set represents.

---

### 7. Golden-set labels are single-annotator, explicitly not called "human-labelled" in the crowd-sourced sense

**Decision:** All 228 golden-set labels were produced by one annotator (the
project author) against a written rubric, and this is disclosed prominently
rather than presented as independent human ground truth.

**Why:** The project brief explicitly warns against generating fake human
labels and calling them human-labelled. A single, disclosed annotator
following a public, versioned rubric (`docs/intent_taxonomy.md`,
`docs/escalation_policy.md`) is honest about what it is; claiming
crowd-sourced or panel-reviewed labels would not have been.

**Trade-off:** No inter-annotator agreement statistic exists for the golden
set itself, and the Phase 12 LLM-judge-vs-human study is therefore
judge-vs-single-annotator, a weaker agreement claim than judge-vs-panel.
Documented as a limitation in both `ANNOTATION_METHODOLOGY.md` and
`REPORT.md`.

---

### 8. Weak/heuristic labels are used for two narrow purposes only, never as ground truth

**Decision:** The regex-based `weak_label()` function
(`src/support_agent/weak_labels.py`) is used only to (a) generate silver
training labels for the TF-IDF baseline on the train split, and (b)
stratify which eval-split conversations become golden-set *candidates*.  It
never appears as a ground-truth label anywhere.

**Why:** Hand-labelling the full ~10.5k-example train split was out of
scope for a take-home; a keyword heuristic is a transparent, auditable stand-in.
Its accuracy was spot-checked during golden-set annotation: of ~20
candidates it tagged `POSITIVE_FEEDBACK_OR_RESOLVED` for sampling purposes,
only 4 survived manual review as actually correct - a concrete, measured
illustration of the heuristic's limits (and hence a bound on the TF-IDF
baseline's ceiling, discussed in `REPORT.md`).

**Trade-off:** The TF-IDF baseline's training data is noisy by construction;
its numbers should be read as "TF-IDF given cheap heuristic labels," not
"TF-IDF's true ceiling given clean labels."
