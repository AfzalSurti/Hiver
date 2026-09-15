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

---

### 9. Retrieval uses TF-IDF cosine similarity, not neural sentence embeddings

**Decision:** `src/support_agent/retrieval.py` retrieves historical
resolutions via TF-IDF vectors + cosine similarity, not
sentence-transformers embeddings, despite the brief's "preferred" approach
being embeddings + vector similarity.

**Why:** Importing `sentence_transformers` in this environment crashed the
Python process outright with a Windows fastfail (`0xC0000409`, a
heap-corruption abort) - confirmed by bisecting imports (`torch`,
`transformers`, `tokenizers`, `huggingface_hub` all import fine
individually; the crash only occurs once `sentence_transformers` itself is
imported), most likely a numpy 2.x ABI mismatch with an older compiled
dependency several layers down. Fixing this would mean upgrading/downgrading
packages in the user's shared conda environment, risking breakage in
unrelated projects that also depend on it - too large a blast radius for a
retrieval-quality improvement. TF-IDF vectors are themselves a form of
(sparse) embedding, and cosine similarity over them is a standard,
well-understood retrieval baseline, already proven stable in this
environment via the Phase 6 classifier.

**Trade-off:** TF-IDF is purely lexical (keyword-overlap) and misses
semantic paraphrases a neural embedding would catch (e.g. "I can't sign in"
vs. "login isn't working" share no distinctive n-grams). This is a real
retrieval-quality ceiling, called out again in `REPORT.md`'s failure
analysis and "one more week" section as the highest-leverage fix.

---

### 10. Escalation policy signals and thresholds are explicit and inspectable, not learned

**Decision:** `src/support_agent/escalation.py` implements escalation as a
small set of explicit, ordered rules (risk keywords > always-escalate
intents > confidence threshold > retrieval-evidence threshold) rather than a
learned classifier trained on the golden set's `expected_action` labels.

**Why:** The brief asks for the escalation policy to be "inspectable and
explainable." A learned model over 228 examples would also almost certainly
overfit, and - more importantly - would offer no way to state *why* a given
message escalated beyond "the model said so," which defeats the purpose of
an escalation reason. Explicit rules mean every decision traces to exactly
one named signal (see the `signals` dict returned by `decide()`).

**Trade-off:** Thresholds (confidence floor, minimum retrieval similarity)
are set from reasoning about the problem and the TF-IDF similarity scale,
not fit to data, since the real classifier's confidence distribution isn't
known until Phase 7 runs against a live LLM. They are explicitly flagged in
the module docstring as pending calibration, not claimed as optimal.

---

### 11. LLM calls distinguish "missing configuration" from "bad model output" - only the latter gets a safe fallback

**Decision:** `classify()` and `generate_reply()` catch model-output
failures (invalid JSON, failed validation after retries) and return a safe
fallback value, but explicitly re-raise `LLMConfigError` (e.g. a missing
`OPENROUTER_API_KEY`) rather than swallowing it into the same fallback path.

**Why:** This is a bug found and fixed during development, not a
design decision made up front - worth recording because of what it reveals.
The first version caught every exception uniformly. Running
`scripts/10_run_classifier_eval.py` with no API key configured "succeeded"
silently: every example fell back to `OTHER_AMBIGUOUS`/confidence 0.0, and
the script wrote a complete-looking `ai_classifier.json` with real-looking
(if poor) accuracy/F1 numbers - a result file indistinguishable from a
genuine but bad classifier run, produced entirely by a setup error. That is
exactly the kind of fabricated result the project's own ground rules
prohibit, and it would have been easy to not notice until writing up
results that don't reproduce. The fix separates two genuinely different
failure classes: a model occasionally returning malformed JSON is a
per-example robustness issue worth falling back gracefully on; missing
credentials is a setup problem that should stop the whole run immediately
and say so.

**Trade-off:** None really - this is strictly safer. It's recorded here as
a reminder that "add a fallback for robustness" needs a second look at
*which* exceptions the fallback actually covers.

---

### 12. ESCALATE decisions get a templated hand-off message, never an LLM-generated one

**Decision:** `scripts/11_run_full_pipeline.py` only calls
`generate_reply()` when the escalation policy says `AUTO_HANDLE`. For
`ESCALATE`, the reply is a fixed template string, not a model call.

**Why:** There is no upside to asking a generator to draft a
resolution-shaped reply for a case the system has already decided a human
needs to handle - at best it's wasted API spend, at worst it produces a
plausible-sounding reply that could get sent instead of actually routing to
a human, undermining the whole point of escalating. Templating the
hand-off message removes any hallucination surface area on exactly the
subset of traffic already flagged as highest-risk.

**Trade-off:** The template is generic and doesn't reference the specific
issue, which is slightly less polished than a tailored "we're looking into
your duplicate charge" message would be - judged an acceptable cost for
zero hallucination risk on the highest-stakes cases.

---

### 13. The LLM judge uses a different model than generation, and is validated without seeing its own kind of self-bias

**Decision:** `.env.example` defaults `OPENROUTER_GENERATION_MODEL` to an
OpenAI model and `OPENROUTER_JUDGE_MODEL` to an Anthropic model.
Additionally, `scripts/13_sample_human_review.py` deliberately withholds
the judge's own scores from the file a human rater fills in.

**Why:** Two distinct self-bias risks, two distinct mitigations. First,
models are known to rate their own outputs more favorably than a different
model's outputs ("self-preference bias" in LLM-as-judge literature) - using
a different provider/model for judging is a cheap, standard mitigation.
Second, a human rater who can see "the judge already said 4/5" while
scoring the same reply will anchor toward that number even when trying not
to, which would inflate the measured judge-human agreement and defeat the
point of running the study at all.

**Trade-off:** Using two different models means the judge's rubric
interpretation might genuinely differ from how the generation model would
"grade itself," which is exactly the point, but also means judge scores
reflect one specific model's rubric interpretation, not some
model-agnostic ground truth - a third model might score differently again.

---

### 14. Escalation reason categories are a fixed, validated 6-item enum, not free text

**Decision:** `escalation_reason` in both the golden set
(`scripts/06_build_golden_set.py`) and the runtime policy
(`src/support_agent/escalation.py`) is constrained to exactly six values
(`SENSITIVE_FINANCIAL`, `ACCOUNT_SPECIFIC_INVESTIGATION`,
`SECURITY_CONCERN`, `AMBIGUOUS_OR_INSUFFICIENT_INFO`,
`REPEATED_UNRESOLVED_ISSUE`, `OUT_OF_SCOPE`), enforced at golden-set build
time (`scripts/06` raises on anything else).

**Why:** Free-text escalation reasons would be more expressive per-example
but impossible to aggregate into "what fraction of escalations are
financial vs. security vs. ambiguous" - a breakdown this project relies on
for the results section and for judging whether the automated policy's
*reasons* (not just its accept/reject decisions) look like a human's
reasons. A fixed enum, chosen from the assignment's own example categories
and refined against what actually showed up in the golden-set annotation
pass, keeps every downstream analysis a simple groupby.

**Trade-off:** A handful of golden examples had reasoning that didn't
cleanly fit one bucket (e.g., `spotify_1997120`: financial dispute AND 3
prior unresolved contact attempts) and had to be force-fit to the single
most actionable category, with the secondary consideration left in the
free-text `notes` field instead of a structured field.

---

### 15. LLM phases run on free-tier OpenRouter models, not the paid defaults in `.env.example`

**Decision:** The actual `.env` used to produce Phase 7/9/11 results points
`OPENROUTER_CLASSIFIER_MODEL`/`OPENROUTER_GENERATION_MODEL` at
`nvidia/nemotron-3-super-120b-a12b:free` and `OPENROUTER_JUDGE_MODEL` at
`dots-studio/dots-3-note-preview:free`, rather than the `gpt-4o-mini`/
`claude-3.5-sonnet` defaults `.env.example` recommends.

**Why:** The provided OpenRouter key returned `402 Insufficient credits` on
the first real call - a billing constraint, not a code issue (confirmed by
first testing with a trivial classification call before assuming anything
was broken). Given the choice to wait for credits, use free models, or stop,
free models were chosen to keep moving. Model selection was empirical, not
assumed: `google/gemma-4-31b-it:free` was tried first (a well-known model
family) and consistently returned `429` "temporarily rate-limited upstream"
from OpenRouter's shared free pool on repeated attempts minutes apart, so it
was dropped rather than built around and hoped to work during a 228-example
batch run. `nvidia/nemotron-3-super-120b-a12b:free` and
`dots-studio/dots-3-note-preview:free` were both smoke-tested on the actual
classify/generate/judge code paths (not just "hello world") before being
adopted, and specifically chosen as two different providers/model families
for classification+generation vs. judging, preserving the anti-self-preference
intent of decision #13 even though neither is the original paid pick.

**Trade-off:** Free-tier models are less capacity-tested for structured
output discipline and instruction-following than `gpt-4o-mini`/
`claude-3.5-sonnet`, and are subject to shared-pool rate limiting that could
cause `LLMConfigError`-adjacent failures mid-batch (the `call_json` retry
logic handles malformed *output*, but a persistent 429/402 mid-run would
still need re-running from cache - see `scripts/10-12`'s per-conversation_id
caching, which makes a resumed re-run cheap). Reported LLM-based results
should be read as "what these specific free models produce," not as a
ceiling on what the architecture could achieve with the originally-intended
paid models.
