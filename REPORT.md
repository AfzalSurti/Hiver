# REPORT: AI Customer Support Agent for SpotifyCares

## 1. Executive Summary

This project builds an AI support agent for one brand from the Kaggle
"Customer Support on Twitter" dataset - **SpotifyCares** - that classifies
incoming customer messages into a 9-intent taxonomy, retrieves grounded
historical resolutions, drafts a reply, and decides AUTO_HANDLE vs. ESCALATE
with a reason. The project prioritizes **evaluation credibility** over
architectural sophistication: a 228-example hand-labelled golden set, two
baselines, a full LLM-based pipeline (classifier, retrieval-grounded
generator, LLM-as-judge), and an explicit escalation policy, all built on a
leakage-safe train/eval split.

**Current status, stated plainly:** the non-LLM pipeline (data, taxonomy,
golden set, both baselines, retrieval, escalation policy) is complete,
tested, and has real, measured results. The LLM-dependent components
(intent classifier, reply generator, judge, human-agreement study) are
fully implemented but have **not executed**, because no LLM API key was
available in the build environment. Per this project's own rule against
fabricating results, no numbers are invented for these components - their
methodology is documented in full, and every script is ready to run the
moment a key is supplied (see `README.md`).

**Headline result available today:** TF-IDF + Logistic Regression scores
**macro F1 = 0.604** on the golden set (vs. a trivial baseline's 0.022) -
see Section 12 for what that number does and doesn't prove.

## 2. Problem Framing

The real question is not "can an LLM chatbot answer Spotify support
tweets" but **"can we build credible evidence for when an automated agent
should be trusted, and when it should hand off to a human?"** Every design
choice below is in service of that: a written escalation policy with
inspectable signals, a golden set built to stress ambiguity and edge cases
rather than only easy examples, and an evaluation harness that never treats
an unexecuted step as a zero or a guess.

## 3. Dataset and Brand Selection

The raw dataset (2,811,774 tweets) was explored in `scripts/01_explore_data.py`
(see `reports/eda/eda_summary.md`). **SpotifyCares** was selected over
higher-volume candidates (AmazonHelp: 169,840 agent messages, AppleSupport:
106,860) because it is a single product/service, so its ~43,265 agent
messages and 13,476+ conversation threads (99.8% in-thread reply rate)
cluster into a small, coherent intent set - unlike Amazon (spans all retail
categories) or Apple (spans many hardware lines). Full evidence:
`docs/decision_log.md` #1.

Because Kaggle's API requires per-user credentials unavailable in this
environment, the dataset was sourced from a verified Hugging Face mirror
(`SunidhiSriram/twcs`), schema- and row-count-checked against the known
canonical dataset (decision log #2).

**Conversation construction** (`scripts/03_build_conversations.py`)
reconstructs reply chains, cleans HTML entities and mentions, filters
short/duplicate/unresolved/non-English threads (documented rules in the
script), and produces 13,195 conversation examples. **Critically**, each
conversation is assigned to a `train` (80%) or `eval` (20%) split via a
deterministic hash of its `conversation_id` - the retrieval index and
TF-IDF baseline are built only from `train`; the golden set is sampled only
from `eval`, so the system can never retrieve or have trained on a golden
example's own historical resolution.

## 4. Intent Taxonomy

9 intents, derived from reading ~150 real customer messages, not chosen a
priori: `ACCOUNT_LOGIN_ACCESS`, `PAYMENT_BILLING_ISSUE`, `REFUND_REQUEST`,
`SUBSCRIPTION_PLAN_MANAGEMENT`, `TECHNICAL_PLAYBACK_ISSUE`,
`CONTENT_CATALOG_AVAILABILITY`, `FEATURE_REQUEST_OR_INFO`,
`POSITIVE_FEEDBACK_OR_RESOLVED`, `OTHER_AMBIGUOUS`. The financial cluster
(first three) is kept fine-grained deliberately - each has a different
escalation profile, and collapsing them would just move that distinction
into the escalation policy instead (`docs/decision_log.md` #4). Full
definitions, inclusion/exclusion criteria, and ambiguous-case notes:
`docs/intent_taxonomy.md`.

## 5. System Architecture

```
customer message
  -> classify intent (LLM: src/support_agent/classifier.py; baseline: TF-IDF+LogReg)
  -> retrieve top-k similar historical resolutions (TF-IDF cosine similarity, train split only)
  -> decide AUTO_HANDLE / ESCALATE (rule-based: risk keywords > always-escalate intents >
     confidence threshold > retrieval-evidence threshold)
  -> if AUTO_HANDLE: generate a reply grounded in retrieved evidence (LLM)
     if ESCALATE: templated hand-off message (no LLM call - eliminates hallucination
     risk on exactly the highest-risk subset, decision log #12)
```

Every LLM call goes through `src/support_agent/llm_client.py`, which
requests structured JSON, validates it, and retries with a corrective
message on malformed output before falling back safely (never silently on
a missing API key - see decision log #11, a real bug found and fixed during
development).

## 6. Evaluation Methodology

**Golden set**: 228 examples (required range 150-250), sampled from the
eval split only, stratified up to ~20 per intent plus dedicated
short/ambiguous/multi-signal buckets (`scripts/05`), labelled by the
project author against a written rubric (`docs/escalation_policy.md`,
`docs/intent_taxonomy.md`) and disclosed as single-annotator, not
crowd-sourced (`data/golden/ANNOTATION_METHODOLOGY.md`). A key finding from
labelling: SpotifyCares' real historical replies are almost always "please
DM us" regardless of severity, so escalation ground truth is judged from
message content, not from mimicking historical brand behavior.

**Metrics**: identical code path (`src/support_agent/evaluate_intent.py`,
`evaluate_escalation.py`) for every system, so baseline/AI comparisons are
apples-to-apples.

## 7. Baselines

- **Trivial** (`scripts/07`): always predicts the majority intent from the
  *train* split's weak labels (never the golden set's own distribution,
  which would be circular). Accuracy 0.110, macro F1 0.022.
- **TF-IDF + Logistic Regression** (`scripts/08`): trained on heuristic
  weak labels over the full ~10.5k train split (hand-labelling that much
  data was out of scope), `class_weight="balanced"` to counter the weak
  labeler's 53%-`OTHER_AMBIGUOUS` skew. Accuracy 0.618, macro F1 0.604,
  weighted F1 0.649.

## 8. Results

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Trivial (majority class) | 0.110 | 0.022 | 0.022 |
| TF-IDF + LogReg | 0.618 | 0.604 | 0.649 |
| LLM classifier | *pending API key* | *pending* | *pending* |

Per-class F1 (TF-IDF) ranges from 0.24 (`POSITIVE_FEEDBACK_OR_RESOLVED`,
only 4 golden examples) to 0.86 (`PAYMENT_BILLING_ISSUE`) - full breakdown
in `reports/eval/baseline_tfidf.json`.

**Escalation policy** (`scripts/15`, driven by the TF-IDF baseline's intent
+ confidence as a real, runnable-today stand-in for the LLM classifier):
accuracy 0.553, ESCALATE-class F1 0.582, AUTO_HANDLE-class F1 0.519 (full
confusion matrix: `reports/eval/escalation_tfidf.json`). This is explicitly
**not** the final AI system's escalation performance - see Section 12.

## 9. Reply Quality + LLM Judge

**Methodology (complete)**: `src/support_agent/generation.py` drafts a
reply grounded only in retrieved evidence, explicitly instructed not to
invent policies, refunds, URLs, or account actions unsupported by that
evidence. `src/support_agent/judge.py` scores replies 1-5 on six rubric
dimensions (correctness, relevance, groundedness, helpfulness, brand_tone,
safety) plus an overall score, using a **different model than generation**
to reduce self-preference bias.

**Results: pending an OpenRouter API key** (`scripts/11`, `scripts/12`).
No reply-quality numbers are reported because none have been measured yet.

## 10. Human/Judge Agreement

**Methodology (complete)**: `scripts/13` samples ~50 judged replies,
stratified across the judge's own score buckets, into a template with the
judge's own scores deliberately withheld (to avoid anchoring the human
rater). `scripts/14` computes Spearman correlation, exact agreement, and
within-1-point agreement per dimension. Full disclosure of the
single-annotator design: `data/eval/HUMAN_REVIEW_METHODOLOGY.md`.

**Results: pending** (requires Section 9's outputs first).

## 11. Failure Analysis

Five failure modes derived from real evaluation outputs (TF-IDF classifier
+ TF-IDF-driven escalation), full detail in `docs/failure_analysis.md`:

1. Noisy majority-class weak labels drag minority intents toward
   `OTHER_AMBIGUOUS` (36% of all classifier errors).
2. "Thank you" phrasing fools the classifier into
   `POSITIVE_FEEDBACK_OR_RESOLVED` even on open questions.
3. `REFUND_REQUEST` vs. `PAYMENT_BILLING_ISSUE` boundary confusion when the
   refund ask is implicit rather than literal.
4. **Highest severity**: 55/228 golden examples were classified correctly
   and confidently but still auto-handled when a human would escalate -
   intent confidence alone can't detect "this specific instance needs
   investigation" (e.g., a password reset already tried twice).
5. Miscalibrated TF-IDF confidence causes 47/228 unnecessary
   over-escalations - the "safe" failure direction, lower severity than #4.

## 12. What Is Misleading About My Headline Number?

Full version: `docs/misleading_headline_number.md`. In brief, the TF-IDF
macro F1 (0.604) is inflated or obscured by: a stratified (not
traffic-proportional) golden set, noisy training labels the model partly
learned to imitate, thin support on minority classes (4 examples for
`POSITIVE_FEEDBACK_OR_RESOLVED`), single-annotator ground truth with no
measured inter-rater agreement, an English-only sampling filter, residual
near-duplicate leakage risk despite conversation-level splitting, and the
general offline-vs-production gap. This number will need to be
re-examined, not just re-run, once the LLM classifier produces its own.

## 13. What I'd Do With One More Week

Full version: `docs/one_more_week.md`. Priority order: (1) actually run the
LLM classifier/generator/judge - blocks everything else; (2) replace
intent-confidence-only escalation triggers with a "case needs investigation"
signal (fixes the highest-severity failure mode found); (3) calibrate
escalation thresholds against the real classifier's confidence
distribution; (4) fix the retrieval embedding environment issue properly,
in an isolated venv; (5) get a second, genuinely independent human rater;
(6) explore multi-intent classification for genuinely multi-issue messages.

## 14. Decision Log

14 non-obvious decisions with reasoning and trade-offs, in
`docs/decision_log.md`: brand selection, dataset sourcing, LLM gateway
choice, intent granularity, escalation ground-truth methodology,
stratified sampling, single-annotator disclosure, weak-label scope, the
retrieval-approach pivot (sentence-transformers crash), rule-based vs.
learned escalation, the fabricated-result bug fix, templated ESCALATE
replies, judge/generation model separation, and the fixed escalation-reason
enum.

## 15. What We Chose Not to Build

Per the assignment's own scope guidance: no frontend, no real Twitter
integration, no payment system, no real customer accounts, no autonomous
refunds, no multi-brand support, no complex infrastructure (no
message queue, no database beyond flat files - JSONL is sufficient at this
scale and keeps every intermediate artifact human-readable and diffable).
Also chose not to build: a learned escalation policy (would overfit 228
examples and lose inspectability, decision log #10), neural retrieval
embeddings (environment-blocked, decision log #9, TF-IDF is an adequate
stand-in for this project's scope), and multi-label/multi-intent
classification (noted as a "one more week" item rather than in-scope now).

## 16. Limitations

Stated without hedging: the LLM-dependent components have not executed;
golden-set and human-review labels are single-annotator; retrieval is
lexical, not semantic; escalation thresholds are reasoned placeholders, not
calibrated; the dataset is English-only by construction; and offline
evaluation on 228 examples is a floor on production readiness, not a
ceiling on production performance. See `docs/interview_notes.md` for the
unprompted version of this list and how to discuss each point live.
