# Interview Notes

Quick-reference for explaining and modifying this system live. Cross-refs
point to the fuller writeups.

## Architecture in one paragraph

Raw tweets (`data/raw/twcs.csv`) → conversation threads reconstructed for
one brand, SpotifyCares (`scripts/03_build_conversations.py`) → split
conversation-level into train (retrieval corpus + baseline training) and
eval (golden-set sampling frame) so nothing in evaluation ever leaks into
what the system learned from. An incoming message goes: **classify intent**
(`src/support_agent/classifier.py`, LLM; `scripts/08` for the TF-IDF
baseline) → **retrieve** similar historical resolutions
(`src/support_agent/retrieval.py`, TF-IDF cosine similarity) → **decide**
AUTO_HANDLE/ESCALATE (`src/support_agent/escalation.py`, rule-based on
intent + confidence + retrieval quality + risk keywords) → if AUTO_HANDLE,
**generate** a reply grounded in the retrieved evidence
(`src/support_agent/generation.py`); if ESCALATE, a templated hand-off
message, never a generated one (see decision log #12).

## Why SpotifyCares (not Amazon/Apple/Airlines)

Single product/service → intents cluster into a small, coherent taxonomy.
Amazon spans every retail category; Apple spans many hardware lines - both
would need either a much bigger taxonomy or a uselessly coarse one. Backed
by EDA evidence: 43,265 agent messages, 13,476+ root threads, 99.8%
in-thread reply rate. Full reasoning: `docs/decision_log.md` #1,
`reports/eda/eda_summary.md`.

**Likely question: "Why not the biggest brand (Amazon)?"** More volume
isn't the bottleneck here - the assignment caps at a representative
subsample anyway. Taxonomy cleanliness and consistent domain scope mattered
more for building something explainable in the time available.

## Why 9 intents, not 6-8

Financial cluster (`PAYMENT_BILLING_ISSUE` / `REFUND_REQUEST` /
`SUBSCRIPTION_PLAN_MANAGEMENT`) is kept fine-grained on purpose because
each has a different escalation profile - collapsing them would just move
that distinction-making into the escalation policy instead of the
taxonomy. Full reasoning: `docs/decision_log.md` #4,
`docs/intent_taxonomy.md`.

**Likely question: "Isn't 9 intents a lot for a demo?"** It's within the
brief's own guidance ("approximately 6-10... unless the data strongly
suggests otherwise") and was derived from reading real messages, not
picked arbitrarily - `docs/intent_taxonomy.md` shows the actual examples
that drove each split.

## Why each baseline

- **Trivial**: always predict the majority intent from the *train* split's
  weak labels (not the golden set's own labels - that would be circular).
  Scores almost nothing (macro F1 0.022) - that's the point.
- **TF-IDF + LogisticRegression**: standard, fast, interpretable simple-ML
  baseline; `class_weight="balanced"` compensates for the weak-labeled
  training set being 53% one class. Scores macro F1 0.604 - a real,
  non-trivial number to beat.

Both trained/evaluated identically via `src/support_agent/evaluate_intent.py`
so numbers are directly comparable. `docs/decision_log.md` #8.

**Likely question: "Why not an SVM, or a small transformer, for the second
baseline?"** The brief explicitly suggests TF-IDF+LogReg or LinearSVM;
LogReg was picked for its native `predict_proba` (needed as a confidence
signal for the escalation policy sanity check in `scripts/15`), which SVM
doesn't give as naturally.

## Why the train/eval split prevents leakage

`scripts/03_build_conversations.py` assigns each conversation to train/eval
via `hash(conversation_id) % 100`, deterministic and independent of any
model. The **golden set is sampled only from eval** (`scripts/05`); the
**retrieval index and TF-IDF baseline are built only from train**
(`scripts/09`, `scripts/08`). So a golden example's exact historical
resolution can never be retrieved by the system being evaluated on it.

**Likely question: "Doesn't near-duplicate content still leak across the
split?"** Yes, partially - flagged honestly in
`docs/misleading_headline_number.md` #6. Conversation-level splitting
prevents exact leakage, not near-duplicate leakage (common in this dataset,
e.g. many near-identical "why isn't X available" complaints).

## How retrieval actually works (and why it's TF-IDF, not embeddings)

`src/support_agent/retrieval.py`: fits one `TfidfVectorizer` over every
train-split customer message, L2-normalizes rows, and at query time does a
sparse matrix-vector product (cosine similarity since rows are normalized)
to get top-k. Originally built with `sentence-transformers`, which crashed
the Python process with a native fault in this environment - full
diagnostic story in `docs/decision_log.md` #9. Be ready to explain *why*
that's an acceptable engineering trade (proven-stable, "sparse embedding"
framing) not a corner cut silently.

## How escalation actually works

`src/support_agent/escalation.py::decide()` checks, in order: (1) risk
keywords ("hacked," "chargeback," etc.) → immediate escalate regardless of
everything else; (2) intent-based unconditional escalation
(`REFUND_REQUEST` always); (3) confidence below a per-intent threshold; (4)
weak top retrieval similarity. Every decision returns which signal fired
(`EscalationDecision.signals`), so it's inspectable, not a black box.
Ground truth for what *should* escalate is `docs/escalation_policy.md`,
built by hand-judging message content - NOT by mirroring the real brand's
historical replies, because SpotifyCares' real replies are almost always
"please DM us" regardless of severity (a genuinely surprising finding, see
`data/golden/ANNOTATION_METHODOLOGY.md`).

**Likely question: "Why not learn the escalation policy from the golden
set?"** 228 examples would overfit badly, and a learned model gives no
inspectable "why" - decision log #10 has the full argument.

## Why the evaluation is trustworthy (and where it isn't)

Trustworthy: fixed golden set with a written labelling rubric, leakage
prevention by construction, identical metrics code path for every
baseline, everything reproducible from scripts with a fixed seed, caching
so re-runs don't cost API credits. Not fully trustworthy yet: single
annotator (no inter-rater agreement), stratified not
traffic-proportional sampling, and - the biggest current gap - **the LLM
classifier/generator/judge haven't actually run** (blocked on an API key),
so every "AI system" number in this repo doesn't exist yet. See
`docs/misleading_headline_number.md` for the full list.

## Key files to know cold

| File | What it does |
|---|---|
| `scripts/03_build_conversations.py` | Conversation reconstruction + the leakage-preventing split |
| `src/support_agent/weak_labels.py` | Heuristic labeler - sampling/training aid ONLY, never ground truth |
| `data/golden/golden_set.jsonl` + `ANNOTATION_METHODOLOGY.md` | The 228-example evaluation set and how it was built |
| `src/support_agent/escalation.py` | The whole escalation policy in ~100 lines |
| `src/support_agent/retrieval.py` | TF-IDF retrieval index |
| `src/support_agent/llm_client.py` | Structured-JSON LLM calling with validation/retry |
| `docs/decision_log.md` | Every non-obvious call and why |

## Limitations, unprompted (say these before being asked)

1. LLM classifier/generator/judge are unexecuted - code-complete, pending
   an API key.
2. Golden-set and human-review labels are single-annotator, disclosed as
   such rather than dressed up as independent human labelling.
3. Retrieval is lexical (TF-IDF), not semantic - documented environment
   constraint, not a design preference.
4. Escalation thresholds are reasoned-about placeholders pending
   calibration against a real classifier's confidence distribution.
5. English-only by construction (an ASCII-ratio proxy filter).
