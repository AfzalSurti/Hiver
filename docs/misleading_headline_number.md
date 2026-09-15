# What Is Misleading About My Headline Number?

**Headline number:** TF-IDF + Logistic Regression intent-classification
**macro F1 = 0.604** on the 228-example golden set
(`reports/eval/baseline_tfidf.json`) - the strongest real, measured result
in this project as of writing (the LLM classifier is code-complete but has
not run - see `README.md`). Everything below applies equally once the LLM
system's headline number exists; the specific mechanisms are general, not
particular to this one metric.

It would be easy to read "macro F1 0.604, beating the trivial baseline's
0.022 by 27x" as "the system classifies intent well." Here's what that
framing hides.

### 1. The golden set is stratified, not traffic-proportional

`scripts/05_sample_golden_candidates.py` deliberately samples up to ~20
examples per intent so every class has enough support to compute a
meaningful F1 (see `docs/decision_log.md` #6). Real SpotifyCares traffic is
almost certainly dominated by a handful of intents, not evenly spread
across 9. **Macro F1 weights every class equally regardless of how common
it actually is** - a model that is excellent on the 3 most common real
intents but poor on rare ones would score similarly to one with the reverse
profile, even though their production impact is completely different.
Weighted F1 (0.649) is closer to what a traffic-weighted number would look
like, but still weighted by the golden set's *artificial* distribution, not
real traffic.

### 2. Training labels are noisy, so the number partly measures label-noise robustness, not classification skill

The TF-IDF model trained on heuristic weak labels that are 53.4%
`OTHER_AMBIGUOUS` due to limited regex recall (`docs/decision_log.md` #8).
Any accuracy the model shows on golden `OTHER_AMBIGUOUS` examples (25 of
them) partly reflects "the model learned to imitate the heuristic's
blind spots," not "the model understands ambiguity." The failure analysis
(`docs/failure_analysis.md` #1-2) shows this concretely: minority intents
get pulled toward `OTHER_AMBIGUOUS` and `POSITIVE_FEEDBACK_OR_RESOLVED`
specifically because of training-label artifacts, not inherent task
difficulty.

### 3. Evaluation-set size limits precision on minority classes

`POSITIVE_FEEDBACK_OR_RESOLVED` has only 4 golden examples. Its F1 (0.24)
is computed from a handful of predictions - one or two different outcomes
would swing that class's F1 by 0.25-0.5. Macro F1 gives this thin class
equal weight to `ACCOUNT_LOGIN_ACCESS` (37 examples, F1 0.825), so a
meaningful chunk of the headline number's value is driven by a class where
the number itself is barely a stable estimate.

### 4. Annotation is single-annotator, so "ground truth" carries real uncertainty

Every golden label came from one annotator (the project author) applying a
written rubric, not an independent panel (`data/golden/ANNOTATION_METHODOLOGY.md`).
There is no inter-annotator agreement statistic for the golden labels
themselves. Some genuinely ambiguous cases (truncated tweets, referent-free
follow-ups) were resolved by documented convention rather than by
consensus - a different annotator applying the same rubric might land on a
handful of different labels, which would move the headline number by some
amount that hasn't been measured.

### 5. Sampling bias from the English-only filter

`scripts/03_build_conversations.py`'s `ascii_ratio >= 0.85` filter is a
coarse English-language proxy. It systematically excludes non-English
complaints, so the golden set - and any number computed from it - describes
performance on the English-speaking slice of SpotifyCares' actual customer
base, not the whole thing.

### 6. Data-leakage risk is mitigated but not eliminated

The train/eval split is conversation-level and hash-based
(`scripts/03`), which prevents the *exact* conversation a golden example
comes from from being in the training or retrieval pool. It does **not**
prevent near-duplicate or templated messages (common in this dataset - many
customers file very similar complaints, e.g. "why isn't [song] available")
from appearing on both sides of the split. Some of the TF-IDF baseline's
apparent skill on common phrasings could be inflated by lexical overlap
with near-duplicate train examples rather than genuine generalization.

### 7. Offline evaluation is not production evaluation

228 examples, evaluated once, with no time-based drift, no adversarial
inputs, no multi-lingual traffic, and no real user reaction to auto-handled
replies (which the golden set can't capture at all - there is no measurement
here of whether an auto-handled reply actually *resolved* the customer's
issue in the way the historical brand reply's "problem fixed, thank you"
follow-ups would indicate). A headline offline number is a floor on
production readiness, not a ceiling estimate of production performance.

### 8. This number will look different, possibly quite different, once the LLM classifier runs

The TF-IDF number is anchored on a specific vectorization + noisy-label
combination. The LLM classifier (Phase 7) sees the same golden set but a
completely different signal (zero-shot reasoning over the taxonomy
descriptions, not n-gram frequency), so there is no guarantee the two
systems fail on the same examples, or that the LLM's number is strictly
better - it could be worse on some classes if the taxonomy descriptions are
ambiguous to the LLM in ways the TF-IDF's training data happened to
disambiguate. This section should be re-examined, not just re-run, once
that number exists.
