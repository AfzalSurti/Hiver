# What I'd Do With One More Week

Prioritized by expected impact on the project's actual goal - credible
evidence for when the agent should be trusted vs. escalate - not by what's
most interesting to build.

## 1. Get the LLM classifier/generator/judge actually running (highest priority, blocking everything else)

Every number in this repo beyond the TF-IDF/trivial baselines is
code-complete but unexecuted, pending an `OPENROUTER_API_KEY`
(`README.md`). This is the single highest-leverage next step - the whole
point of the project is evaluating an *AI* agent's trustworthiness, and
that evaluation literally cannot start until Phase 7/9/11/12 run at least
once. Expected impact: unblocks every other item below, which all assume a
real LLM classifier/generator exist to improve.

## 2. Replace intent-level confidence thresholds with a "case-specific investigation needed" signal

Failure analysis #4 (`docs/failure_analysis.md`) found this is the highest-severity
failure mode: 55/228 golden examples were correctly, confidently classified
but still auto-handled when a human would escalate, because *intent*
confidence says nothing about whether *this instance* needs
account-specific investigation (e.g., "reset my password twice, still
broken" vs. a first-time password question). Concretely: add a cheap
keyword/pattern layer for "already tried X," "again," "still not fixed,"
"twice," "no response yet" that forces escalation independent of intent
confidence. Expected impact: directly fixes the most dangerous failure
mode found so far (unsafe over-automation), likely the single biggest
lift to genuine escalation-policy safety.

## 3. Calibrate escalation thresholds against the real classifier's confidence distribution

`src/support_agent/escalation.py`'s thresholds are reasoned-about
placeholders, explicitly flagged as such. Once the LLM classifier
(item 1) produces real confidence scores on the golden set, thresholds
should be swept to find the point that maximizes escalation F1 (or
precision at a fixed recall floor, if false-negative escalations are judged
worse than false positives - a business call worth surfacing rather than
picking silently). Expected impact: failure analysis #5 shows TF-IDF's
uncalibrated confidence caused 47/228 unnecessary escalations - a properly
calibrated LLM confidence signal should recover a meaningful chunk of
automation without sacrificing safety.

## 4. Upgrade retrieval from TF-IDF to neural embeddings (fix the environment issue properly)

Retrieval is currently TF-IDF cosine similarity because
`sentence-transformers` crashed the Python process in this environment
(`docs/decision_log.md` #9) - a real numpy/native-library conflict, not a
design preference. With a week, the right fix is an isolated virtual
environment (not touching the user's shared conda install) with a
known-compatible numpy/torch/sentence-transformers combination, so
retrieval can catch semantic paraphrases TF-IDF misses entirely (e.g. "I
can't sign in" vs. "login isn't working" share no useful n-grams). Expected
impact: better retrieval evidence directly improves groundedness scores
(Phase 11) and reduces the "weak evidence" escalation trigger firing on
cases that do have a good historical match, just not a lexical one.

## 5. Get a second, genuinely independent human rater for both the golden set and the judge-agreement study

Both `data/golden/ANNOTATION_METHODOLOGY.md` and
`data/eval/HUMAN_REVIEW_METHODOLOGY.md` disclose single-annotator labelling
by the project author - someone who also wrote the taxonomy and the
escalation policy, so isn't a blind reviewer of the system. A week is
enough to recruit one more rater, have them label a 50-100 example subset
of the golden set independently, and report real inter-annotator agreement
(Cohen's kappa) alongside the existing single-annotator labels. Expected
impact: turns "here's my rubric and my judgment" into an actually-validated
ground truth, which is the credibility the whole project is trying to
establish.

## 6. Multi-intent classification for genuinely multi-issue messages

The current taxonomy forces one intent per message
(`docs/intent_taxonomy.md`), but several golden examples clearly carry two
(e.g. `spotify_2492535`: a billing correction + a refund ask + a how-to
question in one tweet, labelled `REFUND_REQUEST` because the financial ask
dominates for *action* purposes, but the reply should really address all
three). A week is enough to prototype a multi-label version of the
classifier prompt and see how often single-issue framing actually loses
information worth acting on. Expected impact: likely modest on the golden
set's current size, but this is exactly the kind of edge case that
compounds at real traffic volume - worth measuring before dismissing.
