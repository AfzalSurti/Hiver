# Failure Analysis

**Scope note:** this analysis is based on the two systems that have actually
run end-to-end against the golden set so far - the TF-IDF+LogReg intent
baseline (`reports/eval/baseline_tfidf.json`) and the escalation policy
driven by that baseline's predictions (`reports/eval/escalation_tfidf.json`,
`escalation_tfidf_details.jsonl`). The LLM classifier/generator/judge
(Phases 7/9/11) are code-complete but have not run - see
`docs/decision_log.md` and `README.md` - so this document analyzes real,
measured failures of the system that has actually run, not the LLM system.
It should be extended once an API key is available and Phase 13's full
pipeline evaluation runs (see `docs/one_more_week.md`).

All examples below are real golden-set messages and real model outputs -
none are invented.

---

## 1. Noisy majority-class weak labels drag minority intents toward OTHER_AMBIGUOUS

**Real examples (true_intent -> predicted_intent):**
- `TECHNICAL_PLAYBACK_ISSUE -> OTHER_AMBIGUOUS`: *"Is anyone else getting a
  504 Gateway Time-Out error on Spotify?"*
- `FEATURE_REQUEST_OR_INFO -> OTHER_AMBIGUOUS`: *"When are the lyrics
  coming back"*
- `CONTENT_CATALOG_AVAILABILITY -> OTHER_AMBIGUOUS`, `SUBSCRIPTION_PLAN_MANAGEMENT
  -> OTHER_AMBIGUOUS`: same pattern, 7 and 6 cases respectively.

**Expected vs. actual:** expected the specific intent; actual is the
catch-all bucket in 10, 8, 7, and 6 of the 87 total errors respectively -
this single confusion direction accounts for **31 of 87 errors (36%)**.

**Why it failed:** the TF-IDF baseline is trained on heuristic weak labels
(`scripts/04`) that are 53.4% `OTHER_AMBIGUOUS` (see
`docs/decision_log.md` #8) purely because the keyword regexes have limited
recall, not because most real traffic is unclassifiable. `class_weight="balanced"`
corrects for this partially but the model still learned `OTHER_AMBIGUOUS`
as a very broad, low-specificity catch-all, so short or lexically-unusual
phrasings of real intents fall into it.

**Hypothesis:** the training *label noise*, not the modeling approach, is
the bottleneck - `POSITIVE_FEEDBACK_OR_RESOLVED` per-class F1 is only 0.24
(4 support) while classes with cleaner heuristic coverage
(`PAYMENT_BILLING_ISSUE`, `ACCOUNT_LOGIN_ACCESS`) score 0.86 and 0.83.

**Potential fix:** hand-label a modest stratified sample of the train split
(even 500-1000 examples) instead of relying purely on heuristics, or use
the eventual LLM classifier itself to weak-label the train split (higher
recall than regexes) as a second-generation silver-label set.

---

## 2. "Thank you" phrasing fools the classifier into POSITIVE_FEEDBACK_OR_RESOLVED

**Real examples:**
- *"Although I would like to know if the app works on the watch. Thank
  you"* (true: `FEATURE_REQUEST_OR_INFO`)
- *"Hey are you ever gona put globe prepaid back as a method of payment?
  From Philippines here. Thanks."* (true: `FEATURE_REQUEST_OR_INFO`)
- *"is there an ETA on when the iOS app will get iPhone X support,
  thanks."* (true: `FEATURE_REQUEST_OR_INFO`)

**Expected vs. actual:** expected `FEATURE_REQUEST_OR_INFO` (5 cases) or
`OTHER_AMBIGUOUS` (5 cases) or `CONTENT_CATALOG_AVAILABILITY` (4 cases);
actual `POSITIVE_FEEDBACK_OR_RESOLVED` in all of these.

**Why it failed:** this is the exact weak-label quality problem documented
in `data/golden/ANNOTATION_METHODOLOGY.md` - the heuristic labeler used to
build training silver labels matches "thank(s)" as a positive-feedback
signal (`src/support_agent/weak_labels.py`), so *every* message that ends
politely with "thanks" - even one asking an open question - gets
mislabeled in training data. The model faithfully learned this spurious
correlation.

**Hypothesis:** superficial politeness markers are a weak, ambiguous
signal for this task, and the heuristic's negative lookahead (excluding
"thanks for the (charge|bug)") isn't nearly broad enough to catch every
non-positive use of "thanks."

**Potential fix:** require positive-feedback heuristic matches to have
*no* trailing question mark and no interrogative opener ("is there,"
"when will," "how do I"), or simply drop this intent from the weak-label
training set entirely given how rare it is in the true golden distribution
(4/228) and how much training-label noise it introduces relative to its
real-world frequency.

---

## 3. REFUND_REQUEST vs. PAYMENT_BILLING_ISSUE boundary confusion

**Real examples:**
- *"I canceled Spotify months ago! And I noticed recently I'm still
  getting charged the premium rate. But my account says free."* (true:
  `REFUND_REQUEST`, predicted: `PAYMENT_BILLING_ISSUE`)
- *"i was tryin to buy student premium account n i got charged 2 times in
  a row... can i get refunded?"* (true: `REFUND_REQUEST`, predicted:
  `PAYMENT_BILLING_ISSUE`)

**Expected vs. actual:** 4 of 87 errors are this specific confusion
direction.

**Why it failed:** per `docs/intent_taxonomy.md`, `REFUND_REQUEST` is
defined by an *explicit* money-back ask, which the weak labeler's regex
requires fairly literal phrasing ("refund," "money back," "charged...cancel").
Messages that imply a refund ask without those exact words, or where the
refund ask is a secondary clause after a longer billing complaint, get
weak-labeled (and then predicted) as the more generic
`PAYMENT_BILLING_ISSUE`.

**Hypothesis:** this is the single most consequential confusion pair for
downstream safety, since `REFUND_REQUEST` always escalates in the ground-truth
policy (`docs/escalation_policy.md`) but `PAYMENT_BILLING_ISSUE` only
escalates below a confidence threshold - see failure mode #4 below for what
happens when this misclassification combines with high confidence.

**Potential fix:** broaden the `REFUND_REQUEST` weak-label pattern to catch
"charged...still" / "charged...free now" long-running-unwanted-charge
phrasing (already partially done - see `docs/decision_log.md`'s note on
`spotify_2036095`-style ground truth in the golden set - but the automated
weak-labeler and the automated escalation `ALWAYS_ESCALATE_INTENTS` set
don't yet reflect that refinement).

---

## 4. Overconfident auto-handling on account/billing cases that need investigation (the highest-severity failure mode)

**Real examples (all predicted AUTO_HANDLE, true ESCALATE):**
- *"Really need help, my credit card was charged twice with an hour
  interval of spotify (the student promo)"* - correctly classified
  `PAYMENT_BILLING_ISSUE` at **97.8% confidence**; true reason
  `SENSITIVE_FINANCIAL`.
- *"I can't log in on any device and I've reset my password twice"* -
  correctly classified `ACCOUNT_LOGIN_ACCESS` at **99.6% confidence**; true
  reason `ACCOUNT_SPECIFIC_INVESTIGATION` (generic reset already exhausted).
- *"just upgraded to premium - invited family members but cannot seem to
  set up new accounts"* - correctly classified `SUBSCRIPTION_PLAN_MANAGEMENT`
  at 76% confidence; true reason `ACCOUNT_SPECIFIC_INVESTIGATION`.

**Expected vs. actual:** 55 of 228 golden examples (24%) are this failure
direction - the intent was classified *correctly and confidently*, but the
automated escalation policy still auto-handled a case a human would
escalate.

**Why it failed:** this is not a classification error at all - it's a gap
between what the ground-truth policy actually judges (whether *this
specific message* needs account-specific investigation, e.g. "reset my
password twice, still can't log in" vs. a first-time "I forgot my
password") and what the automated policy can see (just the intent label
and a confidence score). High classifier confidence on the *intent* says
nothing about whether the specific instance is a routine case or an
already-exhausted-self-serve edge case. The current
`INTENT_CONFIDENCE_THRESHOLDS` treat all `ACCOUNT_LOGIN_ACCESS` messages
alike regardless of this distinction.

**Hypothesis:** confidence calibrated against *intent* is fundamentally the
wrong signal for *this* kind of escalation trigger - no amount of
confidence-threshold tuning fixes it, because the classifier was never
asked the right question.

**Potential fix:** the highest-leverage fix identified in this project (see
`docs/one_more_week.md`) - add a second, cheap signal specifically for
"has the customer already tried the standard fix / contacted support
before" (keyword flags: "twice," "again," "still," "already tried," "no
response") that forces escalation independent of intent confidence, mirroring
how the golden-set annotator actually made this call during labelling (see
`data/golden/ANNOTATION_METHODOLOGY.md`).

---

## 5. Miscalibrated confidence causes the "safe-direction" failure: over-escalation

**Real examples (all predicted ESCALATE via LOW_CONFIDENCE, true AUTO_HANDLE):**
- *"Why is Spotify not available in Saudi Arabia App Store?"* - correct
  intent (`CONTENT_CATALOG_AVAILABILITY`) but only 45.1% confidence.
- *"WHY THE FUCK i CANT PLAY WOLVES ON SPOTIFY"* - correct-ish intent but
  38.9% confidence (profanity/caps likely confuse the TF-IDF features).

**Expected vs. actual:** 47 of 228 examples (21%) over-escalate this way -
notably, **100% of over-escalations were due to LOW_CONFIDENCE**, and all
47 had TF-IDF confidence below 0.5 despite often getting the intent right.

**Why it failed:** TF-IDF + LogisticRegression's `predict_proba` output is
not a calibrated probability - across a 9-way classification with noisy
training labels, correct predictions often still get diffuse probability
mass (confidence <50%) simply because the model is uncertain among several
plausible classes even when its top choice is right. The
`DEFAULT_CONFIDENCE_THRESHOLD = 0.55` in `src/support_agent/escalation.py`
was reasoned about, not calibrated against this specific model's score
distribution (explicitly flagged as a placeholder in that module's
docstring).

**Hypothesis:** this failure mode is a direct, expected consequence of
plugging an uncalibrated confidence source into a fixed threshold - it is
*not* evidence that the escalation policy's logic is wrong, only that its
threshold needs to be calibrated per upstream classifier. This is the
"safe" failure direction (unnecessary escalation, not unsafe automation),
which is the deliberately preferred failure mode per
`docs/escalation_policy.md`'s "default: AUTO_HANDLE, but escalate when in
doubt" philosophy - still worth fixing for automation-rate reasons, but
lower severity than failure mode #4.

**Potential fix:** calibrate `DEFAULT_CONFIDENCE_THRESHOLD` and
`INTENT_CONFIDENCE_THRESHOLDS` against the actual confidence distribution
of whichever classifier is deployed (e.g., pick the threshold that
maximizes escalation-F1 on a held-out slice), and re-calibrate again once
the LLM classifier replaces TF-IDF, since LLM-reported confidence has a very
different distribution than TF-IDF's `predict_proba`.
