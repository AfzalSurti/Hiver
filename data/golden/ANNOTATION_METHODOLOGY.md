# Golden Set Annotation Methodology

## Who labelled this and why that matters

**Every label in `golden_set.jsonl` was produced by the project author acting
as a single annotator** (working through this repository with AI-assistant
tooling, applying the documented rubrics below), **not by independent human
crowd-workers or domain-expert reviewers.** This is explicitly disclosed
rather than presented as blind third-party human labelling, per the
project's own ground rule not to call something "human-labelled" if it
wasn't produced that way. Concretely this means:

- There is no inter-annotator agreement statistic for the golden set itself
  (only one annotator produced it).
- The LLM-judge-vs-human agreement study (Phase 12) is therefore judge-vs.
  *single-annotator*, not judge-vs.-panel - a real limitation, discussed
  again in `REPORT.md`'s "what's misleading about the headline number"
  section.
- Labels follow a written, versioned rubric (`docs/intent_taxonomy.md`,
  `docs/escalation_policy.md`) precisely so a second annotator *could*
  reproduce or audit them later - the rubric being explicit and public is
  what stands in for inter-annotator agreement here.

## Sampling (scripts/05_sample_golden_candidates.py)

228 candidates were drawn from the **eval split only** (2,634 conversations,
20% of all built conversations, held out by a deterministic hash of
`conversation_id` - see `scripts/03_build_conversations.py`). The **train
split is never touched by this script**, which is what prevents the
retrieval/generation system from having seen the exact resolution of a
message that ends up being evaluated.

A heuristic keyword/regex labeler (`src/support_agent/weak_labels.py`) was
run over the eval pool purely to **stratify sampling**, not to assign
ground truth:

- up to 20 candidates per intent (whichever intent the heuristic guessed)
- ~15 "multi-signal" messages where 2+ intent keyword patterns fired at
  once, as a cheap proxy for multi-issue/ambiguous cases
- ~15 short messages (≤6 words) to stress-test low-signal input
- ~20 pure-random top-up from the remaining pool, so the set isn't entirely
  an artifact of the heuristics' blind spots

All buckets were de-duplicated by `conversation_id` and shuffled before
annotation so label order wouldn't cluster by sampling bucket. Final size:
228, within the required 150-250 range.

## Labelling process (this document + manual pass -> annotations.jsonl)

For each of the 228 candidates, the annotator read the customer's opening
message (and, for context only - see below - the real historical brand
reply) and assigned:

- **`intent`**: one of the 9 labels in `docs/intent_taxonomy.md`, using that
  document's inclusion/exclusion criteria.
- **`expected_action`**: `AUTO_HANDLE` or `ESCALATE`, per the ground-truth
  policy in `docs/escalation_policy.md`.
- **`escalation_reason`**: required and validated (see
  `scripts/06_build_golden_set.py`) whenever `expected_action=ESCALATE`,
  one of six categories (`SENSITIVE_FINANCIAL`,
  `ACCOUNT_SPECIFIC_INVESTIGATION`, `SECURITY_CONCERN`,
  `AMBIGUOUS_OR_INSUFFICIENT_INFO`, `REPEATED_UNRESOLVED_ISSUE`,
  `OUT_OF_SCOPE`); must be `null` for `AUTO_HANDLE`.
- **`notes`**: a one- or two-sentence rationale, required for every example.
  This is what makes disagreements auditable later instead of just being an
  opaque label.

### Key methodological decision: labels are based on the customer's opening
### message alone, not the full resolved thread

The classifier and escalation policy this project builds only ever see an
*incoming* message - they don't get to see how the conversation eventually
played out. So instead of using the real historical brand reply to *decide*
the label (which would leak the resolution into the ground truth and make
the task artificially easy), the annotator judged `intent` and
`expected_action` from the customer's message content alone, exactly as the
deployed system would. The real brand reply was read only as background
context (e.g. to understand dataset artifacts like truncated tweets), and in
a few cases to confirm a real-world routing pattern (e.g. artist/rights
requests really do get routed to a separate Artist Support team in
Spotify's actual process, which is why those are labelled `OUT_OF_SCOPE`
here too).

### An important, non-obvious finding from doing this by hand

The overwhelming majority of SpotifyCares' real historical replies are a
boilerplate "please DM us your account email/username" **regardless of how
severe or trivial the issue is** - it's their standard privacy/verification
practice, not a signal of how serious the brand considered the issue. This
matters a lot: it means the `brand_response` field in this dataset is a poor
signal for "was this auto-handleable," and building the escalation ground
truth by mirroring what the brand literally did (e.g., "brand asked for a
DM, so this must need escalation") would have produced a policy that
escalates almost everything, since almost every historical reply says some
version of "DM us." This is exactly why the escalation labels here are
based on independent judgment against the written policy, not on imitating
the historical brand behavior. It's also why `AUTO_HANDLE` replies generated
by this project's system (Phase 9) are necessarily *synthesized* generic
guidance grounded in the *content* of similar historical resolutions, not
verbatim copies of "please DM us" - seeing that pattern early shaped the
reply-generation design, documented in `docs/decision_log.md`.

### Weak-label accuracy, observed during annotation

The heuristic sampling label was corrected during manual annotation in a
large fraction of cases - most visibly, of the ~20 candidates the heuristic
tagged `POSITIVE_FEEDBACK_OR_RESOLVED`, only 4 survived manual review as
actually positive-feedback-with-no-ask; the rest were re-labelled to
whatever their real underlying ask was (a "thank you" wrapped around a
still-open bug report, a feature question, etc.). This is expected and
fine - the heuristic's only job was sampling diversity - but it's a visible,
quantified illustration of why the same heuristic module is an honest but
limited baseline-training tool (Phase 6), not a source of ground truth.

### Handling genuinely ambiguous cases

Some messages are irreducibly ambiguous given only the opening message (no
external context, no attachments, no thread history beyond what's in
`conversation_context`):

- **Truncated tweets** (e.g. `spotify_1101315`, cut off mid-sentence) - a
  real artifact of this dataset, not an annotation error. Labelled
  `OTHER_AMBIGUOUS` / `AMBIGUOUS_OR_INSUFFICIENT_INFO`.
- **Referent-free follow-ups** ("bring the feature back", "add this
  functionality") that clearly continued an earlier exchange not visible in
  the root message. Labelled `OTHER_AMBIGUOUS` for consistency, even though
  a human with the full original thread might resolve them.
- **Rights-holder / artist-relations requests** ("my artist's track is
  under the wrong name") were consistently labelled `CONTENT_CATALOG_AVAILABILITY`
  / `OUT_OF_SCOPE`, distinguished from an ordinary listener noticing the same
  kind of metadata error (which auto-handles) by whether the customer
  identifies themselves as the rights-holder ("my music", "our EP",
  "we've released").

### Limitations of this golden set (see also `REPORT.md`)

- Single annotator, not a panel - no inter-annotator agreement is reported
  for the golden labels themselves.
- English-only by construction (the `ascii_ratio` filter in
  `scripts/03_build_conversations.py` is a coarse proxy for "primarily
  English" and undercounts non-English complaints).
- Deliberately **not proportional to real traffic** - intents were sampled
  up to a target count each (stratified), so accuracy/F1 numbers on this set
  describe performance on a *diversity-balanced* set, not on the intent
  distribution SpotifyCares actually receives. `POSITIVE_FEEDBACK_OR_RESOLVED`
  ended up under-represented (4/228) purely because true positive-feedback
  messages were rarer than the heuristic's false positives for that class.
