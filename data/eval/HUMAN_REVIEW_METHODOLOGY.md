# LLM-Judge vs. Human Agreement: Methodology

## Status

This is a **pending manual step**, exactly like the golden-set labelling in
`data/golden/ANNOTATION_METHODOLOGY.md`. `scripts/13_sample_human_review.py`
produces `human_review_template.jsonl` with the rubric fields left as
`null`; those fields must be filled in by hand before
`scripts/14_compute_judge_agreement.py` will run (it fails loudly on any
unfilled field rather than silently treating `null` as a score). No
agreement numbers are reported anywhere in this repository until that
manual pass is actually done - see `docs/decision_log.md` for why this
project treats "never fabricate a result" as a hard rule.

## Who does the rating, and why that's disclosed

Same as the golden set: **the project author is the single human rater**,
not an independent panel. This is stated plainly rather than dressed up,
per the project's own rule against calling something "human-labelled" if it
wasn't produced by an independent process. Concretely:

- This measures judge-vs-single-annotator agreement, not judge-vs-panel
  agreement - a real, disclosed limitation on how strong a claim "the judge
  agrees with humans" actually is here.
- The rating rubric is identical to the one given to the LLM judge
  (`src/support_agent/judge.py`'s `SYSTEM_PROMPT` - the six dimensions:
  correctness, relevance, groundedness, helpfulness, brand_tone, safety,
  plus an overall score), so the comparison is apples-to-apples.

## Why the judge's own scores are withheld from the rating file

`scripts/13_sample_human_review.py` deliberately does NOT include the LLM
judge's scores in `human_review_template.jsonl`. A human rater who can see
"the judge said 4/5" while rating tends to anchor on that number even when
trying not to - this would inflate the measured agreement and defeat the
point of the study. The judge's scores are joined back in only at
`scripts/14`, purely for computing the correlation after all human scores
are already locked in.

## Sampling

Sampled from `reports/eval/judge_scores.jsonl` (i.e., only replies that
were actually AUTO_HANDLE-decided and successfully judged - see
`scripts/11`/`scripts/12`), stratified across the judge's own overall-score
buckets (1-2 "low", 3 "mid", 4-5 "high") so the sample isn't just whatever
the natural score distribution concentrates on. Target size: 50, within the
brief's 40-60 range. Seeded (see `support_agent.config.SEED`) for
reproducibility of *which* examples are chosen - the ratings themselves are
still a manual step that must be re-done if the sample changes.

## Rating process

For each of the ~50 sampled `(customer_message, reply)` pairs, the rater:

1. Reads the customer message and the drafted reply (retrieved evidence is
   intentionally not re-shown here, mirroring what the judge itself
   evaluated against - if evidence context is needed to judge groundedness,
   consult `reports/eval/pipeline_outputs.jsonl` by `conversation_id`).
2. Assigns 1-5 scores for each of the six rubric dimensions plus an overall
   score, using the exact rubric text in `src/support_agent/judge.py`.
3. Writes a short rationale, particularly for any low score, so
   disagreements with the judge are auditable rather than opaque numbers.

## Reported statistic

`scripts/14_compute_judge_agreement.py` reports, per dimension and overall:

- **Spearman rank correlation** (appropriate for ordinal 1-5 scores, robust
  to the two raters using the scale slightly differently) with its p-value.
- **Exact agreement rate** and **within-1-point agreement rate** as more
  interpretable companions to the correlation coefficient.

## Known limitations (see also `REPORT.md`)

- Single annotator - no inter-rater reliability among *humans* is measured,
  only judge-vs-human.
- The rater is the same person who wrote the golden-set labels and the
  intent taxonomy, so is not a naive/blind reviewer of this system - a
  genuinely independent rater (unfamiliar with the project) would be a
  stronger design, noted in `REPORT.md`'s "what I'd do with one more week."
- 50 examples gives a rough correlation estimate, not a tight confidence
  interval - the exact sample size vs. the reported Spearman r's precision
  should be read with that in mind.
