# Hiver — AI Customer Support Agent (SpotifyCares)

An AI support agent built on the Kaggle ["Customer Support on
Twitter"](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, scoped to one brand: **SpotifyCares**. Given an incoming customer
message it classifies intent, retrieves grounded historical resolutions,
drafts a reply, and decides AUTO_HANDLE vs ESCALATE with a reason.

See `docs/decision_log.md` for why this brand/architecture was chosen, and
`REPORT.md` (added once evaluation is complete) for the full write-up:
methodology, baselines, results, failure analysis, and limitations.

## Status

- [x] Repo scaffolding, `.env.example`, `.gitignore`
- [x] Dataset acquired (`data/raw/twcs.csv`, gitignored — see below)
- [x] Data exploration (`reports/eda/`) and brand selection
- [x] Conversation construction (`docs/decision_log.md`) + intent taxonomy (`docs/intent_taxonomy.md`)
- [x] Golden evaluation set — 228 hand-labelled examples (`data/golden/`)
- [x] Trivial + TF-IDF baselines (`reports/eval/baseline_*.json`)
- [x] Local TF-IDF retrieval index (`src/support_agent/retrieval.py`)
- [x] Escalation policy, fully rule-based and tested (`src/support_agent/escalation.py`)
- [x] Escalation policy evaluation harness (`reports/eval/escalation_tfidf.json`, no API key needed)
- [x] Failure analysis, misleading-headline-number, one-more-week, decision log (`docs/`)
- [x] `REPORT.md`, `docs/interview_notes.md`
- [x] 32 passing tests (`tests/`)
- [ ] LLM intent classifier — code complete, **pending an OpenRouter API key to run** (see below)
- [ ] Retrieval-grounded reply generation — code complete, pending API key
- [ ] LLM-as-judge + human agreement study — code complete, pending API key

See `REPORT.md` for the full write-up and `README`'s reproduction sections
below for exact commands.

## Getting the dataset

The Kaggle API requires a personal token, so this project downloads the
identical `twcs.csv` (verified by schema and row count) from a public mirror
instead:

```bash
curl -L "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv" -o data/raw/twcs.csv
```

If you have Kaggle credentials and prefer the canonical source:

```bash
pip install kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip
```

Either way you should end up with `data/raw/twcs.csv` (~515 MB, 2,811,774
rows). It is gitignored — not ours to redistribute.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY, see below
```

### Getting an OpenRouter API key

Everything through the TF-IDF baseline, retrieval index, and escalation
policy runs with **no API key at all**. An OpenRouter key
(https://openrouter.ai/keys) is only needed for the LLM intent classifier,
grounded reply generation, and LLM-as-judge steps (scripts 10-14). Put it in
`.env` as `OPENROUTER_API_KEY=sk-or-...` — never commit this file.

## Reproducing results (no API key required) — ~10 minutes, measured end-to-end

Most of this time is loading the 515MB/2.8M-row raw CSV twice (scripts 01
and 03 each do a full pass over it); everything after that is fast.

```bash
python scripts/01_explore_data.py           # ~2-3 min — EDA over the full raw dataset
python scripts/02_select_brand.py           # instant — records the brand-selection decision
python scripts/03_build_conversations.py    # ~3-4 min — builds the conversation dataset + train/eval split
python scripts/04_weak_label_train.py       # ~10s — heuristic labels for the TF-IDF baseline
python scripts/05_sample_golden_candidates.py  # instant — resamples golden-set candidates (already labelled in data/golden/)
python scripts/06_build_golden_set.py       # instant — validates + assembles the golden set
python scripts/07_baseline_trivial.py       # instant — trivial baseline
python scripts/08_baseline_tfidf.py         # ~10-20s — TF-IDF + LogReg baseline
python scripts/09_build_retrieval_index.py  # ~10-20s — builds the TF-IDF retrieval index
python scripts/15_evaluate_escalation_tfidf.py  # ~10s — escalation policy sanity check (TF-IDF driven)
python -m pytest tests/ -q                  # ~15-20s — 32 tests
```

(Measured total on a laptop CPU: 9m25s for the full sequence above.)

## Reproducing results (requires OPENROUTER_API_KEY)

```bash
python scripts/10_run_classifier_eval.py       # LLM intent classifier vs. both baselines
python scripts/11_run_full_pipeline.py         # classify -> retrieve -> escalate -> generate, over the golden set
python scripts/12_run_judge.py                 # LLM-as-judge scores generated replies
python scripts/13_sample_human_review.py       # samples ~50 replies for human rating (manual step, see data/eval/HUMAN_REVIEW_METHODOLOGY.md)
# ... fill in the human_* fields in data/eval/human_review_template.jsonl by hand ...
python scripts/14_compute_judge_agreement.py   # judge-vs-human agreement statistics
```

All LLM calls are cached per-`conversation_id` under `data/llm_cache/`
(gitignored), so re-running any of these after a partial failure does not
re-spend API credits on examples already processed.

## Repository layout

```
data/
  raw/            raw dataset (gitignored)
  processed/      derived conversation dataset + weak labels (gitignored, regenerable)
  golden/         228-example hand-labelled golden set (committed - it's a deliverable)
  eval/           human-review sampling/template for judge agreement (committed once filled)
  embeddings/     TF-IDF retrieval index cache (gitignored, regenerable)
  llm_cache/      cached LLM call results (gitignored, regenerable)
docs/             intent taxonomy, escalation policy, decision log
reports/
  eda/            data-exploration summary
  eval/           baseline/classifier/judge metrics and outputs
scripts/          numbered, run-in-order pipeline scripts
src/support_agent/  importable package - all real logic lives here, scripts are thin orchestrators
tests/            pytest suite
```
