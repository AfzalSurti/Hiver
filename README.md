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

This repository is under active development. This README will be updated
with exact reproduction commands and expected runtime as each phase lands.
Current state:

- [x] Repo scaffolding, `.env.example`, `.gitignore`
- [x] Dataset acquired (`data/raw/twcs.csv`, gitignored — see below)
- [x] Data exploration (`reports/eda/`) and brand selection
- [ ] Conversation construction + intent taxonomy
- [ ] Golden evaluation set (150-250 examples)
- [ ] Trivial + TF-IDF baselines
- [ ] LLM intent classifier
- [ ] Retrieval-grounded reply generation
- [ ] Escalation policy
- [ ] LLM-as-judge + human agreement study
- [ ] Final evaluation, failure analysis, REPORT.md

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
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

## Reproducing results so far

```bash
python scripts/01_explore_data.py   # ~30-60s, writes reports/eda/
python scripts/02_select_brand.py   # instant, writes data/processed/selected_brand.txt
```
