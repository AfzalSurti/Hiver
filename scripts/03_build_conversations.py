"""Phase 2: Build normalized conversation-level examples for the selected brand.

For every customer-initiated thread addressed to the brand, walks the reply
chain (root customer tweet -> ... -> final brand reply) using a
parent->first-child index (a linear-thread approximation; branching threads
are rare for this dataset as shown in reports/eda/eda_summary.md) and emits:

    {
      "conversation_id": "spotify_<root_tweet_id>",
      "customer_message": "<cleaned first customer turn>",
      "brand_response": "<cleaned final brand turn>",
      "conversation_context": [{"author": "customer"|"brand", "text": ..., "created_at": ...}, ...],
      "timestamp": "<root created_at, ISO>",
      "brand": "SpotifyCares",
      "num_turns": <int>,
      "resolved": <bool, has >=1 brand reply>,
      "split": "train" | "eval"   # conversation-level split, see below
    }

Filtering rules (documented here since they materially shape the dataset):
  1. Root must be an inbound (customer) tweet with no parent, mentioning the
     brand handle, i.e. a fresh complaint/question (not a mid-thread reply).
  2. Root text must have >=3 words after stripping @mentions and URLs -
     excludes contentless pings like "@SpotifyCares" alone.
  3. Root text must be >=85% ASCII by character count - a cheap proxy for
     "primarily English", since golden-set labelling is done by one
     English-speaking annotator. This is a real limitation (documented in
     docs/decision_log.md) - it skews the resulting dataset toward
     English-speaking markets and undercounts non-English complaints.
  4. Exact-duplicate customer_message text is deduplicated (keep first) to
     avoid copy-paste/near-spam threads dominating the corpus.
  5. Conversations with zero brand replies anywhere in the chain are dropped
     entirely - there is no historical resolution to learn from or ground
     replies in for these.

Leakage prevention: each conversation is assigned to "train" or "eval" via a
deterministic hash of its conversation_id (80/20). The "train" split is the
only pool used to build the retrieval index and train the TF-IDF baseline;
the golden evaluation set (Phase 4) is sampled ONLY from "eval". This means
the retrieval/generation system never has direct access to the exact
historical resolution of a message that ends up in the golden set.

Runtime: ~1-2 minutes on the full 2.8M-row file.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import RAW_TWCS_PATH, PROCESSED_DIR, get_selected_brand, ensure_dirs  # noqa: E402

MAX_CHAIN_DEPTH = 20
TRAIN_SPLIT_PCT = 80  # conversation-level, hash-based (see module docstring)

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    # Raw scrape leaves HTML entities un-decoded (e.g. "&gt;", "&amp;") - fix
    # those before stripping mentions/URLs and collapsing whitespace.
    t = html.unescape(text)
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text)


def assign_split(conversation_id: str) -> str:
    h = int(hashlib.md5(conversation_id.encode("utf-8")).hexdigest(), 16)
    return "train" if (h % 100) < TRAIN_SPLIT_PCT else "eval"


def load_raw() -> pd.DataFrame:
    dtype = {
        "tweet_id": "int64",
        "author_id": "string",
        "inbound": "string",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    }
    df = pd.read_csv(RAW_TWCS_PATH, dtype=dtype)
    df["inbound"] = df["inbound"].map({"True": True, "False": False})
    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    return df


def build_child_index(df: pd.DataFrame) -> dict:
    parent_ids = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    mask = parent_ids.notna()
    valid = df.loc[mask, ["tweet_id", "author_id", "text", "inbound", "created_at_parsed"]].copy()
    valid["parent_id"] = parent_ids[mask].astype("int64")
    valid = valid.drop_duplicates(subset="parent_id", keep="first")
    index = {}
    for row in valid.itertuples(index=False):
        index[row.parent_id] = row
    return index


def walk_chain(root_row, child_index: dict) -> list:
    chain = [root_row]
    current = root_row
    for _ in range(MAX_CHAIN_DEPTH):
        nxt = child_index.get(current.tweet_id)
        if nxt is None:
            break
        chain.append(nxt)
        current = nxt
    return chain


def main() -> None:
    ensure_dirs()
    brand = get_selected_brand()
    print(f"Building conversations for brand: {brand}")

    print(f"Loading {RAW_TWCS_PATH} ...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows.")

    print("Building parent->child index...")
    child_index = build_child_index(df)

    roots = df[
        df["inbound"]
        & df["in_response_to_tweet_id"].isna()
        & df["text"].str.contains(f"@{brand}", case=False, na=False)
    ]
    print(f"Found {len(roots):,} candidate root (opening) customer tweets.")

    seen_customer_texts = set()
    records = []
    n_dropped_short = 0
    n_dropped_nonascii = 0
    n_dropped_duplicate = 0
    n_dropped_unresolved = 0

    for root in roots.itertuples(index=False):
        cleaned_root_text = clean_text(root.text)
        if len(cleaned_root_text.split()) < 3:
            n_dropped_short += 1
            continue
        if ascii_ratio(root.text) < 0.85:
            n_dropped_nonascii += 1
            continue
        if cleaned_root_text in seen_customer_texts:
            n_dropped_duplicate += 1
            continue

        chain = walk_chain(root, child_index)
        brand_turns = [t for t in chain if t.author_id == brand]
        if not brand_turns:
            n_dropped_unresolved += 1
            continue

        seen_customer_texts.add(cleaned_root_text)

        conversation_id = f"spotify_{root.tweet_id}"
        context = [
            {
                "author": "brand" if t.author_id == brand else "customer",
                "text": clean_text(t.text),
                "created_at": str(t.created_at_parsed),
            }
            for t in chain
        ]
        final_brand_text = clean_text(brand_turns[-1].text)

        records.append(
            {
                "conversation_id": conversation_id,
                "customer_message": cleaned_root_text,
                "brand_response": final_brand_text,
                "conversation_context": context,
                "timestamp": str(root.created_at_parsed),
                "brand": brand,
                "num_turns": len(chain),
                "resolved": True,
                "split": assign_split(conversation_id),
            }
        )

    print(f"Built {len(records):,} conversation examples.")
    print(
        f"Dropped: short={n_dropped_short:,}, non_ascii={n_dropped_nonascii:,}, "
        f"duplicate={n_dropped_duplicate:,}, unresolved={n_dropped_unresolved:,}"
    )

    n_train = sum(1 for r in records if r["split"] == "train")
    n_eval = len(records) - n_train
    print(f"Split: train={n_train:,} ({n_train / len(records):.1%}), eval={n_eval:,} ({n_eval / len(records):.1%})")

    out_path = PROCESSED_DIR / "conversations.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
