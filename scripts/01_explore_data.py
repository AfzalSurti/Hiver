"""Phase 1: Data exploration for the raw Kaggle 'Customer Support on Twitter' dump.

Reads data/raw/twcs.csv (~2.8M tweets) and writes a JSON + markdown summary to
reports/eda/ covering: schema, brand volumes, missing values, duplicates,
conversation-thread structure, and message-length distributions. This is the
evidence base for brand selection in script 02.

Runtime: ~30-60s on a laptop CPU for the full file (single pass, few columns).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from support_agent.config import RAW_TWCS_PATH, REPORTS_DIR, ensure_dirs  # noqa: E402

EDA_DIR = REPORTS_DIR / "eda"


def load_raw() -> pd.DataFrame:
    if not RAW_TWCS_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_TWCS_PATH} not found. Download the dataset first (see README)."
        )
    dtype = {
        "tweet_id": "int64",
        "author_id": "string",
        "inbound": "string",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    }
    df = pd.read_csv(RAW_TWCS_PATH, dtype=dtype, parse_dates=False)
    df["inbound"] = df["inbound"].map({"True": True, "False": False})
    df["created_at_parsed"] = pd.to_datetime(
        df["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )
    return df


def brand_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Count agent-side tweets per brand account (inbound == False)."""
    brand_msgs = df[~df["inbound"]]
    counts = brand_msgs["author_id"].value_counts()
    return counts


def customer_threads_per_brand(df: pd.DataFrame) -> pd.Series:
    """For each brand, count customer tweets that are *replies to* that brand
    (in_response_to_tweet_id points at a tweet authored by the brand), which
    approximates the number of customer-initiated support turns per brand.
    """
    id_to_author = df.set_index("tweet_id")["author_id"]
    id_to_inbound = df.set_index("tweet_id")["inbound"]

    customer_rows = df[df["inbound"]].copy()
    customer_rows["parent_id"] = pd.to_numeric(
        customer_rows["in_response_to_tweet_id"], errors="coerce"
    )
    valid = customer_rows.dropna(subset=["parent_id"])
    valid["parent_id"] = valid["parent_id"].astype("int64")
    valid = valid[valid["parent_id"].isin(id_to_author.index)]
    valid["parent_author"] = valid["parent_id"].map(id_to_author)
    valid["parent_is_brand"] = ~valid["parent_id"].map(id_to_inbound).fillna(True)
    brand_replies = valid[valid["parent_is_brand"]]
    return brand_replies["parent_author"].value_counts()


def build_child_index(df: pd.DataFrame) -> dict:
    """tweet_id -> first child tweet_id that replies to it (O(n) build, O(1) lookup)."""
    parent_ids = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    valid = df.loc[parent_ids.notna(), ["tweet_id", "author_id"]].copy()
    valid["parent_id"] = parent_ids[parent_ids.notna()].astype("int64")
    # Keep first child per parent for a simple linear-chain approximation.
    valid = valid.drop_duplicates(subset="parent_id", keep="first")
    return dict(zip(valid["parent_id"], zip(valid["tweet_id"], valid["author_id"])))


def conversation_root_stats(df: pd.DataFrame, brand: str, child_index: dict, sample_size: int = 20000) -> dict:
    """For a given brand, find conversations where a customer tweet has no
    parent (a fresh complaint) and is later answered by the brand, and report
    how many turns those conversations tend to have. Uses a prebuilt
    tweet_id -> child index for O(1) traversal steps instead of scanning df.
    """
    roots = df[
        df["inbound"]
        & df["in_response_to_tweet_id"].isna()
        & df["text"].str.contains(f"@{brand}", case=False, na=False)
    ]
    roots = roots.head(sample_size)

    lengths = []
    resolved = 0
    for current_id in roots["tweet_id"]:
        chain_len = 1
        seen_brand_reply = False
        cid = current_id
        for _ in range(20):  # cap traversal depth
            child = child_index.get(cid)
            if child is None:
                break
            child_id, child_author = child
            chain_len += 1
            if child_author == brand:
                seen_brand_reply = True
            cid = child_id
        lengths.append(chain_len)
        if seen_brand_reply:
            resolved += 1

    n = len(lengths)
    return {
        "n_root_conversations_sampled": n,
        "pct_with_brand_reply": round(resolved / n, 3) if n else None,
        "avg_chain_length": round(sum(lengths) / n, 2) if n else None,
        "max_chain_length": max(lengths) if lengths else None,
    }


def main() -> None:
    ensure_dirs()
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_TWCS_PATH} ...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows.")

    summary: dict = {}
    summary["n_rows"] = int(len(df))
    summary["n_unique_authors"] = int(df["author_id"].nunique())
    summary["n_inbound_customer_msgs"] = int(df["inbound"].sum())
    summary["n_outbound_brand_msgs"] = int((~df["inbound"]).sum())
    summary["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns if c != "created_at_parsed"}
    summary["duplicate_text_rows"] = int(df["text"].duplicated().sum())
    summary["date_range"] = [
        str(df["created_at_parsed"].min()),
        str(df["created_at_parsed"].max()),
    ]

    print("Computing brand volumes (agent-side message counts)...")
    brand_counts = brand_volume(df)
    top_brands = brand_counts.head(25)
    summary["top_brands_by_agent_messages"] = top_brands.to_dict()

    print("Computing customer-initiated volume per brand (may take ~1 min)...")
    cust_counts = customer_threads_per_brand(df)
    top_customer_brands = cust_counts.head(25)
    summary["top_brands_by_customer_replies_to_them"] = top_customer_brands.to_dict()

    # Message length stats (character count), split by inbound/outbound.
    df["text_len"] = df["text"].str.len()
    summary["text_length_stats"] = {
        "customer": df.loc[df["inbound"], "text_len"].describe().to_dict(),
        "brand": df.loc[~df["inbound"], "text_len"].describe().to_dict(),
    }

    # Candidate brand shortlist: top 8 by both agent volume and customer volume.
    candidates = sorted(
        set(top_brands.index[:15]) & set(top_customer_brands.index[:15])
    )
    print(f"Candidate brands (high volume both sides): {candidates}")

    print("Building parent->child tweet index...")
    child_index = build_child_index(df)

    conv_stats = {}
    for brand in candidates[:8]:
        print(f"  Probing conversation structure for {brand} ...")
        conv_stats[brand] = conversation_root_stats(df, brand, child_index)
    summary["candidate_conversation_stats"] = conv_stats

    out_json = EDA_DIR / "eda_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_json}")

    # Human-readable markdown too.
    lines = ["# Data Exploration Summary\n"]
    lines.append(f"- Total rows: {summary['n_rows']:,}")
    lines.append(f"- Unique authors: {summary['n_unique_authors']:,}")
    lines.append(f"- Customer (inbound) messages: {summary['n_inbound_customer_msgs']:,}")
    lines.append(f"- Brand (outbound) messages: {summary['n_outbound_brand_msgs']:,}")
    lines.append(f"- Duplicate text rows: {summary['duplicate_text_rows']:,}")
    lines.append(f"- Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    lines.append("\n## Top brands by agent-side (outbound) message volume\n")
    for b, c in list(top_brands.items())[:15]:
        lines.append(f"- {b}: {c:,}")
    lines.append("\n## Candidate brands: conversation structure (sampled up to 20k root threads)\n")
    for b, s in conv_stats.items():
        lines.append(
            f"- **{b}**: n_sampled={s['n_root_conversations_sampled']}, "
            f"pct_with_brand_reply={s['pct_with_brand_reply']}, "
            f"avg_chain_len={s['avg_chain_length']}, max_chain_len={s['max_chain_length']}"
        )
    out_md = EDA_DIR / "eda_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
