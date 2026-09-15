"""TF-IDF cosine-similarity retrieval over historically-resolved
conversations.

Originally implemented with sentence-transformers (neural embeddings), but
importing that library crashed the Python process outright in this
environment (Windows fastfail 0xC0000409 - a native heap-corruption abort,
most likely a numpy 2.x ABI mismatch with an older compiled dependency deep
in the sentence-transformers stack). Upgrading/downgrading packages in the
shared conda environment to chase that down risked breaking unrelated
projects, so retrieval uses TF-IDF vectors + cosine similarity instead - a
legitimate "embeddings + vector similarity" approach (TF-IDF vectors are
sparse embeddings), and one already proven stable via the Phase 6 baseline.
See docs/decision_log.md for the full writeup.

Indexes the TRAIN split only (see scripts/03's leakage-prevention split) -
the golden/eval set is never in this index, so a golden example can never
retrieve its own historical resolution.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from support_agent.config import EMBEDDINGS_DIR, PROCESSED_DIR

VECTORIZER_PATH = EMBEDDINGS_DIR / "retrieval_tfidf_vectorizer.joblib"
MATRIX_PATH = EMBEDDINGS_DIR / "retrieval_tfidf_matrix.npz"
METADATA_PATH = EMBEDDINGS_DIR / "retrieval_metadata.jsonl"


def build_index() -> None:
    """Fit a TF-IDF vectorizer over every TRAIN-split customer_message and
    cache the vectorizer + document matrix + metadata. Run via
    scripts/09_build_retrieval_index.py.
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    conv_path = PROCESSED_DIR / "conversations.jsonl"

    records = []
    with conv_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] == "train":
                records.append(rec)

    print(f"Fitting TF-IDF over {len(records):,} train-split customer messages ...")
    texts = [r["customer_message"] for r in records]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    # L2-normalize rows so cosine similarity reduces to a dot product at query time.
    matrix = sparse.csr_matrix(matrix / np.maximum(sparse.linalg.norm(matrix, axis=1), 1e-12)[:, None])

    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, matrix)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(
                    {
                        "conversation_id": rec["conversation_id"],
                        "customer_message": rec["customer_message"],
                        "brand_response": rec["brand_response"],
                        "num_turns": rec["num_turns"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {VECTORIZER_PATH}, {MATRIX_PATH} ({matrix.shape}), {METADATA_PATH}")


class RetrievalIndex:
    """Loads the cached vectorizer/matrix/metadata once; retrieve() is then
    a sparse matrix-vector product (rows are pre-normalized, so the dot
    product is cosine similarity).
    """

    def __init__(self) -> None:
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"{VECTORIZER_PATH} not found. Run scripts/09_build_retrieval_index.py first."
            )
        self.vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)
        self.matrix = sparse.load_npz(MATRIX_PATH)
        self.metadata = [json.loads(line) for line in METADATA_PATH.open(encoding="utf-8")]

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        norm = np.linalg.norm(query_vec.toarray())
        if norm > 0:
            query_vec = query_vec / norm
        sims = (self.matrix @ query_vec.T).toarray().ravel()
        top_idx = np.argsort(-sims)[:k]
        results = []
        for idx in top_idx:
            meta = self.metadata[idx]
            results.append(
                {
                    "conversation_id": meta["conversation_id"],
                    "customer_message": meta["customer_message"],
                    "brand_response": meta["brand_response"],
                    "similarity": round(float(sims[idx]), 4),
                }
            )
        return results
