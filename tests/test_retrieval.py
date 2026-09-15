import pytest

from support_agent.retrieval import RetrievalIndex, VECTORIZER_PATH


@pytest.fixture(scope="module")
def index():
    if not VECTORIZER_PATH.exists():
        pytest.skip("Retrieval index not built - run scripts/09_build_retrieval_index.py")
    return RetrievalIndex()


def test_retrieve_returns_requested_k(index):
    results = index.retrieve("I can't log in to my account", k=5)
    assert len(results) == 5


def test_retrieve_results_sorted_by_similarity_descending(index):
    results = index.retrieve("I forgot my password", k=10)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_retrieve_results_have_expected_fields(index):
    results = index.retrieve("refund my subscription please", k=3)
    for r in results:
        assert "conversation_id" in r
        assert "customer_message" in r
        assert "brand_response" in r
        assert "similarity" in r


def test_login_query_retrieves_login_related_results(index):
    results = index.retrieve("I forgot my password and cannot log in", k=5)
    combined_text = " ".join(r["customer_message"].lower() for r in results)
    assert "log" in combined_text or "password" in combined_text
