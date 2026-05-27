from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from query import SYSTEM_PROMPT, answer, retrieve, retrieve_mmr, retrieve_with_rerank


@patch("query.load_index")
def test_retrieve_calls_similarity_search_with_score(mock_load_index, tmp_path):
    doc = Document(
        page_content="Question: What is flu?\n\nAnswer: A viral infection.",
        metadata={"source_id": "test-1"},
    )
    mock_store = MagicMock()
    mock_store.similarity_search_with_score.return_value = [(doc, 0.42)]
    mock_load_index.return_value = mock_store

    results = retrieve("flu symptoms", tmp_path, k=3)

    mock_load_index.assert_called_once_with(Path(tmp_path))
    mock_store.similarity_search_with_score.assert_called_once_with("flu symptoms", k=3)
    assert len(results) == 1
    assert results[0][0].metadata["source_id"] == "test-1"
    assert results[0][1] == 0.42

@patch("query.load_index")
def test_get_vectorstore_caches(mock_load_index, tmp_path):
    from query import get_vectorstore, _store_cache

    _store_cache.clear()  # fresh cache for test
    mock_store = MagicMock()
    mock_load_index.return_value = mock_store

    get_vectorstore(tmp_path)
    get_vectorstore(tmp_path)

    mock_load_index.assert_called_once()


@patch("query.load_index")
def test_retrieve_mmr_calls_max_marginal_relevance_search(mock_load_index, tmp_path):
    doc = Document(
        page_content="Question: What is flu?\n\nAnswer: A viral infection.",
        metadata={"source_id": "test-1"},
    )
    mock_store = MagicMock()
    mock_store.max_marginal_relevance_search.return_value = [doc]
    mock_load_index.return_value = mock_store

    results = retrieve_mmr("flu symptoms", tmp_path, k=2, fetch_k=10, diversity=0.5)

    mock_store.max_marginal_relevance_search.assert_called_once_with(
        "flu symptoms",
        k=2,
        fetch_k=10,
        lambda_mult=0.5,
    )
    assert len(results) == 1
    assert results[0][0].metadata["source_id"] == "test-1"
    assert results[0][1] == 0.0


@patch("query.ChatOpenAI")
@patch("query.retrieve_mmr")
def test_answer_uses_mmr_when_flag_set(mock_retrieve_mmr, mock_chat_cls, tmp_path):
    doc = Document(
        page_content="Question: What is flu?\n\nAnswer: A viral infection.",
        metadata={"source_id": "test-1"},
    )
    mock_retrieve_mmr.return_value = [(doc, 0.0)]

    mock_response = MagicMock()
    mock_response.content = "A viral infection."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_llm

    answer("What is flu?", tmp_path, k=2, use_mmr=True)

    mock_retrieve_mmr.assert_called_once_with("What is flu?", tmp_path, k=2)


@patch("query.CohereRerank")
@patch("query.load_index")
def test_retrieve_with_rerank_uses_cohere(mock_load_index, mock_rerank_cls, tmp_path):
    doc_a = Document(page_content="symptoms", metadata={"source_id": "a"})
    doc_b = Document(
        page_content="treatment",
        metadata={"source_id": "b", "relevance_score": 0.95},
    )
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [doc_a, doc_b]
    mock_load_index.return_value = mock_store

    mock_rerank = MagicMock()
    mock_rerank.compress_documents.return_value = [doc_b]
    mock_rerank_cls.return_value = mock_rerank

    results = retrieve_with_rerank("symptoms and treatment", tmp_path, k=1, fetch_k=10)

    mock_store.similarity_search.assert_called_once_with("symptoms and treatment", k=10)
    mock_rerank_cls.assert_called_once_with(model="rerank-english-v3.0", top_n=1)
    mock_rerank.compress_documents.assert_called_once()
    assert results[0][0].metadata["source_id"] == "b"
    assert results[0][1] == 0.95


@patch("query.ChatOpenAI")
@patch("query.retrieve_with_rerank")
def test_answer_uses_rerank_when_flag_set(mock_retrieve_rerank, mock_chat_cls, tmp_path):
    doc = Document(
        page_content="Question: What is flu?\n\nAnswer: A viral infection.",
        metadata={"source_id": "test-1"},
    )
    mock_retrieve_rerank.return_value = [(doc, 0.95)]

    mock_response = MagicMock()
    mock_response.content = "A viral infection."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_llm

    answer("What is flu?", tmp_path, k=2, use_rerank=True)

    mock_retrieve_rerank.assert_called_once_with("What is flu?", tmp_path, k=2)


@patch("query.ChatOpenAI")
@patch("query.retrieve")
def test_answer_returns_structured_result(mock_retrieve, mock_chat_cls, tmp_path):
    doc = Document(
        page_content="Question: What is flu?\n\nAnswer: A viral infection.",
        metadata={"source_id": "test-1"},
    )
    mock_retrieve.return_value = [(doc, 0.42)]

    mock_response = MagicMock()
    mock_response.content = "A viral infection."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_llm

    result = answer("What is flu?", tmp_path, k=2)

    assert set(result.keys()) == {"answer", "sources", "context_used"}
    assert result["answer"] == "A viral infection."
    assert result["sources"] == ["test-1"]
    assert "test-1" in result["context_used"]
    assert "A viral infection." in result["context_used"]


@patch("query.ChatOpenAI")
@patch("query.retrieve")
def test_answer_deduplicates_sources(mock_retrieve, mock_chat_cls, tmp_path):
    doc = Document(
        page_content="chunk text",
        metadata={"source_id": "same-id"},
    )
    mock_retrieve.return_value = [(doc, 0.1), (doc, 0.2)]

    mock_response = MagicMock()
    mock_response.content = "answer"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_llm

    result = answer("question?", tmp_path)

    assert result["sources"] == ["same-id"]


@patch("query.ChatOpenAI")
@patch("query.retrieve")
def test_answer_passes_system_prompt_to_llm(mock_retrieve, mock_chat_cls, tmp_path):
    doc = Document(page_content="context", metadata={"source_id": "test-1"})
    mock_retrieve.return_value = [(doc, 0.5)]

    mock_response = MagicMock()
    mock_response.content = "answer"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_llm

    answer("What is flu?", tmp_path)

    messages = mock_llm.invoke.call_args[0][0]
    assert messages[0].content == SYSTEM_PROMPT
    assert "What is flu?" in messages[1].content
