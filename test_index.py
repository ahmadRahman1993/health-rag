from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_community.embeddings import FakeEmbeddings

from ingest import Chunk
from index import build_index, load_index


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            text="Question: What are ALL symptoms?\n\nAnswer: fever, tiredness, bruising",
            source_id="test-1",
            metadata={"chunking": "qa_pair", "question": "What are ALL symptoms?"},
        ),
        Chunk(
            text="Question: What is type 2 diabetes?\n\nAnswer: a blood sugar disorder",
            source_id="test-2",
            metadata={"chunking": "qa_pair", "question": "What is type 2 diabetes?"},
        ),
    ]


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings(size=1536)


@patch("index.OpenAIEmbeddings")
def test_build_index_creates_files(mock_openai_cls, tmp_path, sample_chunks, fake_embeddings):
    mock_openai_cls.return_value = fake_embeddings
    index_path = tmp_path / "faiss_test"

    build_index(sample_chunks, index_path)

    assert index_path.is_dir()
    assert any(index_path.iterdir()), "FAISS should write files to disk"


@patch("index.OpenAIEmbeddings")
def test_build_index_vector_count(mock_openai_cls, tmp_path, sample_chunks, fake_embeddings):
    mock_openai_cls.return_value = fake_embeddings
    index_path = tmp_path / "faiss_test"

    store = build_index(sample_chunks, index_path)

    assert store.index.ntotal == len(sample_chunks)


@patch("index.OpenAIEmbeddings")
def test_load_index_roundtrip(mock_openai_cls, tmp_path, sample_chunks, fake_embeddings):
    mock_openai_cls.return_value = fake_embeddings
    index_path = tmp_path / "faiss_test"

    build_index(sample_chunks, index_path)
    store = load_index(index_path)

    assert store.index.ntotal == len(sample_chunks)


@patch("index.OpenAIEmbeddings")
def test_metadata_preserved_after_load(mock_openai_cls, tmp_path, sample_chunks, fake_embeddings):
    mock_openai_cls.return_value = fake_embeddings
    index_path = tmp_path / "faiss_test"

    build_index(sample_chunks, index_path)
    store = load_index(index_path)

    results = store.similarity_search("anything", k=2)
    by_source = {doc.metadata["source_id"]: doc for doc in results}

    assert set(by_source) == {"test-1", "test-2"}
    assert by_source["test-1"].metadata["chunking"] == "qa_pair"
    assert by_source["test-1"].metadata["question"] == "What are ALL symptoms?"
    assert by_source["test-2"].metadata["chunking"] == "qa_pair"
    assert "fever" in by_source["test-1"].page_content.lower()

@patch("index.OpenAIEmbeddings")
def test_similarity_search_returns_results(mock_openai_cls, tmp_path, sample_chunks, fake_embeddings):
    mock_openai_cls.return_value = fake_embeddings
    index_path = tmp_path / "faiss_test"

    build_index(sample_chunks, index_path)
    store = load_index(index_path)

    results = store.similarity_search("symptoms of ALL leukemia", k=1)

    assert len(results) == 1
    assert "source_id" in results[0].metadata
    assert results[0].page_content.startswith("Question:")