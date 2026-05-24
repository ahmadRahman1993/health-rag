from pathlib import Path
import pytest
from ingest import Chunk, fixed_size_chunking, load_corpus, qa_pair_chunking


def test_chunk_dataclass_holds_metadata():
    c = Chunk(text="q + a", source_id="0001", metadata={"category": "factual"})
    assert c.source_id == "0001"
    assert c.metadata["category"] == "factual"


def test_qa_pair_chunking_produces_one_chunk_per_record():
    records = [
        {"id": "1", "question": "What is flu?", "answer": "A viral infection."},
        {"id": "2", "question": "What is iron?", "answer": "A mineral."},
    ]

    chunks = qa_pair_chunking(records)

    assert len(chunks) == len(records)
    assert [chunk.source_id for chunk in chunks] == ["1", "2"]


def test_fixed_size_chunking_respects_chunk_size():
    records = [
        {
            "id": "1",
            "question": "Q" * 30,
            "answer": "A" * 120,
        }
    ]

    chunks = fixed_size_chunking(records, chunk_size=50, overlap=0)

    assert chunks
    assert all(len(chunk.text) <= 50 for chunk in chunks)

def test_load_corpus_parses_qa_pairs(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "sample"
    raw_dir.mkdir(parents=True)

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="test_1">
  <QAPairs>
    <QAPair pid="1">
      <Question qid="test_1-1" qtype="information">What is flu?</Question>
      <Answer>A viral infection.</Answer>
    </QAPair>
    <QAPair pid="2">
      <Question qid="test_1-2" qtype="symptoms">What are flu symptoms?</Question>
      <Answer>Fever and cough.</Answer>
    </QAPair>
  </QAPairs>
</Document>
"""
    (raw_dir / "test_1.xml").write_text(xml_content, encoding="utf-8")

    records = load_corpus(tmp_path)

    assert len(records) == 2
    assert records[0] == {
        "id": "test_1-1",
        "question": "What is flu?",
        "answer": "A viral infection.",
    }
    assert records[1]["id"] == "test_1-2"