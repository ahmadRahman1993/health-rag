#Ingestion and chunking.
#Loads the medical Q&A corpus and splits it into retrievable chunks.


from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Chunk:
    """A single retrievable unit."""
    text: str
    source_id: str
    metadata: dict

def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()

def load_corpus(data_dir: Path) -> list[dict]:
    #Load the raw medical Q&A dataset from data_dir.
    raw_dir = data_dir / "raw"
    if not raw_dir.is_dir():
      raise FileNotFoundError(f"Expected MedQuAD XML under {raw_dir}")

    xml_files = sorted(raw_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {raw_dir}")
    
    records: list[dict] = []

    for xml_path in xml_files:
          tree = ET.parse(xml_path)
          root = tree.getroot()
          for qa_pair in root.findall(".//QAPair"):
                    question_el = qa_pair.find("Question")
                    answer_el = qa_pair.find("Answer")
                    question = _text(question_el)
                    answer = _text(answer_el)
                    if not question or not answer:
                      continue
                    record_id = question_el.get("qid") if question_el is not None else None
                    if not record_id:
                      record_id = f"{xml_path.stem}_{qa_pair.get('pid', '0')}"
                    records.append({
                      "id": record_id,
                      "question": question,
                      "answer": answer,
                    })
    return records

def qa_pair_chunking(records: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []

    for record in records:
        text = f"Question: {record['question']}\n\nAnswer: {record['answer']}"
        chunks.append(Chunk(
            text=text,
            source_id=record["id"],
            metadata={
                "chunking": "qa_pair",
                "question": record["question"],
            },
        ))

    return chunks



def fixed_size_chunking(records: list[dict], chunk_size: int = 500,
                        overlap: int = 50) -> list[Chunk]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks: list[Chunk] = []
    global_index = 0

    for record in records:
        text = f"Question: {record['question']}\nAnswer: {record['answer']}"
        for piece in text_splitter.split_text(text):
            if not piece.strip():
                continue
            chunks.append(Chunk(
                text=piece,
                source_id=f"{record['id']}_{global_index}",
                metadata={
                    "chunking": "fixed_size",
                    "record_id": record["id"],
                    "chunk_index": global_index,
                },
            ))
            global_index += 1

    return chunks



if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    records = load_corpus(data_dir)
    print(f"Loaded {len(records)} Q&A pairs")
    print("First record id:", records[0]["id"])
    print("First question:", records[0]["question"][:80], "...")
    print("Answer length:", len(records[0]["answer"]))
    fixed = fixed_size_chunking(records)
    print(f"Fixed-size chunks: {len(fixed)}")
    print(type(fixed[0]))
    print(fixed[0].source_id)
    ids = [c.source_id for c in fixed]
    print("Unique ids:", len(ids) == len(set(ids)))
    qa_chunks = qa_pair_chunking(records)
    print(f"QA-pair chunks: {len(qa_chunks)}")
    print("QA source_id:", qa_chunks[0].source_id)
    print("QA text preview:")
    print(qa_chunks[0].text[:350], "...")
    print(f"\nCompare: records={len(records)}, qa={len(qa_chunks)}, fixed={len(fixed)}")
    assert len(qa_chunks) == len(records)