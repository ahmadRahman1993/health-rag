#Ingestion and chunking.
#Loads the medical Q&A corpus and splits it into retrievable chunks.


from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


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



if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data"
    records = load_corpus(data_dir)
    print(f"Loaded {len(records)} Q&A pairs")
    print("First record id:", records[0]["id"])
    print("First question:", records[0]["question"][:80], "...")
    print("Answer length:", len(records[0]["answer"]))