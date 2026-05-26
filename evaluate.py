"""
Evaluation harness
"""

import time
from dataclasses import dataclass
from pathlib import Path
import json
from query import answer
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import re

@dataclass
class EvalCase:
    question: str
    expected: str          # reference answer or key facts
    category: str          # e.g. "factual", "multi-source", "edge"

GOLDEN_SET: list[EvalCase] = [
    # --- factual (single-chunk lookup) ---

    EvalCase(
        question="What are the symptoms of Adult Acute Lymphoblastic Leukemia?",
        expected="fever, tired, bruising, bleeding, petechiae, shortness of breath, weight loss, bone pain, infections",
        category="factual",
    ),  # source: 0000001_1-2

    EvalCase(
        question="What is Adult Acute Lymphoblastic Leukemia?",
        expected="cancer, bone marrow, lymphocytes, white blood cell, blood",
        category="factual",
    ),  # source: 0000001_1-1

    EvalCase(
        question="How to diagnose Adult Acute Lymphoblastic Leukemia?",
        expected="blood, bone marrow, complete blood count, biopsy, physical exam",
        category="factual",
    ),  # source: 0000001_1-3

    EvalCase(
        question="Who is at risk for Adult Acute Lymphoblastic Leukemia?",
        expected="chemotherapy, radiation, Down syndrome, male, older than 70",
        category="factual",
    ),  # source: 0000001_1-5

    EvalCase(
        question="What are the symptoms of Adult Acute Myeloid Leukemia?",
        expected="fever, tired, bruising, bleeding, shortness of breath, weight loss, petechiae",
        category="factual",
    ),  # source: 0000001_2-3

    EvalCase(
        question="What are the symptoms of Chronic Lymphocytic Leukemia?",
        expected="swollen lymph nodes, tired, fever, infection, weight loss",
        category="factual",
    ),  # source: 0000001_3-3

    EvalCase(
        question="What are the genetic changes related to Chronic Myelogenous Leukemia?",
        expected="Philadelphia chromosome, gene mutation, tyrosine kinase, chromosome",
        category="factual",
    ),  # source: 0000001_4-3

    EvalCase(
        question="What are the symptoms of Hairy Cell Leukemia?",
        expected="infections, tired, bruising, bleeding, shortness of breath, weight loss, below the ribs",
        category="factual",
    ),  # source: 0000001_5-3

    # --- multi-source (needs 2+ chunks) ---

    EvalCase(
        question="What are the symptoms and treatments for Adult Acute Lymphoblastic Leukemia?",
        expected="fever, bruising, chemotherapy, radiation, remission, stem cell transplant, targeted therapy",
        category="multi-source",
    ),  # 0000001_1-2 + 0000001_1-7

    EvalCase(
        question="What are the symptoms of Chronic Myelogenous Leukemia and what genetic change is associated with it?",
        expected="fever, night sweats, tired, Philadelphia chromosome, gene mutation",
        category="multi-source",
    ),  # 0000001_4-2 + 0000001_4-3

    EvalCase(
        question="How is Hairy Cell Leukemia diagnosed and what treatments are used?",
        expected="bone marrow, blood, biopsy, chemotherapy, watchful waiting, targeted therapy",
        category="multi-source",
    ),  # 0000001_5-4 + 0000001_5-8

    EvalCase(
        question="What is Chronic Lymphocytic Leukemia and who is at risk for it?",
        expected="lymphocytes, bone marrow, older, middle-aged, male, white, family history",
        category="multi-source",
    ),  # 0000001_3-1 + 0000001_3-2

    # --- edge (NOT in corpus — should refuse) ---

    EvalCase(
        question="What is type 2 diabetes?",
        expected="does not contain, insufficient, not enough information, cannot answer, provided context",
        category="edge",
    ),

    EvalCase(
        question="What are the symptoms of iron deficiency anemia?",
        expected="does not contain, insufficient, not enough information, cannot answer, provided context",
        category="edge",
    ),

    EvalCase(
        question="What are the treatments for breast cancer?",
        expected="does not contain, insufficient, not enough information, cannot answer, provided context",
        category="edge",
    ),
]


def score_response(expected: str, got: str) -> float:
    """
    Score a single response in [0, 1].
    """
    if not expected.strip():
        return 0.0
    
    keywords = [k.strip().lower() for k in expected.split(",") if k.strip()]
    if not keywords:
        return 0.0
    
    got_lower = got.lower()

    hits = sum(1 for k in keywords if k in got_lower)
    total = len(keywords)

    return hits / total if total > 0 else 0.0



def _parse_score(text: str) -> float:
    text = text.strip()
    try:
        return max(0.0, min(1.0, float(text.split()[0])))
    except (ValueError, IndexError):
        match = re.search(r"\b(1\.0|0\.\d+|1|0)\b", text)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return 0.0

def score_response_llm(question: str, expected: str, got: str) -> float:
    """
    Score a single response in [0, 1] using an LLM.
    """
    if not expected.strip():
        return 0.0
    
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""You are grading answers from a medical Q&A retrieval system.
    Question: {question}
    Expected key facts: {expected}
    Model answer: {got}
    Rate how well the model answer covers the expected key facts.
    - Accept paraphrases and synonyms.
    - If expected facts indicate refusal (e.g. "does not contain"), score 1.0 if the model appropriately refuses.
    - Return ONLY a number between 0.0 and 1.0.
    """

    response = llm.invoke([
        HumanMessage(content=prompt),
    ])

    return _parse_score(response.content)

def llm_score_fn(case: EvalCase, got: str) -> float:
    return score_response_llm(case.question,case.expected, got)

def keyword_score_fn(case: EvalCase, got: str) -> float:
    return score_response(case.expected, got)

def run_eval(answer_fn, index_path) -> dict:
    """
    Run the full golden set, collect accuracy and latency.
    """
    if not GOLDEN_SET:
        raise ValueError("GOLDEN_SET is empty — add EvalCase entries first.")
    rows = []
    latencies_ms = []

    for case in GOLDEN_SET[:5]:
        start = time.perf_counter()
        result = answer_fn(case.question, index_path)
        elapsed = (time.perf_counter() - start) * 1000

        got = result["answer"]
        keyword_score = keyword_score_fn(case, got)
        llm_score = llm_score_fn(case, got)

        latencies_ms.append(elapsed)

        rows.append({
            "question": case.question,
            "category": case.category,
            "keyword_score": keyword_score,
            "llm_score": llm_score,
            "latency_ms": elapsed,
            "answer_preview": got[:120]
        })

    overall_accuracy = {
        "keyword": sum(row["keyword_score"] for row in rows) / len(rows),
        "llm": sum(row["llm_score"] for row in rows) / len(rows)
    }


    by_category: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"keyword": [], "llm": []}
    )

    for row in rows:
            by_category[row["category"]]["keyword"].append(row["keyword_score"])
            by_category[row["category"]]["llm"].append(row["llm_score"])
            
    category_accuracy = {
            cat: {
                "keyword": sum(scores["keyword"]) / len(scores["keyword"]),
                "llm": sum(scores["llm"]) / len(scores["llm"])
            }
            for cat, scores in by_category.items()
    }

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)
        
    def percentile(p:float) -> float:
            idx = int((n - 1) * p)
            return sorted_latencies[idx]

    p50 = percentile(0.50)
    p95 = percentile(0.95)

    report = {
            "overall_accuracy": overall_accuracy,
            "category_accuracy": category_accuracy,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "rows": rows,
    }
    
    return report



if __name__ == "__main__":
    index_path = Path(__file__).resolve().parent / "index" / "faiss_qa"

    report = run_eval(answer, index_path)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "report.json"
    with open(results_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Results saved to {results_path}")
