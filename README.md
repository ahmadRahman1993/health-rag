# health-rag

A retrieval-augmented generation (RAG) system over a medical Q&A corpus, with a scenario-based evaluation harness.

This project asks a question that matters in production health AI: **how do retrieval and chunking choices affect answer quality when wrong answers are costly?** It pairs a working RAG pipeline with measurable evaluation rather than treating quality as an afterthought.

## What it does

- **Ingests** MedQuAD-style XML Q&A pairs with two chunking strategies (QA-pair vs fixed-size)
- **Indexes** chunks with OpenAI embeddings into a local FAISS store
- **Answers** questions with grounded generation (`gpt-4o-mini`), using only retrieved context
- **Evaluates** a 15-case golden set (factual, multi-source, edge/refusal) with keyword and LLM-as-judge scoring
- **Compares** retrieval strategies: plain similarity, MMR, and Cohere rerank

## Architecture

```
                  ┌──────────────┐
   medical Q&A ──▶ │  ingestion   │  QA-pair + fixed-size chunking
   corpus          │  + chunking  │
                  └──────┬───────┘
                         │ chunks
                         ▼
                  ┌──────────────┐
                  │  embedding   │  OpenAI text-embedding-3-small
                  │  + FAISS     │
                  └──────┬───────┘
                         │ retriever (similarity / MMR / rerank)
                         ▼
                  ┌──────────────┐
   user query ──▶ │  retrieval   │  top-k + optional reranking
                  │  + generation│  context + LLM answer
                  └──────┬───────┘
                         │ answers
                         ▼
                  ┌──────────────┐
                  │  evaluation  │  golden set, dual scoring, latency
                  │  harness     │
                  └──────────────┘
```

## Stack

- Python 3.11+
- LangChain (orchestration)
- OpenAI API (embeddings + generation + eval judge)
- Cohere API (reranking)
- FAISS (local vector store)
- MedQuAD subset — 5 leukemia XML files under `data/raw/1_CancerGov_QA/` (~38 Q&A pairs)

## Project layout

```
health-rag/
  ingest.py                  # load corpus, chunking strategies
  index.py                   # embed + build/load FAISS
  query.py                   # retrieve, rerank, answer
  evaluate.py                # golden set + run_eval()
  run_retrieval_comparison.py
  test_*.py                  # pytest suite
  data/raw/                  # MedQuAD XML (not committed; see Setup)
  index/faiss_qa/            # built index (not committed)
  results/                   # eval reports (not committed)
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install langchain langchain-openai langchain-community langchain-cohere \
            langchain-text-splitters faiss-cpu openai cohere python-dotenv pytest

Create `.env` in the project root:

```
OPENAI_API_KEY=...
COHERE_API_KEY=...       # required for reranking
```

Place MedQuAD XML files under `data/raw/` (e.g. clone [MedQuAD](https://github.com/abachaa/MedQuAD) and copy a subset folder).

## Usage

Build the pipeline:

```bash
python index.py          # load corpus, chunk (qa_pair), embed, save FAISS
```

Query:

```bash
python query.py "What are the symptoms of Adult Acute Lymphoblastic Leukemia?"
```

Run evaluation (15 golden-set cases; saves `results/report.json`):

```bash
python evaluate.py
```

Compare retrieval strategies (plain / MMR / Cohere rerank at k=4):

```bash
python run_retrieval_comparison.py
python run_retrieval_comparison.py --only rerank
```

Run tests:

```bash
pytest -v
```

### Retrieval options in code

```python
from functools import partial
from query import answer

answer(q, index_path)                              # plain similarity (default k=2)
answer(q, index_path, k=4)                         # plain, k=4
answer(q, index_path, k=4, use_mmr=True)           # Max Marginal Relevance
answer(q, index_path, k=4, use_rerank=True)       # Cohere rerank
```

## Evaluation results

On the 15-case golden set (k=4), comparing retrieval strategies:

| Strategy | Keyword accuracy | LLM judge | p50 latency |
|----------|------------------|-----------|-------------|
| Plain similarity | 0.873 | 0.980 | ~2.5s |
| MMR | 0.758 | 0.907 | ~2.3s |
| **Cohere rerank** | 0.863 | **0.987** | ~3.0s |

**Findings:**

- **Plain similarity** is a strong baseline on this small leukemia corpus.
- **MMR** hurt multi-source cases by diversifying across similar diseases (ALL vs AML vs CLL) instead of keeping related chunks from one condition.
- **Cohere rerank** gave the best LLM-judged quality (0.987), improving multi-source answers without MMR's cross-disease problem, at the cost of one extra API call per query.

Reports are saved under `results/` (e.g. `report_retrieve_k4.json`, `report_mmr_k4.json`, `report_rerank_k4.json`).

## Golden set categories

| Category | Count | What it tests |
|----------|-------|---------------|
| `factual` | 8 | Single-chunk lookup from corpus |
| `multi-source` | 4 | Combining info from 2+ chunks |
| `edge` | 3 | Questions outside corpus — should refuse |

Scoring uses both **keyword overlap** (fast, deterministic) and **LLM-as-judge** (handles paraphrases).

## Status

Complete mini-project: ingest → index → query → evaluate, with retrieval strategy comparison and test coverage.

## License

MIT
