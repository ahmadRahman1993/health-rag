# health-rag

A retrieval-augmented generation (RAG) system over a public medical Q&A corpus, with a built-in evaluation harness.

This project explores a question from production health AI work: how do retrieval strategy and chunking choices affect answer quality in a domain where wrong answers are costly? It pairs a working RAG pipeline with a scenario-based evaluation harness rather than treating evaluation as an afterthought.

## Why this exists

Most RAG demos stop at "it returns an answer." In regulated domains the harder question is "how do you know the answer is good, and how do you measure a change?" This repo treats evaluation as a first-class component, drawing on a published scenario-based evaluation methodology for a domain-constrained clinical assistant.

## Architecture

```
                  ┌──────────────┐
   medical Q&A ──▶ │  ingestion   │  document-aware chunking
   corpus          │  + chunking  │
                  └──────┬───────┘
                         │ chunks
                         ▼
                  ┌──────────────┐
                  │  embedding   │  OpenAI embeddings
                  │  + vectorDB  │  vector store + similarity search
                  └──────┬───────┘
                         │ retriever
                         ▼
                  ┌──────────────┐
   user query ──▶ │  retrieval   │  top-k retrieval
                  │  + generation│  context construction + LLM answer
                  └──────┬───────┘
                         │ answers
                         ▼
                  ┌──────────────┐
                  │  evaluation  │  scenario scoring vs. expected
                  │  harness     │  accuracy + latency
                  └──────────────┘
```

## Stack

- Python 3.11+
- LangChain (orchestration)
- OpenAI API (embeddings + generation)
- A local vector store (FAISS to start; pgvector swap documented later)
- A public medical Q&A dataset (MedQuAD or similar, see `data/README.md`)

## Status

Work in progress. Built in public over one week. See commit history.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

## Usage

```bash
python -m src.ingest        # load + chunk the corpus
python -m src.index         # embed + build the vector store
python -m src.query "What are the symptoms of iron deficiency anemia?"
python -m src.evaluate      # run the evaluation harness
```

## License

MIT
