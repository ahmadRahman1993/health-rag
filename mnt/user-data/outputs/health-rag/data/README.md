# Data

This project expects a public medical Q&A dataset. Recommended: **MedQuAD**
(Medical Question Answering Dataset) — ~47k Q&A pairs from NIH sources.

## Getting it

MedQuAD is on GitHub (search "MedQuAD"). Clone or download it into `data/raw/`.
It ships as XML per source; you will normalize it in `src/ingest.load_corpus`.

For a faster first iteration, start with a SINGLE subset folder (a few hundred
pairs) so your embed step is cheap while you are still debugging. Scale up
once the pipeline works end to end.

## Expected normalized format

`load_corpus` should return a list of dicts:

```python
{"id": "0000001", "question": "...", "answer": "..."}
```

## Note on cost

Embedding ~47k pairs with OpenAI is inexpensive but not free. Develop on a
few hundred records, only embed the full set once the pipeline is stable.

Raw data is gitignored.
