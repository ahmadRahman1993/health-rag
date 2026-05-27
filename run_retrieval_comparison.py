"""Compare retrieval strategies on the full golden set."""
import argparse
import json
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evaluate import run_eval
from query import answer

INDEX_PATH = Path(__file__).resolve().parent / "index" / "faiss_qa"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONFIGS = {
    "retrieve": ("report_retrieve_k4.json", partial(answer, k=4)),
    "mmr": ("report_mmr_k4.json", partial(answer, k=4, use_mmr=True)),
    "rerank": ("report_rerank_k4.json", partial(answer, k=4, use_rerank=True)),
}


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval strategies.")
    parser.add_argument(
        "--only",
        choices=list(CONFIGS),
        help="Run a single strategy instead of all",
    )
    args = parser.parse_args()

    to_run = {args.only: CONFIGS[args.only]} if args.only else CONFIGS
    summaries = {}

    for label, (filename, answer_fn) in to_run.items():
        print(f"\n{'=' * 60}\nRunning eval: {label} (k=4)\n{'=' * 60}")
        report = run_eval(answer_fn, INDEX_PATH)
        out = RESULTS_DIR / filename
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=4)
        summaries[label] = {
            "file": str(out),
            "keyword": report["overall_accuracy"]["keyword"],
            "llm": report["overall_accuracy"]["llm"],
            "p50_ms": report["p50_latency_ms"],
            "category": report["category_accuracy"],
        }
        print(f"Saved {out}")
        print(f"  keyword={report['overall_accuracy']['keyword']:.3f}")
        print(f"  llm={report['overall_accuracy']['llm']:.3f}")
        print(f"  p50={report['p50_latency_ms']:.0f}ms")

    print(f"\n{'=' * 60}\nComparison summary\n{'=' * 60}")
    for label, s in summaries.items():
        print(f"{label:10} keyword={s['keyword']:.3f}  llm={s['llm']:.3f}  p50={s['p50_ms']:.0f}ms")
        for cat, scores in s["category"].items():
            print(f"  {cat:14} keyword={scores['keyword']:.3f}  llm={scores['llm']:.3f}")


if __name__ == "__main__":
    main()
