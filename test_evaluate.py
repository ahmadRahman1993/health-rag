from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import evaluate
from evaluate import EvalCase, _parse_score, run_eval, score_response, score_response_llm


def test_score_response_perfect_match():
    score = score_response("fever, cough", "The patient has fever and cough.")
    assert score == 1.0


def test_score_response_partial_match():
    score = score_response("fever, cough, headache", "The patient has fever.")
    assert score == pytest.approx(1 / 3)


def test_score_response_no_match():
    score = score_response("fever, cough", "No relevant symptoms mentioned.")
    assert score == 0.0


def test_score_response_empty_expected():
    assert score_response("", "anything") == 0.0


def test_parse_score_plain_number():
    assert _parse_score("0.85") == pytest.approx(0.85)


def test_parse_score_clamps_to_unit_interval():
    assert _parse_score("1.5") == 1.0
    assert _parse_score("-0.2") == 0.0


def test_parse_score_extracts_from_text():
    assert _parse_score("The score is 0.75") == pytest.approx(0.75)


@patch("evaluate.ChatOpenAI")
def test_score_response_llm(mock_chat_cls):
    mock_response = MagicMock()
    mock_response.content = "0.9"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_chat_cls.return_value = mock_llm

    score = score_response_llm(
        "What is flu?",
        "viral, infection",
        "Influenza is a viral disease.",
    )

    assert score == pytest.approx(0.9)
    prompt = mock_llm.invoke.call_args[0][0][0].content
    assert "What is flu?" in prompt
    assert "viral, infection" in prompt


def test_run_eval_aggregates_dual_scores_and_latency():
    cases = [
        EvalCase(
            question="What is flu?",
            expected="viral, infection",
            category="factual",
        ),
        EvalCase(
            question="What is diabetes?",
            expected="does not contain, insufficient",
            category="edge",
        ),
    ]

    def stub_answer(question: str, index_path):
        if "flu" in question.lower():
            return {
                "answer": "Flu is a viral infection.",
                "sources": ["test-1"],
                "context_used": "context",
            }
        return {
            "answer": "The provided context does not contain enough information.",
            "sources": [],
            "context_used": "",
        }

    def stub_llm_score(case: EvalCase, got: str) -> float:
        return 0.9 if case.category == "factual" else 0.5

    with patch.object(evaluate, "GOLDEN_SET", cases), patch.object(
        evaluate, "llm_score_fn", side_effect=stub_llm_score
    ):
        report = run_eval(stub_answer, Path("/tmp/unused"))

    assert report["overall_accuracy"]["keyword"] == pytest.approx(0.75)
    assert report["overall_accuracy"]["llm"] == pytest.approx(0.7)
    assert report["category_accuracy"]["factual"]["keyword"] == 1.0
    assert report["category_accuracy"]["factual"]["llm"] == pytest.approx(0.9)
    assert report["category_accuracy"]["edge"]["keyword"] == pytest.approx(0.5)
    assert report["category_accuracy"]["edge"]["llm"] == pytest.approx(0.5)
    assert len(report["rows"]) == 2
    assert report["p50_latency_ms"] >= 0
    assert report["p95_latency_ms"] >= 0
    assert report["rows"][0]["category"] == "factual"
    assert report["rows"][0]["keyword_score"] == 1.0
    assert report["rows"][0]["llm_score"] == pytest.approx(0.9)
    assert report["rows"][1]["keyword_score"] == pytest.approx(0.5)
    assert report["rows"][1]["llm_score"] == pytest.approx(0.5)


def test_run_eval_raises_on_empty_golden_set():
    with patch.object(evaluate, "GOLDEN_SET", []):
        with pytest.raises(ValueError, match="GOLDEN_SET is empty"):
            run_eval(lambda q, p: {"answer": "", "sources": [], "context_used": ""}, Path("."))
