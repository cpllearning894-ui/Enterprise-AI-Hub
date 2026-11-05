
from backend.rag.answer_evaluator import evaluate_answer


def test_evaluate_answer_with_matching_keywords():

    question = "Who approves software installation?"

    answer = "The IT department approves software installation."

    context = (
        "Software installation requires approval "
        "from the IT department."
    )

    result = evaluate_answer(
        question,
        answer,
        context
    )

    assert result["faithfulness"] == 0.75
    assert result["answer_relevance"] == 1.0
    assert result["context_relevance"] == 0.6667


def test_evaluate_answer_with_no_matching_keywords():

    question = "What is the vacation policy?"

    answer = "Python is a programming language."

    context = "The server requires regular maintenance."

    result = evaluate_answer(
        question,
        answer,
        context
    )

    assert result["faithfulness"] == 0.0
    assert result["answer_relevance"] == 0.0
    assert result["context_relevance"] == 0.0


def test_evaluation_method():

    result = evaluate_answer(
        "Who approves installation?",
        "IT approves installation.",
        "IT approves software installation."
    )

    assert result["evaluation_method"] == (
        "keyword_overlap_baseline"
    )


def test_empty_text():

    result = evaluate_answer(
        "",
        "",
        ""
    )

    assert result["faithfulness"] == 0.0
    assert result["answer_relevance"] == 0.0
    assert result["context_relevance"] == 0.0