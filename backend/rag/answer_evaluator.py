
import re


# Words that do not provide much useful meaning
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "to", "of", "in", "on", "at", "for", "and",
    "or", "but", "with", "by", "from", "it",
    "this", "that", "these", "those", "what",
    "which", "who", "when", "where", "how",
    "does", "do", "did", "can", "could",
    "should", "would", "will", "may", "might"
}


def extract_terms(text: str) -> set:
    """
    Extract meaningful words from text.
    """

    words = re.findall(r"\b[a-z0-9]+\b", text.lower())

    return {
        word for word in words
        if word not in STOPWORDS
    }


def calculate_overlap(expected_terms: set, actual_terms: set) -> float:
    """
    Calculate the percentage of expected terms
    that appear in the actual text.
    """

    if not expected_terms:
        return 0.0

    matched_terms = expected_terms.intersection(actual_terms)

    score = len(matched_terms) / len(expected_terms)

    return round(score, 4)


def evaluate_answer(
    question: str,
    answer: str,
    context: str
) -> dict:
    """
    Evaluate an AI answer using keyword overlap.

    This is a simple baseline, not a true
    semantic or factual evaluation.
    """

    question_terms = extract_terms(question)
    answer_terms = extract_terms(answer)
    context_terms = extract_terms(context)

    # 1. Faithfulness:
    # How many answer terms appear in the context?
    faithfulness = calculate_overlap(
        answer_terms,
        context_terms
    )

    # 2. Answer Relevance:
    # How many question terms appear in the answer?
    answer_relevance = calculate_overlap(
        question_terms,
        answer_terms
    )

    # 3. Context Relevance:
    # How many question terms appear in the context?
    context_relevance = calculate_overlap(
        question_terms,
        context_terms
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_relevance": context_relevance,
        "evaluation_method": "keyword_overlap_baseline"
    }