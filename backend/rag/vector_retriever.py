
import math


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    if not vector_a or not vector_b:
        raise ValueError("Vectors cannot be empty")

    if len(vector_a) != len(vector_b):
        raise ValueError(
            "Vectors must have the same dimensions"
        )

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Vectors cannot be zero")

    return dot_product / (
        magnitude_a * magnitude_b
    )


if __name__ == "__main__":

    vector_a = [1.0, 2.0, 3.0]
    vector_b = [2.0, 4.0, 6.0]
    vector_c = [-1.0, -2.0, -3.0]

    print(
        "Similarity A-B:",
        cosine_similarity(vector_a, vector_b)
    )

    print(
        "Similarity A-C:",
        cosine_similarity(vector_a, vector_c)
    )