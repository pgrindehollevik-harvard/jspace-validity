"""Output-quality metrics used by the registered coherence check."""


def max_ngram_repetition(token_ids: list[int], n: int = 4) -> float:
    if len(token_ids) < n * 2:
        return 0.0
    grams = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
    counts: dict[tuple[int, ...], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) * n / len(token_ids)
