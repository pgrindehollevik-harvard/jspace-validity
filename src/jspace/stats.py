"""Small dependency-free statistical helpers shared by v1 and v2."""


def mcnemar_exact_p(hits_a: list[bool], hits_b: list[bool]) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    from math import comb

    if len(hits_a) != len(hits_b):
        raise ValueError("paired hit vectors must have equal length")
    b = sum(1 for first, second in zip(hits_a, hits_b) if first and not second)
    c = sum(1 for first, second in zip(hits_a, hits_b) if not first and second)
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, x) for x in range(min(b, c) + 1)) / 2**discordant
    return min(1.0, 2 * tail)
