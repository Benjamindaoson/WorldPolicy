"""Small-sample binomial and paired evaluation helpers."""

from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Invalid success count")
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt((p*(1-p) + z*z/(4*trials)) / trials) / denominator
    return center - radius, center + radius


def paired_counts(a: list[bool], b: list[bool]) -> dict[str, int]:
    if len(a) != len(b) or not a:
        raise ValueError("Paired outcomes require equal nonempty samples")
    return {"both_success": sum(x and y for x, y in zip(a, b, strict=True)),
            "only_a": sum(x and not y for x, y in zip(a, b, strict=True)),
            "only_b": sum(not x and y for x, y in zip(a, b, strict=True)),
            "both_fail": sum(not x and not y for x, y in zip(a, b, strict=True))}


def exact_mcnemar_pvalue(counts: dict[str, int]) -> float:
    """Two-sided exact binomial test on discordant paired outcomes."""
    a, b = counts["only_a"], counts["only_b"]
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a, b) + 1)) / (2**n)
    return min(1.0, 2 * tail)
