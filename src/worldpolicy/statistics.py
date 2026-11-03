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
