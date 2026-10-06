"""
Tiny ASCII helpers shared by all the visualizations.

Everything here is plain text so the output works in any terminal, survives
copy-paste, and needs no plotting library.  Callers can swap `fill`/`track`
characters if they prefer a different look.
"""

from __future__ import annotations

from typing import List, Sequence


def clamp(value: float, low: float, high: float) -> float:
    """Keep `value` inside [low, high]."""
    return max(low, min(high, value))


def bar(value: float, max_value: float, width: int = 40, fill: str = "#") -> str:
    """
    A proportional bar for `value` relative to `max_value`.

    Used for scores and sizes; always returns exactly `width` characters.
    """
    if max_value <= 0 or width <= 0:
        return " " * max(width, 0)
    filled = int(round((value / max_value) * width))
    filled = int(clamp(filled, 0, width))
    return fill * filled + " " * (width - filled)


def min_max_bars(scores: Sequence[float], width: int = 40, fill: str = "#") -> List[str]:
    """
    Bars scaled between the small and large ends of `scores`.

    Using the data range (not 0..max) makes small differences visible, which is
    exactly what you want when comparing retrieval scores.
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low == 0:
        return [fill * width for _ in scores]
    return [bar(score - low, high - low, width, fill) for score in scores]


def span_bar(
    start: int,
    end: int,
    total: int,
    width: int = 64,
    fill: str = "#",
) -> str:
    """
    Draw a character span [start, end) on a `width`-column timeline.

    Overlapping spans (chunk overlap!) naturally draw over each other, so the
    picture immediately shows where chunks share text.
    """
    if width <= 0:
        return ""
    total = max(total, 1)
    start_col = int(start / total * width)
    end_col = int(end / total * width)
    start_col = int(clamp(start_col, 0, width - 1))
    end_col = int(clamp(end_col, start_col + 1, width))
    return " " * start_col + fill * (end_col - start_col) + " " * (width - end_col)


def truncate(text: str, width: int, ellipsis: str = "...") -> str:
    """Flatten newlines and cut `text` to `width` characters."""
    text = " ".join(text.split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= len(ellipsis):
        return text[:width]
    return text[: width - len(ellipsis)] + ellipsis


def rule(width: int = 78, char: str = "-") -> str:
    """A horizontal divider line."""
    return char * width


def title(text: str, width: int = 78, char: str = "=") -> str:
    """A centred-ish title with a rule underneath."""
    return f"{text}\n{rule(width, char)}"
