from collections import Counter

from ..events import Word, norm

EN = ["um", "uh", "like", "you know", "basically", "actually", "so"]
ID = ["eh", "emm", "jadi", "terus", "kayak", "apa ya", "gitu", "nah", "sebenernya"]
DEFAULT_FILLERS = {"en": EN, "id": ID + EN}  # Bahasa speakers mix English fillers in


def find(words: list[Word], fillers: list[str]) -> list[tuple[int, int, str]]:
    """Return (first word index, word count, filler) hits; longest phrase wins."""
    phrases = sorted({tuple(f.lower().split()) for f in fillers if f.strip()}, key=len, reverse=True)
    tokens = [norm(w.text) for w in words]
    hits, i = [], 0
    while i < len(tokens):
        for p in phrases:
            if tuple(tokens[i:i + len(p)]) == p:
                hits.append((i, len(p), " ".join(p)))
                i += len(p)
                break
        else:
            i += 1
    return hits


def summary(words: list[Word], fillers: list[str], minutes: float) -> dict:
    counts = Counter(f for _, _, f in find(words, fillers))
    total = sum(counts.values())
    return {
        "total": total,
        "per_min": round(total / minutes, 1) if minutes > 0 else 0.0,
        "by_word": dict(counts.most_common()),
    }
