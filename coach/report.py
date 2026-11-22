from datetime import datetime
from typing import TYPE_CHECKING

from .analyzers import fillers, pace
from .config import PACE_BAND

if TYPE_CHECKING:
    from .session import Session

HESITATION_GAP = 0.6  # a filler ending this close to a pause start counts as a hesitation


def build_report(s: "Session") -> dict:
    words, duration = s.finals, s.audio_t
    hits = fillers.find(words, s.fillers)
    filler_idx = {i + k for i, n, _ in hits for k in range(n)}
    filler_ends = [words[i + n - 1].end for i, n, _ in hits]
    hesitations = sum(
        any(abs(end - p["start"]) <= HESITATION_GAP for end in filler_ends) for p in s.pauses.items
    )
    speaking_span = (words[-1].end - words[0].start) if words else 0.0
    wpm, sps = pace.rate(words, words[0].start, words[-1].end + 1e-6, s.lang) if speaking_span > 0 else (0, 0)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "lang": s.lang,
        "duration": round(duration, 1),
        "word_count": len(words),
        "avg_wpm": round(wpm),
        "avg_sps": round(sps, 1),
        "pace_band": list(PACE_BAND),
        "pace_timeline": pace.timeline(words, duration, s.lang),
        "fillers": fillers.summary(words, s.fillers, duration / 60),
        "pauses": {"counts": s.pauses.counts(), "items": s.pauses.items},
        "hesitations": hesitations,
        "rtf": round(s.compute_s / duration, 3) if duration else None,
        "transcript": [
            {"text": w.text, "start": w.start, "end": w.end, "conf": w.conf, "filler": i in filler_idx}
            for i, w in enumerate(words)
        ],
    }


def to_markdown(r: dict) -> str:
    f, p = r["fillers"], r["pauses"]["counts"]
    lines = [
        f"# Rehearsal report - {r['created_at']}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Duration | {int(r['duration'] // 60)}:{int(r['duration'] % 60):02d} |",
        f"| Words | {r['word_count']} |",
        f"| Average pace | {r['avg_wpm']} wpm ({r['avg_sps']} syllables/s), target {r['pace_band'][0]}-{r['pace_band'][1]} wpm |",
        f"| Fillers | {f['total']} ({f['per_min']}/min) |",
        f"| Pauses | {p['good']} good, {p['long']} long, {p['freeze']} freezes |",
        f"| Hesitations (filler + pause) | {r['hesitations']} |",
        "",
        "## Fillers",
        "",
        *[f"- {word}: {n}" for word, n in f["by_word"].items()],
        "",
        "## Pauses over 2 s",
        "",
        *[f"- {x['start']:.1f}s - {x['end']:.1f}s ({x['kind']})" for x in r["pauses"]["items"] if x["kind"] != "good"],
        "",
        "## Transcript (fillers in **bold**)",
        "",
        " ".join(f"**{w['text']}**" if w["filler"] else w["text"] for w in r["transcript"]),
        "",
    ]
    return "\n".join(lines)
