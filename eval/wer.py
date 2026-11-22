"""WER / CER with one fixed set of normalisation rules, used for every experiment.

Rules: lowercase; hyphens become spaces; punctuation removed (apostrophes kept);
whitespace collapsed. With --drop-fillers, filled pauses (um, uh, ...) are removed
from both sides, so each run can be scored with and without fillers.

    python -m eval.wer refs.txt results/run1/hyps.txt [--drop-fillers] [--show]

Both files: one utterance per line, "<id> <text>" (tab or space after the id).
"""
import argparse
import re

import jiwer

FILLED_PAUSES = {"um", "uh", "erm", "hmm", "ah", "eh", "emm"}


def normalize(text: str, keep_fillers: bool = True) -> str:
    t = re.sub(r"[^\w\s']", " ", text.lower().replace("-", " "))
    words = t.split()
    if not keep_fillers:
        words = [w for w in words if w not in FILLED_PAUSES]
    return " ".join(words)


def read(path: str) -> dict[str, str]:
    pairs = (line.split(maxsplit=1) for line in open(path, encoding="utf-8") if line.strip())
    return {p[0]: (p[1].strip() if len(p) > 1 else "") for p in pairs}


def score(refs: dict[str, str], hyps: dict[str, str], keep_fillers: bool = True):
    ids = sorted(refs)
    missing = [i for i in ids if i not in hyps]
    if missing:
        raise KeyError(f"No hypothesis for: {missing}")
    r = [normalize(refs[i], keep_fillers) for i in ids]
    h = [normalize(hyps[i], keep_fillers) for i in ids]
    out = jiwer.process_words(r, h)
    summary = {
        "files": len(ids),
        "ref_words": sum(len(x.split()) for x in r),
        "wer": round(out.wer, 4),
        "cer": round(jiwer.cer(r, h), 4),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
    }
    return summary, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs")
    ap.add_argument("hyps")
    ap.add_argument("--drop-fillers", action="store_true")
    ap.add_argument("--show", action="store_true", help="print word alignments for error analysis")
    a = ap.parse_args()
    summary, out = score(read(a.refs), read(a.hyps), keep_fillers=not a.drop_fillers)
    for k, v in summary.items():
        print(f"{k:>14}: {v}")
    if a.show:
        print(jiwer.visualize_alignment(out))
