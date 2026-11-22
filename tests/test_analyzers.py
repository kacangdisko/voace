from types import SimpleNamespace

from coach.analyzers import fillers, pace, pauses
from coach.engines.nemotron import to_words
from coach.events import Word, norm


def words(*items):
    return [Word(text, start, start + 0.3) for text, start in items]


def test_norm():
    assert norm("Hello,") == "hello"
    assert norm("don't") == "don't"
    assert norm("data-data.") == "data-data"


def test_syllables():
    assert [pace.syllables(w) for w in ["the", "make", "little", "presentation", "um"]] == [1, 1, 2, 4, 1]
    assert pace.syllables("mempresentasikan", "id") == 6
    assert pace.syllables("pantai", "id") == 2


def test_fillers_multiword_and_punctuation():
    ws = words(("So,", 0), ("you", 1), ("know", 2), ("um", 3), ("I", 4), ("like", 5), ("it.", 6))
    hits = fillers.find(ws, fillers.EN)
    assert [h[2] for h in hits] == ["so", "you know", "um", "like"]
    assert fillers.summary(ws, fillers.EN, minutes=0.5) == {
        "total": 4, "per_min": 8.0, "by_word": {"so": 1, "you know": 1, "um": 1, "like": 1}}


def test_pace_and_nudge():
    ws = words(*[(f"w{i}", i * 0.3) for i in range(100)])  # 200 wpm
    p = pace.Pace(window=20)
    m = p.measure(ws, now=30.0)
    assert 190 <= m["wpm"] <= 210
    assert p.nudge(m["wpm"], 10.0) is None
    assert p.nudge(m["wpm"], 20.5) == "Slow down a little"
    assert p.nudge(m["wpm"], 25.0) is None  # only once per fast stretch


def test_pauses():
    pa = pauses.Pauses()
    pa.add([(0.0, 2.0), (2.3, 4.0), (5.0, 6.0), (8.5, 9.0), (13.0, 14.0)])
    assert pa.counts() == {"good": 1, "long": 1, "freeze": 1}
    assert not pa.freeze_now(16.0, speaking=True)
    assert pa.freeze_now(17.5, speaking=False)
    assert not pa.freeze_now(18.0, speaking=False)  # flagged once


def test_pause_counted_when_speech_resumes():
    pa = pauses.Pauses()
    pa.add([(0.0, 2.0)])
    pa.add([], ongoing_start=6.0)          # speech resumed, segment not finished yet
    assert pa.counts()["freeze"] == 1
    pa.add([(6.0, 9.0)])                   # same segment finishes later: no double count
    assert pa.counts() == {"good": 0, "long": 0, "freeze": 1}


def test_token_merge():
    res = SimpleNamespace(tokens=[" A", "fter", " ear", "ly", ","], timestamps=[0.56, 0.56, 0.72, 0.72, 0.8],
                          ys_probs=[-0.1, 0.0, -0.5, -0.5, 0.0], start_time=7.0)
    ws = to_words(res)
    assert [w.text for w in ws] == ["After", "early,"]
    assert ws[1].start == 7.72 and ws[1].end == 7.88
    assert 0 < ws[1].conf < 1
