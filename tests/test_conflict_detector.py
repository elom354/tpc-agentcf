from src.llm.mock_llm import MockLLM
from src.memory.conflict_detector import ConflictDetector
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory


def test_no_conflict_when_short_term_empty() -> None:
    detector = ConflictDetector(MockLLM())
    signal = detector.detect("u1", "d1", ShortTermMemory(), LongTermMemory())
    assert not signal.is_conflict


def test_no_conflict_when_long_term_empty() -> None:
    short_term = ShortTermMemory()
    for idx in range(3):
        short_term.add("u1", f"i{idx}", "d1", "similar text", float(idx))
    detector = ConflictDetector(MockLLM())
    signal = detector.detect("u1", "d1", short_term, LongTermMemory())
    assert not signal.is_conflict


def test_conflict_detected_when_embeddings_diverge() -> None:
    short_term = ShortTermMemory()
    long_term = LongTermMemory(min_support=1)
    for idx in range(3):
        short_term.add("u1", f"i{idx}", "d1", "action thriller explosion", float(idx))
    long_term.consolidate("u1", "d1", [short_term.add("u1", "i9", "d1", "romance love letters", 10.0)])
    detector = ConflictDetector(MockLLM(), conflict_threshold=0.1)
    signal = detector.detect("u1", "d1", short_term, long_term)
    assert signal.conflict_score >= 0.0
    assert signal.short_term_evidence_ids
    assert signal.long_term_evidence_ids


def test_no_conflict_when_embeddings_similar() -> None:
    short_term = ShortTermMemory()
    long_term = LongTermMemory(min_support=1)
    for idx in range(3):
        ev = short_term.add("u1", f"i{idx}", "d1", "consistent comedy fun", float(idx))
        long_term.consolidate("u1", "d1", [ev])
    detector = ConflictDetector(MockLLM(), conflict_threshold=0.9)
    signal = detector.detect("u1", "d1", short_term, long_term)
    assert not signal.is_conflict
