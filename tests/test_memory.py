from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory


def test_evidence_insertion_and_window_size() -> None:
    memory = ShortTermMemory(short_term_window=2, lambda_decay=0.05)
    memory.add("u1", "i1", "d1", "hello world", 1.0)
    memory.add("u1", "i2", "d1", "hello again", 2.0)
    memory.add("u1", "i3", "d1", "third item", 3.0)
    evidence = memory.get_evidence("u1", "d1")
    assert len(evidence) == 2
    assert evidence[0].item_id == "i2"


def test_long_term_consolidation_promotes_after_support() -> None:
    short_term = ShortTermMemory()
    long_term = LongTermMemory(min_support=2, sim_threshold=0.1)
    ev1 = short_term.add("u1", "i1", "d1", "same pattern", 1.0)
    ev2 = short_term.add("u1", "i2", "d1", "same pattern", 2.0)
    long_term.consolidate("u1", "d1", [ev1])
    long_term.consolidate("u1", "d1", [ev2])
    assert len(long_term.get_evidence("u1", "d1")) == 1


def test_recency_decay_is_positive() -> None:
    memory = ShortTermMemory()
    evidence = memory.add("u1", "i1", "d1", "text", 1.0)
    assert evidence.recency_weight > 0.0
