"""Pure-function tests for the Second Brain importer (no LLM, no files)."""
from datetime import date, timedelta

from seed_lens import classify_section, earliest_upcoming_deadline, map_priority, regex_clean


def test_classify_section_headers():
    assert classify_section("🔥 This Week") == "high"
    assert classify_section("🔥 Next Actions — U OF T ALIGNMENT") == "high"
    assert classify_section("📅 Current Priority") == "high"
    assert classify_section("🗓️ Next Week") == "high"
    assert classify_section("🔥 Current Status (Apr 27)") == "inherit"  # narrative, not a queue
    assert classify_section("⏸️ Waiting On") == "waiting"
    assert classify_section("🎯 On Deck (August)") == "low"
    assert classify_section("Paused / Deprioritized") == "low"
    assert classify_section("📅 Upcoming (Next 2-4 Weeks)") == "inherit"
    assert classify_section("Inbox (auto-extracted 2026-06-10)") == "inherit"
    assert classify_section("") == "inherit"


def test_earliest_upcoming_deadline_skips_past_dates():
    future_a = (date.today() + timedelta(days=30)).isoformat()
    future_b = (date.today() + timedelta(days=90)).isoformat()
    past = (date.today() - timedelta(days=10)).isoformat()
    deadlines = [
        f"{past} FIN Atlantic submission",
        f"{future_b} TAC Media Artists — companion application",
        f"{future_a} OAC Media Artists application — needs screenplay",
    ]
    assert earliest_upcoming_deadline(deadlines) == future_a
    assert earliest_upcoming_deadline([f"{past} only past"]) is None
    assert earliest_upcoming_deadline([]) is None
    assert earliest_upcoming_deadline(None) is None


def test_map_priority():
    assert map_priority(1) == "high"
    assert map_priority(2) == "normal"
    assert map_priority(3) == "low"
    assert map_priority(None) == "normal"
    assert map_priority("not a number") == "normal"


def test_regex_clean_strips_second_brain_notation():
    raw = "**Submit FIN Atlantic (Jun 12)** — per [[INDEX|Festival Strategy v3]] (`docs/strategy.md`)"
    cleaned = regex_clean(raw)
    assert "**" not in cleaned and "[[" not in cleaned and "`" not in cleaned
    assert "Submit FIN Atlantic" in cleaned
    assert "Festival Strategy v3" in cleaned
