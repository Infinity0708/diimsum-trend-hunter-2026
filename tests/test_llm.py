from __future__ import annotations

from datetime import date, datetime, timezone

from trend_hunter.core import Candidate, Voice
from trend_hunter.llm import _ensure_manipulation_is_reported


def _fake_voice() -> Voice:
    return Voice(
        id="fake-voice",
        parent_id=None,
        platform="Weibo",
        channel="topic",
        user_name="numbered-account-001",
        title="",
        content="新皮肤太美了，必须拿下",
        created_at=datetime(2026, 8, 20, 12, tzinfo=timezone.utc),
        day=date(2026, 8, 20),
        normalized="新皮肤太美了必须拿下",
        display_text="新皮肤太美了，必须拿下",
    )


def _fake_candidate() -> Candidate:
    return Candidate(
        candidate_id=3,
        seeds=["新皮肤太美了", "必须拿下"],
        score=900.0,
        status_hint="not real",
        first_seen=date(2026, 8, 10),
        last_seen=date(2026, 8, 20),
        peak_day=date(2026, 8, 15),
        peak_count=80,
        c3=0,
        c7=0,
        p7=200,
        p14=360,
        prior=0,
        active_days_14=7,
        platform_counts={"Weibo": 560},
        user_count=560,
        duplicate_share=0.55,
        docs={0},
        representative_indices=[0],
    )


def test_high_confidence_manipulation_survives_semantic_omission() -> None:
    as_of = date(2026, 8, 30)
    voice = _fake_voice()
    candidate = _fake_candidate()

    results = _ensure_manipulation_is_reported([], [candidate], [voice], as_of, set())

    assert len(results) == 1
    assert results[0]["status"] == "not real"
    assert "新皮肤太美了" in results[0]["title"]
    assert results[0]["voice_ids"] == ["fake-voice"]


def test_reported_manipulation_is_not_duplicated() -> None:
    results = [{"status": "not real", "title": "already present"}]
    output = _ensure_manipulation_is_reported(
        results,
        [_fake_candidate()],
        [_fake_voice()],
        date(2026, 8, 30),
        {3},
    )
    assert output == results


def test_missing_manipulation_keeps_the_eight_trend_cap() -> None:
    results = [{"status": "growing", "title": f"trend-{i}"} for i in range(8)]
    output = _ensure_manipulation_is_reported(
        results,
        [_fake_candidate()],
        [_fake_voice()],
        date(2026, 8, 30),
        set(),
    )
    assert len(output) == 8
    assert output[-1]["status"] == "not real"
