from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from trend_hunter.core import (
    Voice,
    discover_candidates,
    load_voices,
    normalize_text,
    render_local_trends,
)


def test_normalize_traditional_and_keep_doge_signal() -> None:
    value = normalize_text("角色外號=鐵鍋 <emoji:doge> https://example.com/x")
    assert "角色外号=铁锅" in value
    assert "doge" in value
    assert "http" not in value


def test_loader_blocks_future_and_adds_organic_reply_context(tmp_path) -> None:
    rows = [
        {
            "id": "parent",
            "parent_id": None,
            "platform": "TapTap",
            "channel": "评价",
            "user_name": "player-a",
            "title": None,
            "content": "新活动每天刷三十次，太肝了",
            "created_at": "2026-07-01T10:00:00+08:00",
        },
        {
            "id": "reply",
            "parent_id": "parent",
            "platform": "TapTap",
            "channel": "评价",
            "user_name": "player-b",
            "title": None,
            "content": "确实",
            "created_at": "2026-07-01T10:05:00+08:00",
        },
        {
            "id": "future",
            "parent_id": None,
            "platform": "QQ",
            "channel": "群",
            "user_name": "player-c",
            "title": None,
            "content": "未来才会发生的秘密话题",
            "created_at": "2026-07-02T00:00:01+08:00",
        },
    ]
    source = tmp_path / "voices.jsonl"
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    voices = load_voices(source, date(2026, 7, 1))
    assert [voice.id for voice in voices] == ["parent", "reply"]
    assert "太肝" in voices[1].normalized
    assert "reply to" in voices[1].display_text


def _voice(index: int, day: date, text: str, platform: str) -> Voice:
    created = datetime(day.year, day.month, day.day, 12, tzinfo=timezone(timedelta(hours=8)))
    return Voice(
        id=f"v-{index}",
        parent_id=None,
        platform=platform,
        channel="test",
        user_name=f"user-{index}",
        title="",
        content=text,
        created_at=created,
        day=day,
        normalized=normalize_text(text),
        display_text=text,
    )


def test_emerging_nickname_is_found_without_future_data() -> None:
    as_of = date(2026, 7, 12)
    voices: list[Voice] = []
    platforms = ["QQ", "Bilibili", "TapTap", "Discord"]
    for i in range(90):
        day = as_of - timedelta(days=20 - (i % 21))
        voices.append(_voice(i, day, f"普通聊天内容{i} 今天组队打副本", platforms[i % 4]))
    for j in range(14):
        day = as_of - timedelta(days=j % 6)
        voices.append(
            _voice(
                100 + j,
                day,
                f"洛洛这个法器像茶壶，大家已经开始叫她茶壶了{j}",
                platforms[j % 4],
            )
        )
    voices.sort(key=lambda voice: voice.created_at)

    candidates = discover_candidates(voices, as_of)
    trends = render_local_trends(candidates, voices, as_of)
    assert any(
        candidate.status_hint == "new"
        and any("茶壶" in voices[index].normalized for index in candidate.docs)
        for candidate in candidates
    )
    assert len(trends) <= 8
    allowed_ids = {voice.id for voice in voices if voice.day <= as_of}
    assert all(set(trend["voice_ids"]) <= allowed_ids for trend in trends)
