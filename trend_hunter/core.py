from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Status = Literal["new", "growing", "one-off", "fading", "not real"]
HONG_KONG = ZoneInfo("Asia/Hong_Kong")


try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:  # pragma: no cover - exercised only in minimal environments
    _OPENCC = None


# A small fallback makes the script usable before requirements are installed.  The
# real conversion is performed by OpenCC and covers far more characters.
_T2S_FALLBACK = str.maketrans(
    {
        "鐵": "铁",
        "鍋": "锅",
        "賽": "赛",
        "強": "强",
        "體": "体",
        "驗": "验",
        "極": "极",
        "維": "维",
        "護": "护",
        "登": "登",
        "錄": "录",
        "伺": "伺",
        "務": "务",
        "器": "器",
        "機": "机",
        "率": "率",
        "禮": "礼",
        "課": "课",
        "壓": "压",
        "學": "学",
        "黨": "党",
        "實": "实",
        "裝": "装",
        "歐": "欧",
        "圖": "图",
        "號": "号",
        "線": "线",
        "團": "团",
        "戰": "战",
        "場": "场",
        "環": "环",
        "關": "关",
        "係": "系",
        "還": "还",
        "這": "这",
        "個": "个",
        "麼": "么",
        "會": "会",
        "沒": "没",
        "為": "为",
        "麼": "么",
        "覺": "觉",
        "後": "后",
        "開": "开",
        "掛": "挂",
        "裡": "里",
        "與": "与",
        "說": "说",
        "真": "真",
        "壞": "坏",
        "麗": "丽",
        "薦": "荐",
        "買": "买",
        "膚": "肤",
        "鬆": "松",
        "傷": "伤",
        "達": "达",
        "進": "进",
        "退": "退",
        "補": "补",
        "償": "偿",
        "網": "网",
        "優": "优",
        "化": "化",
    }
)

_URL_RE = re.compile(r"https?://\S+", re.I)
_IMAGE_RE = re.compile(r"<image:[^>]*>", re.I)
_CUSTOM_EMOJI_RE = re.compile(r"<:[^>]*>")
_EMOJI_RE = re.compile(r"<emoji:([^>]*)>", re.I)
_MENTION_RE = re.compile(r"@[\w\-\u3400-\u9fff]+")
_ASTRAL_RE = re.compile(r"[\U00010000-\U0010ffff]")
_SPACE_RE = re.compile(r"\s+")
_KEEP_RE = re.compile(r"[^\u3400-\u9fffA-Za-z0-9]+")
_NUMBER_RE = re.compile(r"\d+")

_BOILERPLATE = (
    "q群管家",
    "欢迎来到",
    "请先阅读群公告",
    "bug反馈请私聊管理",
    "本周活动预告详情见图",
    "每周攻略速递",
)

_BAD_EXACT_PHRASES = {
    "今天",
    "真的",
    "不是",
    "一下",
    "这个",
    "什么",
    "怎么",
    "有点",
    "还是",
    "感觉",
    "玩家",
    "游戏",
    "现在",
    "一个",
    "没有",
    "可以",
    "就是",
    "已经",
    "有人",
    "大家",
    "这么",
    "那边",
    "出来",
    "自己",
    "越来越",
    "好像",
    "时候",
    "的话",
    "官方",
    "问题",
    "新手",
    "萌新",
    "求问",
    "说真",
    "家人",
    "讲道",
    "有一",
    "老玩家",
    "绷不",
    "离谱",
    "笑死",
    "yyds",
    "哈哈",
    "hhhh",
    "2333",
    "orz",
    "qq",
    "emoji",
    "image",
    "com",
    "http",
    "https",
}

_MEANINGFUL_ASCII = (
    "pvp",
    "paytowin",
    "serverdown",
    "selena",
    "broken",
    "steam",
    "bug",
    "fps",
)


@dataclass(slots=True)
class Voice:
    id: str
    parent_id: str | None
    platform: str
    channel: str
    user_name: str
    title: str
    content: str
    created_at: datetime
    day: date
    normalized: str = ""
    display_text: str = ""


@dataclass(slots=True)
class Seed:
    term: str
    score: float
    recent_docs: set[int]
    c3: int
    c7: int
    p7: int
    p14: int
    old14: int


@dataclass(slots=True)
class Candidate:
    candidate_id: int
    seeds: list[str]
    score: float
    status_hint: Status
    first_seen: date
    last_seen: date
    peak_day: date
    peak_count: int
    c3: int
    c7: int
    p7: int
    p14: int
    prior: int
    active_days_14: int
    platform_counts: dict[str, int]
    user_count: int
    duplicate_share: float
    docs: set[int] = field(repr=False)
    representative_indices: list[int] = field(default_factory=list, repr=False)

    def prompt_record(self, voices: list[Voice]) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "seed_phrases": self.seeds[:8],
            "status_hint": self.status_hint,
            "metrics": {
                "first_seen": self.first_seen.isoformat(),
                "last_seen": self.last_seen.isoformat(),
                "messages_last_3d": self.c3,
                "messages_last_7d": self.c7,
                "messages_previous_7d": self.p7,
                "messages_previous_14d": self.p14,
                "active_days_last_14d": self.active_days_14,
                "platforms": self.platform_counts,
                "unique_users": self.user_count,
                "near_duplicate_share": round(self.duplicate_share, 3),
                "peak": {"date": self.peak_day.isoformat(), "messages": self.peak_count},
            },
            "representative_voices": [
                {
                    "id": voices[i].id,
                    "date": voices[i].day.isoformat(),
                    "platform": voices[i].platform,
                    "text": voices[i].display_text[:420],
                }
                for i in self.representative_indices
            ],
        }


def _to_simplified(text: str) -> str:
    if _OPENCC is not None:
        return _OPENCC.convert(text)
    return text.translate(_T2S_FALLBACK)


def normalize_text(text: str) -> str:
    """Normalize platform noise while retaining meaningful slang such as doge."""

    text = unicodedata.normalize("NFKC", text or "").lower()
    text = _EMOJI_RE.sub(lambda m: " doge " if "doge" in m.group(1).lower() else " ", text)
    text = _URL_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _CUSTOM_EMOJI_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _ASTRAL_RE.sub(" ", text)
    text = _to_simplified(text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def feature_text(text: str) -> str:
    return _KEEP_RE.sub("", text)


def load_voices(path: str | Path, as_of: date) -> list[Voice]:
    """Load only information available by the end of `as_of` in Hong Kong."""

    voices: list[Voice] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                created = datetime.fromisoformat(row["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=HONG_KONG)
                hk_day = created.astimezone(HONG_KONG).date()
                if hk_day > as_of:
                    continue
                title = row.get("title") or ""
                content = row.get("content") or ""
                display = " — ".join(part for part in (title.strip(), content.strip()) if part)
                voice = Voice(
                    id=str(row["id"]),
                    parent_id=row.get("parent_id"),
                    platform=str(row.get("platform") or "unknown"),
                    channel=str(row.get("channel") or ""),
                    user_name=str(row.get("user_name") or "unknown"),
                    title=title,
                    content=content,
                    created_at=created,
                    day=hk_day,
                    normalized=normalize_text(display),
                    display_text=display,
                )
                voices.append(voice)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSONL record on line {line_number}: {exc}") from exc

    voices.sort(key=lambda v: (v.created_at, v.id))
    _add_reply_context(voices)
    return voices


def _add_reply_context(voices: list[Voice]) -> None:
    """Attach parent meaning to otherwise context-free replies such as '确实'."""

    by_id = {voice.id: voice for voice in voices}
    acknowledgement = re.compile(
        r"^(确实|同意|是的|真的|哈哈+|笑死|\+1|有道理|好耶|草|艹|绷不住了)[!！?？。.~～\s]*$",
        re.I,
    )
    for voice in voices:
        compact = feature_text(voice.normalized)
        if not voice.parent_id or (len(compact) >= 10 and not acknowledgement.fullmatch(compact)):
            continue
        parent = by_id.get(voice.parent_id)
        # A one-word reaction to an official PV/guide is too ambiguous to count as
        # a separate trend voice.  Organic parent posts, however, are essential
        # for understanding replies such as "确实".
        if (
            parent is not None
            and parent.created_at <= voice.created_at
            and not _is_official_post(parent)
        ):
            voice.normalized = f"{parent.normalized} {voice.normalized}".strip()
            voice.display_text = f"{voice.display_text} ↳ reply to: {parent.display_text}".strip()


def _is_official_post(voice: Voice) -> bool:
    official_name = "官方" in voice.user_name or "官方" in voice.channel
    official_title = any(
        marker in voice.title.lower()
        for marker in (
            "pv",
            "公告",
            "攻略速递",
            "版本更新",
            "皮肤展示",
            "活动预告",
            "直播回放",
            "详情见图",
        )
    )
    return official_name and official_title


def _is_boilerplate(voice: Voice) -> bool:
    text = feature_text(voice.normalized)
    return (
        voice.user_name.lower() in {"q群管家", "bot", "system"}
        or _is_official_post(voice)
        or any(marker in text for marker in _BOILERPLATE)
    )


def _good_phrase(term: str) -> bool:
    if term in _BAD_EXACT_PHRASES or len(set(term)) <= 1:
        return False
    if any(
        marker in term
        for marker in (
            "emoji",
            "image",
            "http",
            "comtw",
            "editor",
            "emotion",
            "q群管家",
            "欢迎来到",
            "日常水",
            "第一次发帖",
            "回复上下文",
        )
    ):
        return False
    if re.fullmatch(r"[0-9a-f]{3,}", term):
        return False
    if re.fullmatch(r"[a-z0-9]+", term):
        return any(token in term for token in _MEANINGFUL_ASCII)
    if len(term) == 2 and term[0] in "的了是在和也都又就还我你他她它这那":
        return False
    return True


def _window_count(matrix, days_ago: np.ndarray, low: int, high: int) -> np.ndarray:
    mask = (days_ago >= low) & (days_ago <= high)
    if not np.any(mask):
        return np.zeros(matrix.shape[1], dtype=float)
    return np.asarray(matrix[mask].sum(axis=0)).ravel().astype(float)


def _seed_candidates(voices: list[Voice], as_of: date) -> list[Seed]:
    horizon_start = as_of - timedelta(days=41)
    local_to_global = [
        i
        for i, voice in enumerate(voices)
        if voice.day >= horizon_start
        and len(feature_text(voice.normalized)) >= 4
        and not _is_boilerplate(voice)
    ]
    if len(local_to_global) < 12:
        return []

    texts = [feature_text(voices[i].normalized) for i in local_to_global]
    min_df = 3 if len(texts) < 1_000 else 4
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(2, 9),
        binary=True,
        min_df=min_df,
        max_df=0.35,
        max_features=80_000,
        dtype=np.int8,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    terms = np.asarray(vectorizer.get_feature_names_out())
    days_ago = np.asarray([(as_of - voices[i].day).days for i in local_to_global])
    c3 = _window_count(matrix, days_ago, 0, 2)
    c7 = _window_count(matrix, days_ago, 0, 6)
    p7 = _window_count(matrix, days_ago, 7, 13)
    p14 = _window_count(matrix, days_ago, 14, 27)
    old14 = _window_count(matrix, days_ago, 28, 41)

    growth = np.log1p(c7) * np.maximum(0.0, np.log2((c7 + 2.0) / (p7 + 2.0)))
    fading = np.log1p(p7) * np.maximum(0.0, np.log2((p7 + 2.0) / (c7 + 2.0)))
    novelty = (c7 + p7) / (c7 + p7 + p14 + old14 + 2.0)
    recent_novelty = np.log1p(c7) * novelty
    early_acceleration = np.log1p(c3) * np.maximum(
        0.0,
        np.log2((c3 / 3.0 + 1.0) / (np.maximum(0.0, c7 - c3) / 4.0 + 1.0)),
    )
    score_components = [
        growth * 1.5,
        fading,
        recent_novelty * 0.9,
        early_acceleration,
    ]
    scores = np.maximum.reduce(score_components)
    support = np.maximum(c7, p7)

    # Round-robin across four kinds of change.  A huge coordinated burst can
    # otherwise occupy every top slot with slight variations of one slogan and
    # hide smaller organic stories.
    component_orders = [
        sorted(
            range(len(terms)),
            key=lambda j, component=component: (
                component[j],
                min(len(terms[j]), 10),
            ),
            reverse=True,
        )
        for component in score_components
    ]
    order: list[int] = []
    seen_features: set[int] = set()
    for rank in range(min(500, len(terms))):
        for component_order in component_orders:
            j = component_order[rank]
            if j not in seen_features:
                order.append(j)
                seen_features.add(j)
    seeds: list[Seed] = []
    for j in order:
        term = str(terms[j])
        if support[j] < 5 or scores[j] <= 0 or not _good_phrase(term):
            continue
        if len(term) == 2 and support[j] < 10:
            continue

        local_rows = matrix[:, j].nonzero()[0]
        docs = {local_to_global[int(k)] for k in local_rows}
        users = {voices[i].user_name for i in docs}
        if len(users) < 3:
            continue

        # If two substrings describe nearly the same messages, retain the more
        # informative one. Related but differently scoped phrases remain so the
        # next stage can merge them by co-occurrence.
        redundant = False
        similar_count = 0
        for previous in seeds:
            overlap = len(docs & previous.recent_docs) / max(
                1, min(len(docs), len(previous.recent_docs))
            )
            if overlap >= 0.72:
                similar_count += 1
            if term not in previous.term and previous.term not in term:
                continue
            if overlap >= 0.88:
                redundant = True
                break
        if redundant or similar_count >= 4:
            continue

        seeds.append(
            Seed(
                term=term,
                score=float(scores[j]) * (1.0 + 0.025 * min(len(term), 8)),
                recent_docs=docs,
                c3=int(c3[j]),
                c7=int(c7[j]),
                p7=int(p7[j]),
                p14=int(p14[j]),
                old14=int(old14[j]),
            )
        )
        if len(seeds) >= 240:
            break
    return seeds


def _cluster_seeds(seeds: list[Seed]) -> list[list[Seed]]:
    clusters: list[list[Seed]] = []
    cluster_docs: list[set[int]] = []
    for seed in seeds:
        best_index = -1
        best_overlap = 0.0
        for i, docs in enumerate(cluster_docs):
            intersection = len(seed.recent_docs & docs)
            overlap = intersection / max(1, min(len(seed.recent_docs), len(docs)))
            if overlap > best_overlap:
                best_index, best_overlap = i, overlap
        if best_index >= 0 and best_overlap >= 0.34:
            clusters[best_index].append(seed)
            cluster_docs[best_index] |= seed.recent_docs
        else:
            clusters.append([seed])
            cluster_docs.append(set(seed.recent_docs))
    return clusters


def _snippet_signature(voice: Voice, seeds: list[str]) -> str:
    parts = re.split(r"[\n。！？!?；;…]+|\.{2,}", voice.normalized)
    matching = [part for part in parts if any(seed in feature_text(part) for seed in seeds)]
    value = max(matching, key=len, default=voice.normalized)
    value = _NUMBER_RE.sub("#", feature_text(value))
    return value[:100]


def _topic_status(
    as_of: date,
    first_seen: date,
    c7: int,
    p7: int,
    p14: int,
    peak_count: int,
    docs_28: int,
    active_days_14: int,
    platform_counts: Counter,
    user_count: int,
    duplicate_share: float,
) -> Status:
    total = sum(platform_counts.values())
    dominant_platform = max(platform_counts.values(), default=0) / max(1, total)
    user_ratio = user_count / max(1, total)

    if (
        total >= 50
        and dominant_platform >= 0.82
        and user_ratio >= 0.62
        and duplicate_share >= 0.48
    ):
        return "not real"

    if (
        docs_28 >= 12
        and peak_count / max(1, docs_28) >= 0.35
        and (active_days_14 <= 4 or (p7 >= 8 and c7 <= p7 * 0.25))
    ):
        return "one-off"
    if first_seen >= as_of - timedelta(days=6) and c7 >= 6:
        return "new"
    if c7 >= 8 and c7 >= p7 * 1.22 and active_days_14 >= 3:
        return "growing"
    if p7 >= 7 and c7 <= p7 * 0.68:
        return "fading"
    if first_seen >= as_of - timedelta(days=24) and c7 >= 6:
        return "growing"
    if c7 >= 8 and c7 / 7.0 >= ((p7 + p14) / 21.0) * 1.12:
        return "growing"
    return "fading"


def _representatives(
    candidate: Candidate, voices: list[Voice], as_of: date, limit: int = 6
) -> list[int]:
    if candidate.status_hint in {"one-off", "fading", "not real"}:
        target = candidate.peak_day
    else:
        target = as_of

    ranked = sorted(
        candidate.docs,
        key=lambda i: (
            sum(seed in feature_text(voices[i].normalized) for seed in candidate.seeds[:8]),
            int(voices[i].parent_id is None or len(feature_text(voices[i].content)) >= 10),
            -abs((voices[i].day - target).days),
            len(voices[i].display_text),
        ),
        reverse=True,
    )
    selected: list[int] = []
    platforms: set[str] = set()
    users: set[str] = set()

    # First pass maximises platform and author diversity.
    for i in ranked:
        voice = voices[i]
        if voice.platform in platforms or voice.user_name in users:
            continue
        selected.append(i)
        platforms.add(voice.platform)
        users.add(voice.user_name)
        if len(selected) >= limit:
            return selected

    for i in ranked:
        if i not in selected:
            selected.append(i)
        if len(selected) >= limit:
            break
    return selected


def _episode_start(daily: Counter[date], peak_day: date, peak_count: int) -> date:
    """Estimate the start of the current story rather than its first stray mention."""

    threshold = max(2, math.ceil(peak_count * 0.15))
    significant = sorted(day for day, count in daily.items() if count >= threshold and day <= peak_day)
    if not significant:
        return min(daily)
    start = peak_day
    for day in reversed(significant):
        if day > start:
            continue
        if (start - day).days <= 4:
            start = day
        else:
            break
    return start


def _candidate_evidence(candidate: Candidate, voices: list[Voice]) -> str:
    recent_docs = sorted(
        candidate.docs,
        key=lambda i: (voices[i].day, voices[i].created_at),
        reverse=True,
    )[:80]
    return _to_simplified(
        " ".join(candidate.seeds + [voices[i].display_text for i in recent_docs])
    ).lower()


def _candidate_signature(candidate: Candidate, voices: list[Voice]) -> str:
    """Build a compact, game-agnostic description for semantic de-duplication."""

    snippets = Counter(
        _snippet_signature(voices[i], candidate.seeds)
        for i in candidate.docs
        if not _is_boilerplate(voices[i])
    )
    common = [snippet for snippet, _ in snippets.most_common(14) if snippet]
    return " ".join(candidate.seeds[:10] + common)


def discover_candidates(voices: list[Voice], as_of: date, limit: int = 18) -> list[Candidate]:
    """Discover temporally unusual conversation without using future messages."""

    seeds = _seed_candidates(voices, as_of)
    raw_clusters = _cluster_seeds(seeds)
    candidates: list[Candidate] = []

    for cluster in raw_clusters:
        cluster = sorted(cluster, key=lambda seed: seed.score, reverse=True)
        terms = [seed.term for seed in cluster[:10]]
        docs = {
            i
            for i, voice in enumerate(voices)
            if not _is_boilerplate(voice)
            and any(term in feature_text(voice.normalized) for term in terms)
        }
        if len(docs) < 5:
            continue

        daily = Counter(voices[i].day for i in docs)
        platforms = Counter(voices[i].platform for i in docs)
        users = {voices[i].user_name for i in docs}
        raw_first_seen, last_seen = min(daily), max(daily)
        peak_day, peak_count = max(daily.items(), key=lambda pair: (pair[1], pair[0]))
        first_seen = _episode_start(daily, peak_day, peak_count)

        def window(low: int, high: int) -> int:
            return sum(
                count
                for day, count in daily.items()
                if low <= (as_of - day).days <= high
            )

        c3, c7, p7, p14 = window(0, 2), window(0, 6), window(7, 13), window(14, 27)
        prior = sum(count for day, count in daily.items() if day < as_of - timedelta(days=27))
        docs_28 = c7 + p7 + p14
        active_days_14 = sum(
            1 for day in daily if as_of - timedelta(days=13) <= day <= as_of
        )
        signatures = Counter(_snippet_signature(voices[i], terms) for i in docs)
        duplicate_share = sum(count for _, count in signatures.most_common(5)) / max(1, len(docs))
        status = _topic_status(
            as_of,
            first_seen,
            c7,
            p7,
            p14,
            peak_count,
            docs_28,
            active_days_14,
            platforms,
            len(users),
            duplicate_share,
        )

        spread = math.log1p(len(platforms)) * math.log1p(len(users))
        dynamic = max(
            (c7 + 2) / (p7 + 2),
            (p7 + 2) / (c7 + 2),
            peak_count / max(1.0, docs_28 / 7.0),
        )
        freshness = 1.0 / (1.0 + max(0, (as_of - first_seen).days - 35) / 35.0)
        score = max(seed.score for seed in cluster) + math.log1p(docs_28) * dynamic * spread * freshness
        if status == "not real":
            score *= 1.35
        if docs_28 == 0:
            score *= 0.2

        candidates.append(
            Candidate(
                candidate_id=-1,
                seeds=terms,
                score=score,
                status_hint=status,
                first_seen=first_seen,
                last_seen=last_seen,
                peak_day=peak_day,
                peak_count=peak_count,
                c3=c3,
                c7=c7,
                p7=p7,
                p14=p14,
                prior=prior,
                active_days_14=active_days_14,
                platform_counts=dict(platforms),
                user_count=len(users),
                duplicate_share=duplicate_share,
                docs=docs,
            )
        )

    # Remove duplicate candidates created by several phrases from the same story.
    ordered_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    if ordered_candidates:
        signature_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=1,
            max_features=16_000,
            sublinear_tf=True,
        )
        signature_matrix = signature_vectorizer.fit_transform(
            [_candidate_signature(candidate, voices) for candidate in ordered_candidates]
        )
        similarities = cosine_similarity(signature_matrix)
    else:
        similarities = np.empty((0, 0))

    kept: list[Candidate] = []
    kept_indices: list[int] = []
    for ordered_index, candidate in enumerate(ordered_candidates):
        is_duplicate = False
        for previous_index, previous in zip(kept_indices, kept):
            overlap = len(candidate.docs & previous.docs) / max(
                1, min(len(candidate.docs), len(previous.docs))
            )
            if overlap >= 0.62 or similarities[ordered_index, previous_index] >= 0.34:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(candidate)
            kept_indices.append(ordered_index)
        if len(kept) >= limit:
            break

    for candidate_id, candidate in enumerate(kept):
        candidate.candidate_id = candidate_id
        candidate.representative_indices = _representatives(candidate, voices, as_of)
    return kept


def _local_labels(candidate: Candidate, voices: list[Voice]) -> tuple[str, str, str]:
    evidence = _candidate_evidence(candidate, voices)
    if candidate.status_hint == "not real":
        phrase = _readable_phrase(candidate)
        return (
            f"疑似协同刷屏：「{phrase}」",
            f"Suspiciously coordinated posting around ‘{phrase}’",
            "Community authenticity",
        )

    rules = [
        (
            "暗改" in evidence
            or (("爆率" in evidence or "机率" in evidence) and "公示" in evidence)
            or (
                ("抽卡记录" in evidence or "抽卡纪录" in evidence)
                and any(marker in evidence for marker in ("出金", "概率", "机率"))
            ),
            "玩家怀疑抽卡概率被暗改",
            "Players suspect a silent gacha-rate nerf",
            "Gacha economy & trust",
        ),
        (
            any(marker in evidence for marker in ("昵称", "外号", "新人别懵", "为什么大家都叫")),
            f"新玩家昵称正在形成：「{_readable_phrase(candidate)}」",
            f"A new player nickname is taking hold: ‘{_readable_phrase(candidate)}’",
            "Character culture & memes",
        ),
        (
            "pvp" in evidence and any(marker in evidence for marker in ("超模", "强度", "平衡", "排位")),
            f"PVP 强度争议围绕「{_readable_phrase(candidate)}」升温",
            f"A PvP balance backlash is forming around ‘{_readable_phrase(candidate)}’",
            "Combat balance",
        ),
        (
            any(marker in evidence for marker in ("维护延长", "炸服", "服务器崩", "登录不上", "掉线回档")),
            "更新维护后出现集中登录与服务器故障",
            "Login and server failures spike after maintenance",
            "Service reliability",
        ),
        (
            "匹配" in evidence and any(marker in evidence for marker in ("机制", "排位", "等五分钟", "匹配不到")),
            "玩家集中讨论匹配机制变化",
            "Players notice a shift in matchmaking",
            "Competitive experience",
        ),
        (
            any(marker in evidence for marker in ("逼氪", "pay to win", "paytowin", "活动奖励越来越抠")),
            "付费压力与逼氪感升高",
            "Monetisation pressure is becoming harder to ignore",
            "Monetisation & fairness",
        ),
        (
            "外挂" in evidence and any(marker in evidence for marker in ("锁头", "封号", "举报")),
            "外挂正在侵蚀排位体验",
            "Cheating is eroding ranked play",
            "Competitive integrity",
        ),
        (
            any(marker in evidence for marker in ("闪退", "掉帧", "发热严重", "自动保存就会卡死")),
            "性能与稳定性问题升温",
            "Performance and stability complaints are rising",
            "Technical quality",
        ),
        (
            "被削了" in evidence and "手感变差" in evidence,
            "玩家怀疑角色被削弱",
            "Players suspect an unannounced character nerf",
            "Combat balance",
        ),
    ]
    for matches, title, title_en, category in rules:
        if matches:
            return title, title_en, category

    phrase = _readable_phrase(candidate)
    return (
        f"玩家开始集中谈论「{phrase}」",
        f"Player conversation is shifting toward ‘{phrase}’",
        "Emerging player concern",
    )


def _readable_phrase(candidate: Candidate) -> str:
    candidates = [
        phrase.strip("的了啊呀吧呢是")
        for phrase in candidate.seeds
        if 2 <= len(phrase) <= 16
    ]
    candidates = [phrase for phrase in candidates if len(phrase) >= 2]
    if candidates:
        return max(candidates, key=lambda value: (len(value), -candidate.seeds.index(next(seed for seed in candidate.seeds if value in seed))))
    return candidate.seeds[0][:16]


def _local_why(candidate: Candidate) -> str:
    platforms = sorted(candidate.platform_counts, key=candidate.platform_counts.get, reverse=True)
    platform_text = "、".join(platforms[:4])
    if candidate.status_hint == "not real":
        return (
            f"讨论集中在{platform_text or '单一平台'}，前五种近重复表述占"
            f"{candidate.duplicate_share:.0%}，且账号分散但措辞高度同步，更像协同投放而非自然口碑。"
        )
    if candidate.status_hint == "one-off":
        return (
            f"声量在 {candidate.peak_day.isoformat()} 单日达到 {candidate.peak_count} 条，"
            f"随后迅速回落；形态更像一次事件冲击，而非持续形成的新关注。"
        )
    direction = "上升" if candidate.c7 >= candidate.p7 else "回落"
    return (
        f"{candidate.first_seen.isoformat()} 起可见；最近 7 天 {candidate.c7} 条，"
        f"此前 7 天 {candidate.p7} 条，声量正在{direction}，并已覆盖{platform_text or '多个渠道'}。"
    )


def choose_voice_ids(
    source_candidates: Iterable[Candidate], voices: list[Voice], as_of: date, limit: int = 6
) -> list[str]:
    merged_docs: set[int] = set()
    merged_seeds: list[str] = []
    candidates = list(source_candidates)
    for candidate in candidates:
        merged_docs |= candidate.docs
        merged_seeds.extend(candidate.seeds)
    if not merged_docs:
        return []

    synthetic = Candidate(
        candidate_id=-1,
        seeds=list(dict.fromkeys(merged_seeds)),
        score=max(candidate.score for candidate in candidates),
        status_hint=candidates[0].status_hint,
        first_seen=min(candidate.first_seen for candidate in candidates),
        last_seen=max(candidate.last_seen for candidate in candidates),
        peak_day=max(candidates, key=lambda item: item.peak_count).peak_day,
        peak_count=max(candidate.peak_count for candidate in candidates),
        c3=sum(candidate.c3 for candidate in candidates),
        c7=sum(candidate.c7 for candidate in candidates),
        p7=sum(candidate.p7 for candidate in candidates),
        p14=sum(candidate.p14 for candidate in candidates),
        prior=sum(candidate.prior for candidate in candidates),
        active_days_14=max(candidate.active_days_14 for candidate in candidates),
        platform_counts=dict(
            sum((Counter(candidate.platform_counts) for candidate in candidates), Counter())
        ),
        user_count=max(candidate.user_count for candidate in candidates),
        duplicate_share=max(candidate.duplicate_share for candidate in candidates),
        docs=merged_docs,
    )
    return [voices[i].id for i in _representatives(synthetic, voices, as_of, limit)]


def render_local_trends(
    candidates: list[Candidate], voices: list[Voice], as_of: date, limit: int = 8
) -> list[dict]:
    trends: list[dict] = []
    status_weight = {
        "not real": 5.0,
        "one-off": 1.6,
        "new": 1.15,
        "growing": 1.15,
        "fading": 1.0,
    }
    ordered = sorted(
        candidates,
        key=lambda item: item.score * status_weight[item.status_hint],
        reverse=True,
    )
    for candidate in ordered[:limit]:
        title, title_en, category = _local_labels(candidate, voices)
        trends.append(
            {
                "title": title,
                "title_en": title_en,
                "category": category,
                "status": candidate.status_hint,
                "first_seen": candidate.first_seen.isoformat(),
                "voice_ids": choose_voice_ids([candidate], voices, as_of),
                "why": _local_why(candidate),
            }
        )
    return trends
