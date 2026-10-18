# Trend Hunter — hearing a player story before it has a name

This repository is my solution to Diimsum's 2026 Trend Hunter challenge. It detects newly forming stories in noisy Chinese game-community messages while enforcing a strict historical `--as-of` boundary.

## Run

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --input voices.jsonl --as-of 2026-07-24 --out trends.json
```

If `OPENAI_API_KEY` is available, one structured model call refines the locally discovered candidates. Set `OPENAI_MODEL` to override the default model. With no key, no network, an unavailable model, or `--no-llm`, the deterministic fallback still writes a valid result.

```bash
python run.py --input voices.jsonl --as-of 2026-07-24 --out trends.json --no-llm
```

The output contains at most eight trends and uses only the required fields. Diagnostics go to stderr with `--debug`; they never change `trends.json`.

## The idea

The unit of meaning is a player voice, not a keyword. The pipeline:

1. Parses timestamps into Hong Kong time and drops every message after `--as-of` before doing any fitting or counting.
2. Normalises Unicode and Traditional/Simplified Chinese with OpenCC, removes platform markup, but preserves sentiment cues such as `doge`.
3. Reconstructs short organic replies such as “确实” from their parent. A short reply to an official PV or announcement is deliberately not promoted into independent evidence.
4. Mines character n-grams from the trailing 42 days. Character features work for unsegmented Chinese, typos and newly invented words without a predefined topic list.
5. Samples four kinds of temporal change separately: seven-day growth, seven-day decline, recent novelty, and three-day acceleration. Balanced sampling prevents one giant spam burst from occupying the whole candidate set.
6. Groups phrases by message co-occurrence, then measures onset, 3/7/14-day movement, active days, authors, channels, peak concentration and repeated templates.
7. Assigns a provisional status. A concentrated incident that collapses is `one-off`; a cross-platform build is `growing`; a single-platform wave of many one-use accounts repeating a handful of templates is `not real`.
8. Selects representative `voice_ids` with platform and author diversity. When an OpenAI key is present, a single Structured Outputs call merges synonymous candidates, interprets slang/sarcasm, and writes concise bilingual labels and a studio-actionable explanation. The model sees only candidate evidence already inside the historical boundary.

This is intentionally hybrid. Statistics decide *when the shape changed*; language understanding decides *what players mean*. The LLM is not asked to read or summarise all 48,000 messages, which keeps cost bounded and prevents volume from substituting for evidence.

## Status logic

- `new`: credible evidence began in the last week, but persistence is not yet established.
- `growing`: recent volume is increasing across several active days, preferably spreading across channels.
- `one-off`: a sharp peak dominates a short episode and then collapses.
- `fading`: a previously meaningful candidate is materially quieter than the preceding week.
- `not real`: posting is highly synchronised, template-heavy and dominated by one platform rather than organic conversation.

The `first_seen` value is the start of the current episode, not necessarily the first stray historical use of one generic word.

## What did not work

My first attempt fitted whole-corpus NMF topics. It produced clean descriptions of permanent background themes—pity questions, performance, art and matchmaking—but consistently buried the small phrase that had appeared only yesterday. Increasing the number of topics mainly split those large themes into more large themes.

A second attempt ranked every emerging n-gram in one list. It caught “铁锅” early, but a coordinated campaign later filled the candidate list with dozens of variations of the same slogan. The final design therefore balances four temporal samplers and groups overlapping evidence before ranking stories.

Raw frequency also misread forum threads: many replies inherit one title, while “+1” has no meaning outside its parent. Reply reconstruction and root-post preference fixed that failure mode.

## What is happening in 《星环守望》 by the end of August

The most important organic story is a loss of trust in the gacha rates. The first serious evidence is not the later angry word “暗改”, but players on Bahamut and Discord exporting their pull histories around 18 July and comparing them with the published probability. From 23 July the claim becomes cross-platform, acquires repeatable language (“暗改爆率”, “保留抽卡记录”, “集体投诉”), and survives well beyond a one-day run of bad luck. By late August the acceleration has eased, but the trust problem has not disappeared.

The 2.0 character Selena creates a different shape. Launch chatter begins on 21 July, then turns into a balance story as players report that she clears PvP teams, appears in most ranked matches and pushes older characters out. That is an actionable combat-balance issue, not merely banner excitement.

The 31 July 2.1 maintenance is the clean one-off: login failures, server crashes and rollback complaints rise together and then fall rapidly. It should be recorded as an incident, not reported weeks later as a growing community trend.

“Iron Pot” is the positive cultural signal. From 6 July, players independently explain that Aileen's round shield looks like a pot, and the nickname spreads across Simplified and Traditional Chinese communities. It is a small story, but an early detector should notice it because it becomes shared player vocabulary.

Finally, the Frostmoon skin praise is not organic. From 10–20 August, hundreds of numbered Weibo accounts post a tiny set of near-identical endorsements and the activity stops abruptly; almost no comparable spread appears elsewhere. The useful studio insight is not “players love the skin”, but that apparent reach is being manufactured and should be excluded from product decisions.

## Validation

```bash
pytest -q
python run.py --input voices.jsonl --as-of 2026-06-21 --out /tmp/june.json --no-llm
python run.py --input voices.jsonl --as-of 2026-07-24 --out /tmp/july.json --no-llm
python run.py --input voices.jsonl --as-of 2026-08-20 --out /tmp/august.json --no-llm
```

Tests cover Traditional/Simplified normalisation, reply context, the time boundary, detection of a newly invented nickname, the eight-trend cap and voice-ID validity.

## AI-use disclosure

OpenAI Codex was used as a coding partner for implementation, tests, backtesting and editing this explanation. The submitted runtime can optionally use the OpenAI Responses API for one structured semantic-refinement call; all temporal candidate generation and anti-manipulation measurements are local and auditable. No API key or invitation credential is stored in this repository.

The five language warm-ups are in [WARMUP.md](WARMUP.md).

For a from-scratch Chinese walkthrough of the code, design decisions, limitations, and interview
questions, see [docs/PROJECT_WALKTHROUGH_ZH.md](docs/PROJECT_WALKTHROUGH_ZH.md).
