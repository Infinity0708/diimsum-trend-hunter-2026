#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from trend_hunter.core import (
    discover_candidates,
    load_version_events,
    load_voices,
    render_local_trends,
)
from trend_hunter.llm import refine_with_openai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect newly forming stories in Chinese game-community messages."
    )
    parser.add_argument("--input", required=True, help="Path to voices.jsonl")
    parser.add_argument("--as-of", required=True, type=date.fromisoformat, help="YYYY-MM-DD (HKT)")
    parser.add_argument("--out", required=True, help="Path for the JSON result")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip optional OpenAI semantic refinement and use the local fallback.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print candidate metrics to stderr; never changes the output schema.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        voices = load_voices(args.input, args.as_of)
        version_events = load_version_events(args.input, args.as_of)
        candidates = discover_candidates(voices, args.as_of)
        if args.debug:
            for candidate in candidates:
                print(
                    json.dumps(candidate.prompt_record(voices), ensure_ascii=False),
                    file=sys.stderr,
                )

        trends = (
            None
            if args.no_llm
            else refine_with_openai(candidates, voices, args.as_of, version_events)
        )
        if trends is None:
            trends = render_local_trends(candidates, voices, args.as_of)

        result = {"as_of": args.as_of.isoformat(), "trends": trends[:8]}
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"trend-hunter: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
