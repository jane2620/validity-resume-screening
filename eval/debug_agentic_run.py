#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reuse the exact agentic builders and OpenRouter caller
from resume_validity.eval.evaluate_agentic import (
    load_pairs,
    build_summariser_messages,
    build_decider_messages,
    generate_openrouter,
    extract_answer as agentic_extract_answer,
)


def pick_examples(df_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Pick one example for each of the requested pair types.

    Targets: preferred, underqualified, reworded (or equal/equivalent).
    """
    targets: List[Tuple[str, List[str]]] = [
        ("preferred", ["preferred"]),
        ("underqualified", ["underqualified"]),
        ("reworded", ["reworded", "equal", "equivalent"]),
    ]
    chosen: Dict[str, Dict[str, Any]] = {}
    for rec in df_records:
        pt = str(rec.get("pair_type") or "").strip().lower()
        for label, aliases in targets:
            if label in chosen:
                continue
            if pt in aliases:
                chosen[label] = rec
        if len(chosen) == len(targets):
            break
    return chosen


def print_step(title: str, content: str) -> None:
    print(f"\n=== {title} ===")
    print(content)
    sys.stdout.flush()


def run_one(model_name: str, label: str, example: Dict[str, Any]) -> None:
    print(f"\n#############################")
    print(f"# Pair type: {label} (original={example.get('pair_type')})")
    print(f"#############################\n")

    # Summariser
    sum_msgs = build_summariser_messages(example)
    sum_sys = sum_msgs[0]["content"]
    sum_usr = sum_msgs[1]["content"]
    print_step("Summariser SYSTEM", sum_sys)
    print_step("Summariser USER", sum_usr)
    sum_out = generate_openrouter(model_name, [sum_msgs])[0]
    print_step("Summariser OUTPUT", sum_out)

    # Decider (no job description passed; summaries only)
    dec_msgs = build_decider_messages(sum_out)
    dec_sys = dec_msgs[0]["content"]
    dec_usr = dec_msgs[1]["content"]
    print_step("Decider SYSTEM", dec_sys)
    print_step("Decider USER", dec_usr)
    dec_out = generate_openrouter(model_name, [dec_msgs])[0]
    print_step("Decider OUTPUT", dec_out)

    # Decisions are final in the updated agentic pipeline; no naming step


def main() -> None:
    parser = argparse.ArgumentParser("Run agentic pipeline on 3 examples and print I/O")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/zs7353/resume_validity/data/pairs_from_harvest/pairs_all_with_names_and_jd.jsonl",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/llama-3.1-8b-instruct",
        help="OpenRouter model id",
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(2)

    input_fp = Path(args.input)
    if not input_fp.exists():
        print(f"ERROR: Input file not found: {input_fp}", file=sys.stderr)
        sys.exit(2)

    try:
        df = load_pairs(input_fp)
        records = df.to_dict("records")
        chosen = pick_examples(records)
        order = ["preferred", "underqualified", "reworded"]
        missing = [lbl for lbl in order if lbl not in chosen]
        if missing:
            print(f"WARNING: Missing examples for: {', '.join(missing)}", file=sys.stderr)
        for lbl in order:
            if lbl in chosen:
                run_one(args.model_name, lbl, chosen[lbl])
    except Exception as e:
        print("EXCEPTION while running debug pipeline:", file=sys.stderr)
        print(repr(e), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


