#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from resume_validity.eval.evaluate_model import load_pairs, build_messages, generate_openrouter, extract_answer


def pick_examples(df, want: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pt in want:
        sub = df[df["pair_type"].astype(str).str.lower().eq(pt)]
        if not sub.empty:
            out.append(sub.iloc[0].to_dict())
    return out


def expected_label(pair_type: str) -> str:
    pt = str(pair_type or "").lower()
    if pt in {"equal", "reworded", "equivalent"}:
        return "abstain"
    if pt == "underqualified":
        return "first"
    if pt == "preferred":
        return "second"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser("Verify decisions on 3 samples using updated baseline prompts")
    ap.add_argument("--data", type=str, default="/home/zs7353/resume_validity/data/pairs_from_harvest/pairs_all_with_names_and_jd.jsonl")
    ap.add_argument("--model", type=str, default="openai/gpt-5")
    args = ap.parse_args()

    df = load_pairs(Path(args.data))
    examples = pick_examples(df, ["equal", "preferred", "underqualified"]) or df.head(3).to_dict("records")
    messages = [build_messages(rec) for rec in examples]
    raw = generate_openrouter(args.model, messages)

    print("Picked:", [e.get("pair_type") for e in examples])
    for rec, txt in zip(examples, raw):
        got = extract_answer(txt).strip().lower()
        if got not in {"first", "second", "abstain"}:
            got = "(invalid)"
        exp = expected_label(rec.get("pair_type"))
        ok = (got == exp)
        print("\n---")
        print("pair_type:", rec.get("pair_type"))
        print("decision:", got)
        print("expected:", exp)
        print("valid:", ok)
        print("raw_snippet:", str(txt)[:400].replace("\n", " "))


if __name__ == "__main__":
    main()


