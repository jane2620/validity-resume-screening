#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from resume_validity.eval.evaluate_model import (
    load_pairs,
    build_messages,
    generate_openrouter,
    extract_answer,
)


def pick_examples(df: pd.DataFrame, k: int = 2) -> List[Dict[str, Any]]:
    pts = ["equal", "preferred", "underqualified"]
    chosen: List[Dict[str, Any]] = []
    for pt in pts:
        sub = df[df["pair_type"].astype(str).str.lower().eq(pt)]
        if not sub.empty:
            chosen.append(sub.iloc[0].to_dict())
        if len(chosen) >= k:
            break
    while len(chosen) < min(k, len(df)):
        chosen.append(df.iloc[len(chosen)].to_dict())
    return chosen[:k]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser("Print sample raw outputs for a given OpenRouter model")
    ap.add_argument("--model", type=str, default="google/gemini-2.5-pro")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    data_path = Path("/home/zs7353/resume_validity/data/pairs_from_harvest/pairs_all_with_names_and_jd.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    df = load_pairs(data_path)
    if df.empty:
        raise RuntimeError("pairs dataset empty")
    examples = pick_examples(df, k=args.k)

    messages = [build_messages(rec) for rec in examples]
    print("Selected pair_types:", [rec.get("pair_type") for rec in examples])

    print("Calling:", args.model)
    raw = generate_openrouter(args.model, messages)

    for i, (rec, msgs, txt) in enumerate(zip(examples, messages, raw), start=1):
        print(f"\n===== SAMPLE {i} =====")
        print("pair_type:", rec.get("pair_type"))
        # show last few lines of user prompt
        umsg = [m for m in msgs if m["role"] == "user"][0]["content"]
        tail = "\n".join(umsg.strip().splitlines()[-6:])
        print("\n--- Prompt tail ---\n" + tail)
        print("\n--- Raw output ---\n" + str(txt))
        print("\n--- Parsed extract_answer ---")
        print(extract_answer(txt))


if __name__ == "__main__":
    main()


