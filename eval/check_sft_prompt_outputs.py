#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from resume_validity.eval.evaluate_model import load_pairs, generate_openrouter
from resume_validity.ft.sft_llama_pairs import (
    SYSTEM_PROMPT as SFT_SYS,
    build_user_message as sft_build_user_message,
    extract_answer as sft_extract,
)


def pick_examples(df, want: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pt in want:
        sub = df[df["pair_type"].astype(str).str.lower().eq(pt)]
        if not sub.empty:
            out.append(sub.iloc[0].to_dict())
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Check SFT prompt outputs on 3 samples")
    ap.add_argument("--data", type=str, default="/home/zs7353/resume_validity/data/pairs_from_harvest/pairs_all_with_names_and_jd.jsonl")
    ap.add_argument("--model", type=str, default="meta-llama/llama-3.1-8b-instruct")
    args = ap.parse_args()

    df = load_pairs(Path(args.data))
    examples = pick_examples(df, ["equal", "preferred", "underqualified"]) or df.head(3).to_dict("records")

    messages: List[List[Dict[str, str]]] = []
    for rec in examples:
        user_msg = sft_build_user_message(rec)
        messages.append([
            {"role": "system", "content": SFT_SYS},
            {"role": "user", "content": user_msg},
        ])

    raw = generate_openrouter(args.model, messages)
    print("Picked:", [e.get("pair_type") for e in examples])
    for rec, txt in zip(examples, raw):
        ans = sft_extract(txt)
        print("\n---")
        print("pair_type:", rec.get("pair_type"))
        print("extracted:", ans)
        print("raw_snippet:", str(txt)[:600].replace("\n", " "))


if __name__ == "__main__":
    main()


