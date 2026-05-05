import json, random, re, traceback
from pathlib import Path
from typing import Tuple, List, Dict

from .demographics import load_names, wb_only_groups, enumerate_wb_pairs
from .assign_names import inject_name, inject_company_and_school, COMPANIES, SCHOOLS


def sample_from_group(names_map: Dict[str, List[str]], group: str) -> str:
    return random.choice(names_map[group])


def build_balanced_pairs(input_path: str, out_path: str, pairs_per_demo_pair: int = 400):
    names_full = load_names()
    names_map = wb_only_groups(names_full)
    demo_pairs = enumerate_wb_pairs()  # 10 pairs including same-group

    inp = Path(input_path)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # We will round-robin over demo pairs and over source pairs to meet exact totals.
    # Each line of input is one pair; we replicate with different names.
    with open(inp, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
        rows = [json.loads(l) for l in f_in if 'pair_type' in l]
        # Ensure we have enough source pairs to cover the required output
        required_total = pairs_per_demo_pair * len(demo_pairs)
        if len(rows) == 0:
            print("No source pairs to assign names to.")
            return
        idx = 0
        for (g1, g2) in demo_pairs:
            for _ in range(pairs_per_demo_pair):
                row = rows[idx % len(rows)]
                idx += 1
                name_a = sample_from_group(names_map, g1)
                name_b = sample_from_group(names_map, g2)
                company = random.choice(COMPANIES)
                school = random.choice(SCHOOLS)
                a = inject_company_and_school(inject_name(row["base_resume"], name_a), company, school)
                b = inject_company_and_school(inject_name(row["variant_resume"], name_b), company, school)
                out = {**row, "name_a": name_a, "name_b": name_b, "group_a": g1, "group_b": g2, "base_resume": a, "variant_resume": b}
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {pairs_per_demo_pair*len(demo_pairs)} balanced W/B pairs → {outp}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/home/zs7353/resume_validity/data/pairs_real_v2.jsonl")
    ap.add_argument("--out", type=str, default="/home/zs7353/resume_validity/data/pairs_balanced_wb.jsonl")
    ap.add_argument("--pairs_per_demo_pair", type=int, default=400)
    args = ap.parse_args()
    build_balanced_pairs(args.input, args.out, args.pairs_per_demo_pair)


if __name__ == "__main__":
    main()


