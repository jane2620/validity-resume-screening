import json
from pathlib import Path
from resume_validity.build.assign_names import inject_name, inject_company_and_school, COMPANIES, SCHOOLS


def show_pairs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    # Choose up to 8 pairs for a single posting: 3 preferred, 3 underqualified, 2 equal
    preferred = [r for r in rows if r.get("pair_type") == "preferred"][:3]
    under = [r for r in rows if r.get("pair_type") == "underqualified"][:3]
    equal = [r for r in rows if r.get("pair_type") == "equal"][:2]
    sel = preferred + under + equal
    name_a = "Alex Taylor"
    name_b = "Jordan Lee"
    for i, r in enumerate(sel, 1):
        a = inject_company_and_school(inject_name(r["base_resume"], name_a))
        b = inject_company_and_school(inject_name(r["variant_resume"], name_b))
        print(f"\n==== Pair {i} ({r['pair_type']}) ====")
        print("-- Base --\n" + a)
        print("\n-- Variant --\n" + b)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True)
    args = ap.parse_args()
    show_pairs(args.path)


if __name__ == "__main__":
    main()


