import json, re, traceback, random
from pathlib import Path
from .demographics import load_names, sample_same_group_pair, sample_cross_group_pair


def inject_name(md: str, name: str) -> str:
    try:
        return re.sub(r"\{\{CANDIDATE_NAME\}\}", name, md, count=1)
    except Exception as e:
        print(f"Failed to inject name: {e}")
        traceback.print_exc()
        return md


COMPANIES = [
    "OpenAI", "Anthropic", "Google", "DeepMind", "Meta", "Microsoft", "Amazon",
    "NVIDIA", "Apple", "Bytedance", "TikTok", "Tesla", "Stripe", "Databricks",
    "Snowflake", "Netflix", "Uber", "Airbnb", "Coinbase", "Pinterest", "Shopify"
]

SCHOOLS = [
    "Massachusetts Institute of Technology", "Stanford University", "University of California, Berkeley",
    "Carnegie Mellon University", "Harvard University", "Princeton University", "California Institute of Technology",
    "University of Washington", "University of Illinois Urbana-Champaign", "Georgia Institute of Technology",
    "University of California, Los Angeles", "University of California, San Diego", "University of Toronto",
    "University of Oxford", "University of Cambridge"
]


def inject_company_and_school(md: str, company: str, school: str) -> str:
    try:
        md = re.sub(r"\{\{COMPANY_NAME\}\}", company, md)
        md = re.sub(r"\{\{SCHOOL_NAME\}\}", school, md)
        return md
    except Exception as e:
        print(f"Failed to inject company/school: {e}")
        traceback.print_exc()
        return md


def build_named_pairs(input_path: str, same_out: str, cross_out: str, target_same: int = 2000, target_cross: int = 2000):
    names = load_names()
    inp = Path(input_path)
    same_p = Path(same_out)
    cross_p = Path(cross_out)
    same_p.parent.mkdir(parents=True, exist_ok=True)
    cross_p.parent.mkdir(parents=True, exist_ok=True)

    same_written = 0
    cross_written = 0
    with open(inp, "r", encoding="utf-8") as f_in, \
         open(same_p, "w", encoding="utf-8") as f_same, \
         open(cross_p, "w", encoding="utf-8") as f_cross:
        for line in f_in:
            row = json.loads(line)
            if "pair_type" not in row:
                continue
            if same_written < target_same:
                n1, n2, grp = sample_same_group_pair(names)
                company = random.choice(COMPANIES)
                school = random.choice(SCHOOLS)
                a = inject_company_and_school(inject_name(row["base_resume"], n1), company, school)
                b = inject_company_and_school(inject_name(row["variant_resume"], n2), company, school)
                row_same = {**row, "name_a": n1, "name_b": n2, "group_a": grp, "group_b": grp, "base_resume": a, "variant_resume": b}
                f_same.write(json.dumps(row_same, ensure_ascii=False) + "\n")
                same_written += 1
            if cross_written < target_cross:
                n1, n2, g1, g2 = sample_cross_group_pair(names)
                company = random.choice(COMPANIES)
                school = random.choice(SCHOOLS)
                a = inject_company_and_school(inject_name(row["base_resume"], n1), company, school)
                b = inject_company_and_school(inject_name(row["variant_resume"], n2), company, school)
                row_cross = {**row, "name_a": n1, "name_b": n2, "group_a": g1, "group_b": g2, "base_resume": a, "variant_resume": b}
                f_cross.write(json.dumps(row_cross, ensure_ascii=False) + "\n")
                cross_written += 1
            if same_written >= target_same and cross_written >= target_cross:
                break
    print(f"Wrote {same_written} same-group pairs → {same_p}")
    print(f"Wrote {cross_written} cross-group pairs → {cross_p}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/home/zs7353/resume_validity/data/pairs.jsonl")
    ap.add_argument("--same_out", type=str, default="/home/zs7353/resume_validity/data/pairs_same.jsonl")
    ap.add_argument("--cross_out", type=str, default="/home/zs7353/resume_validity/data/pairs_cross.jsonl")
    ap.add_argument("--target_same", type=int, default=2000)
    ap.add_argument("--target_cross", type=int, default=2000)
    args = ap.parse_args()
    build_named_pairs(args.input, args.same_out, args.cross_out, args.target_same, args.target_cross)


if __name__ == "__main__":
    main()


