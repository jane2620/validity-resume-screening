import json, sys, traceback
from pathlib import Path
from typing import Optional

from resume_validity.scrape.greenhouse_scraper import fetch_greenhouse_jobs
from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.qualification_extractor import extract_qualifications
from resume_validity.llm.resume_builder import build_preferred_resume


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="/home/zs7353/resume_validity/data/pairs_real.jsonl")
    ap.add_argument("--out_path", type=str, default="/home/zs7353/resume_validity/data/pair_preferred_example.json")
    ap.add_argument("--row_index", type=int, default=0, help="Zero-based index into JSONL to pick base resume")
    args = ap.parse_args()

    in_p = Path(args.in_path)
    out_p = Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Load selected row
    with open(in_p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if args.row_index < 0 or args.row_index >= len(lines):
        print("Row index out of range")
        sys.exit(2)
    row = json.loads(lines[args.row_index])

    base_md = row.get("base_resume", "")
    role = row.get("job_role", "")
    src = row.get("job_source", {})
    company = src.get("company", "")
    url = src.get("url", "")
    if not (base_md and role and company and url):
        print("Missing required fields in selected row")
        sys.exit(2)

    # Fetch postings for this company and find the matching one by absolute_url
    postings = fetch_greenhouse_jobs([company], max_jobs_per_company=50)
    match = None
    for p in postings:
        if p.url == url:
            match = p
            break
    if match is None:
        print("Could not re-fetch the matching posting by URL; aborting without fallback")
        sys.exit(3)

    # Extract qualifications and pick one bonus qualification
    gemini = GeminiClient()
    quals = extract_qualifications(gemini, match.content_text)
    bonus = [q.text for q in (quals.get("bonus") or []) if getattr(q, "text", "").strip()]
    if not bonus:
        print("No bonus qualifications extracted; aborting without fallback")
        sys.exit(4)
    added = bonus[0]

    # Build preferred variant using the exact base resume as context
    preferred_md = build_preferred_resume(gemini, role, base_md, added)

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "job_role": role,
            "company": company,
            "job_url": url,
            "added_bonus_qualification": added,
            "base_resume": base_md,
            "preferred_resume": preferred_md,
        }, f, ensure_ascii=False)
    print(f"Wrote preferred example → {out_p}")


if __name__ == "__main__":
    main()


