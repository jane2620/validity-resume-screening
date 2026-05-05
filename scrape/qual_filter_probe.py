import json
from collections import defaultdict
from typing import Dict

from resume_validity.scrape.greenhouse_scraper import fetch_greenhouse_jobs, map_title_to_role, pick_top_roles
from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.qualification_extractor import extract_qualifications


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=str, default="datadog,stripe")
    ap.add_argument("--max_jobs_per_company", type=int, default=20)
    ap.add_argument("--require_basic", type=int, default=3)
    ap.add_argument("--require_bonus", type=int, default=3, help="alias: validity_count >= 3")
    ap.add_argument("--restrict_top_roles", action="store_true", help="Apply top-5 roles with <=2 CS-related roles")
    args = ap.parse_args()

    companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    posts = fetch_greenhouse_jobs(companies, max_jobs_per_company=args.max_jobs_per_company)
    if not posts:
        print("No posts fetched.")
        return

    if args.restrict_top_roles:
        roles = pick_top_roles(posts, top_k=5, max_cs_roles=2)
        posts = [p for p in posts if map_title_to_role(p.title) in roles]
        print(f"Restricting to top-5 roles (<=2 CS): {roles}")

    gemini = GeminiClient()
    total = 0
    pass_cnt = 0
    role_totals: Dict[str, int] = defaultdict(int)
    role_pass: Dict[str, int] = defaultdict(int)

    for p in posts:
        role = map_title_to_role(p.title)
        role_totals[role] += 1
        total += 1
        try:
            quals = extract_qualifications(gemini, p.content_text)
            basic = [q.text for q in (quals.get("basic") or []) if getattr(q, "text", "").strip()]
            bonus = [q.text for q in (quals.get("bonus") or []) if getattr(q, "text", "").strip()]
            bc, vc = len(basic), len(bonus)
            ok = (bc >= args.require_basic) and (vc >= args.require_bonus)
            if ok:
                pass_cnt += 1
                role_pass[role] += 1
            print(json.dumps({
                "role": role, "url": p.url, "basic_count": bc, "validity_count": vc, "pass": ok
            }))
        except Exception as e:
            print(json.dumps({"role": role, "url": p.url, "error": str(e)}))

    print("\nSummary:")
    print(f"Scanned posts: {total}")
    print(f"Pass threshold basic>={args.require_basic} & validity>={args.require_bonus}: {pass_cnt}")
    print("Per-role totals:")
    for r, c in sorted(role_totals.items(), key=lambda kv: kv[1], reverse=True):
        pr = role_pass.get(r, 0)
        print(f"  {r}: {pr}/{c} pass")


if __name__ == "__main__":
    main()


