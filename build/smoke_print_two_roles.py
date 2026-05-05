import json
from pathlib import Path
from resume_validity.scrape.greenhouse_scraper import fetch_greenhouse_jobs, map_title_to_role
from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.qualification_extractor import extract_qualifications
from resume_validity.llm.resume_builder import (
    build_basic_resume,
    build_underqualified_resume_multi,
    build_preferred_resume_multi,
    build_reworded_equivalent_resume,
)
from resume_validity.build.assign_names import inject_name, inject_company_and_school, COMPANIES, SCHOOLS
import random


def generate_8_pairs_for_post(gemini, post):
    role = map_title_to_role(post.title)
    q = extract_qualifications(gemini, post.content_text)
    basic = [x.text for x in q.get("basic", []) if x.text]
    bonus = [x.text for x in q.get("bonus", []) if x.text]
    if not basic or not bonus:
        return []
    basic_md = build_basic_resume(gemini, role, basic)
    # Preferred 1,2,3
    pairs = []
    for k in [1,2,3]:
        if len(bonus) >= k:
            pref = build_preferred_resume_multi(gemini, role, basic_md, random.sample(bonus, k))
            pairs.append(("preferred", basic_md, pref))
    # Under 1,2,3
    for k in [1,2,3]:
        if len(basic) >= k:
            under = build_underqualified_resume_multi(gemini, role, basic_md, random.sample(basic, k))
            pairs.append(("underqualified", basic_md, under))
    # Equal 2
    for _ in range(2):
        eq = build_reworded_equivalent_resume(gemini, role, basic_md)
        pairs.append(("equal", basic_md, eq))
    return pairs[:8]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=str, default="datadog,stripe")
    args = ap.parse_args()
    companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    posts = fetch_greenhouse_jobs(companies, max_jobs_per_company=5)
    # pick one CS and one non-CS
    cs_roles = {"Software Engineer","ML Engineer","Data Scientist"}
    cs_post = next((p for p in posts if map_title_to_role(p.title) in cs_roles), None)
    non_post = next((p for p in posts if map_title_to_role(p.title) not in cs_roles), None)
    if not cs_post or not non_post:
        print("Could not find both CS and non-CS postings in small fetch.")
        return
    gemini = GeminiClient()
    for label, post in [("CS", cs_post), ("Non-CS", non_post)]:
        print(f"===== {label} Post: {post.title} ({post.url}) =====")
        pairs = generate_8_pairs_for_post(gemini, post)
        name_base = random.choice(["Alex Lee","Jordan Smith","Taylor Johnson","Casey Kim"])  # simple names for printing
        company = random.choice(COMPANIES)
        school = random.choice(SCHOOLS)
        for idx, (ptype, base, var) in enumerate(pairs, start=1):
            a = inject_company_and_school(inject_name(base, name_base), company, school)
            b = inject_company_and_school(inject_name(var, name_base), company, school)
            print(f"--- Pair {idx} [{ptype}] ---")
            print("[BASE]\n" + a + "\n")
            print("[VARIANT]\n" + b + "\n")


if __name__ == "__main__":
    main()


