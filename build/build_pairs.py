import os, json, math, random, traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

from resume_validity.scrape.greenhouse_scraper import fetch_greenhouse_jobs, pick_top_roles, bucket_by_role, map_title_to_role
from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.qualification_extractor import extract_qualifications, Qualification
from resume_validity.llm.resume_builder import (
    build_basic_resume,
    build_underqualified_resume,
    build_preferred_resume,
    build_reworded_equivalent_resume,
    build_underqualified_resume_multi,
    build_preferred_resume_multi,
)


@dataclass
class Pair:
    job_role: str
    job_source: Dict[str, str]
    pair_type: str  # underqualified | preferred | equal
    base_resume: str
    variant_resume: str
    perturbation: Dict[str, str]  # { type: removed|added|reworded, qualification: <text> }


def ensure_name_placeholder(md: str) -> str:
    if "{{CANDIDATE_NAME}}" not in md:
        return "Name: {{CANDIDATE_NAME}}\n\n" + md
    return md


def construct_pairs_for_post(gemini: GeminiClient, post, max_under: int = 3, max_pref: int = 3) -> Tuple[List[Pair], Dict[str, object]]:
    pairs: List[Pair] = []
    role = map_title_to_role(post.title)
    debug: Dict[str, object] = {"role": role, "url": post.url}
    try:
        q = extract_qualifications(gemini, post.content_text)
        basic_q = [qq.text for qq in q.get("basic", []) if qq.text]
        bonus_q = [qq.text for qq in q.get("bonus", []) if qq.text]
    except Exception as e:
        return pairs, {**debug, "error": f"qualification_extraction_failed: {e}"}
    debug.update({
        "basic_count": len(basic_q) if 'basic_q' in locals() else 0,
        "bonus_count": len(bonus_q) if 'bonus_q' in locals() else 0,
        "basic_sample": (basic_q[:5] if 'basic_q' in locals() else []),
        "bonus_sample": (bonus_q[:5] if 'bonus_q' in locals() else []),
    })
    # No filtering: proceed even if one of the lists is empty (will just yield fewer variants)

    # Build the basic resume
    try:
        basic_md = build_basic_resume(gemini, role, basic_q)
    except Exception as e:
        return pairs, {**debug, "error": f"basic_resume_failed: {e}"}
    basic_md = ensure_name_placeholder(basic_md)

    # Under-qualified resumes: remove exactly one basic qualification per variant, capped at 3
    for removed in basic_q[: min(max_under, len(basic_q))]:
        try:
            under_md = build_underqualified_resume(gemini, role, basic_md, removed)
        except Exception as e:
            return pairs, {**debug, "error": f"underqualified_failed: {e}", "removed": removed}
        under_md = ensure_name_placeholder(under_md)
        pairs.append(Pair(
            job_role=role,
            job_source={"source": post.source, "company": post.company, "title": post.title, "url": post.url},
            pair_type="underqualified",
            base_resume=basic_md,
            variant_resume=under_md,
            perturbation={"type": "removed", "qualification": removed, "delta_count": 1},
        ))

    # Preferred resumes: add exactly one bonus qualification per variant, capped at 3
    for added in bonus_q[: min(max_pref, len(bonus_q))]:
        try:
            pref_md = build_preferred_resume(gemini, role, basic_md, added)
        except Exception as e:
            return pairs, {**debug, "error": f"preferred_failed: {e}", "added": added}
        pref_md = ensure_name_placeholder(pref_md)
        pairs.append(Pair(
            job_role=role,
            job_source={"source": post.source, "company": post.company, "title": post.title, "url": post.url},
            pair_type="preferred",
            base_resume=basic_md,
            variant_resume=pref_md,
            perturbation={"type": "added", "qualification": added, "delta_count": 1},
        ))

    # Two equal resumes via rewording
    for _ in range(2):
        try:
            eq_md = build_reworded_equivalent_resume(gemini, role, basic_md)
        except Exception as e:
            return pairs, {**debug, "error": f"equal_failed: {e}"}
        eq_md = ensure_name_placeholder(eq_md)
        pairs.append(Pair(
            job_role=role,
            job_source={"source": post.source, "company": post.company, "title": post.title, "url": post.url},
            pair_type="equal",
            base_resume=basic_md,
            variant_resume=eq_md,
            perturbation={"type": "reworded", "qualifications": [], "delta_count": 0},
        ))

    return pairs, debug


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=str, default="stripe,doordash,datadog,notion,asana,rippling,roblox,instacart,opendoor,brex")
    ap.add_argument("--max_jobs_per_company", type=int, default=30)
    ap.add_argument("--top_roles", type=int, default=5)
    ap.add_argument("--max_cs_roles", type=int, default=2)
    ap.add_argument("--per_role_postings", type=int, default=100, help="Exactly N postings per selected role")
    ap.add_argument("--target_pairs", type=int, default=0, help="If 0, computed as 8 * top_roles * per_role_postings")
    ap.add_argument("--per_role_cap", type=int, default=120, help="Max unique postings to consume per top role")
    ap.add_argument("--out", type=str, default="/home/zs7353/resume_validity/data/pairs.jsonl")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_progress", action="store_true", help="Disable progress bars for clean logs")
    ap.add_argument("--max_attempts_per_role", type=int, default=50, help="Stop after N attempts without finding an 8-pair posting")
    ap.add_argument("--debug_log", type=str, default="/home/zs7353/resume_validity/data/pairs_debug.log")
    args = ap.parse_args()

    companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    posts = fetch_greenhouse_jobs(companies, max_jobs_per_company=args.max_jobs_per_company)
    if not posts:
        print("No posts fetched; exiting.")
        return

    top_roles = pick_top_roles(posts, top_k=args.top_roles, max_cs_roles=args.max_cs_roles)
    buckets = bucket_by_role(posts)

    # Decide how many to scrape per role to reach target_pairs approximately.
    # Each posting yields up to (min(3, len(basic))) under + (min(3, len(bonus))) pref + 1 equal pairs.
    # We iterate role buckets round-robin until reaching target.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gemini = None
    if not args.dry_run:
        gemini = GeminiClient()

    total_written = 0
    pairs_per_post = 8
    planned_total = args.target_pairs if args.target_pairs > 0 else (pairs_per_post * len(top_roles) * args.per_role_postings)
    print(f"Planning to write {planned_total} pairs: {len(top_roles)} roles x {args.per_role_postings} postings x {pairs_per_post} pairs/post")
    with open(out_path, "w", encoding="utf-8") as f, open(args.debug_log, "w", encoding="utf-8") as flog:
        pbar = tqdm(total=planned_total, desc="Building pairs") if not args.no_progress else None
        for role in top_roles:
            bucket = buckets.get(role, [])
            role_idx = 0
            accepted_posts = 0
            # Keep selecting postings (cycling if needed) until we have exactly per_role_postings accepted posts
            seen_failures = 0
            attempts = 0
            while accepted_posts < args.per_role_postings:
                if not bucket:
                    print(f"No postings for role {role}")
                    break
                post = bucket[role_idx % len(bucket)]
                role_idx += 1
                attempts += 1
                if attempts > args.max_attempts_per_role:
                    print(f"Role {role}: reached {args.max_attempts_per_role} attempts without a full 8-pair posting; moving on.")
                    break

            try:
                if args.dry_run:
                    # Write a stub with only source metadata to show planned count
                    stub = {
                        "job_role": role,
                        "job_source": {"source": post.source, "company": post.company, "title": post.title, "url": post.url},
                        "note": "dry_run_no_llm"
                    }
                    f.write(json.dumps(stub) + "\n")
                    total_written += 1
                    if pbar:
                        pbar.update(1)
                    accepted_posts += 1
                    continue

                pairs, dbg = construct_pairs_for_post(gemini, post)
                # Accept only postings that yield exactly 8 pairs
                if len(pairs) == pairs_per_post:
                    print(f"ACCEPT role={role} url={post.url} basic_count={dbg.get('basic_count')} bonus_count={dbg.get('bonus_count')}")
                    flog.write(json.dumps({"accept": True, **dbg}) + "\n")
                    for pr in pairs:
                        f.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
                        total_written += 1
                        if pbar:
                            pbar.update(1)
                    accepted_posts += 1
                else:
                    print(f"REJECT role={role} url={post.url} basic_count={dbg.get('basic_count')} bonus_count={dbg.get('bonus_count')} reason={dbg.get('reason','not_enough_pairs')} error={dbg.get('error','')}")
                    flog.write(json.dumps({"accept": False, **dbg}) + "\n")
                    seen_failures += 1
                    if seen_failures % 10 == 0:
                        print(f"Role {role}: {seen_failures} postings yielded != {pairs_per_post} pairs; continuing")
            except Exception as e:
                print(f"Error while constructing pairs: {e}")
                traceback.print_exc()
                continue
    print(f"Wrote {total_written} pairs to {out_path}")


if __name__ == "__main__":
    main()


