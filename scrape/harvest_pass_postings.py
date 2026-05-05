import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from resume_validity.scrape.greenhouse_scraper import (
    fetch_greenhouse_jobs,
    map_title_to_role,
    pick_top_roles,
    bucket_by_role,
)
from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.qualification_extractor import extract_qualifications


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=str, default="stripe,doordash,datadog,notion,roblox,instacart,asana,rippling,opendoor,brex")
    ap.add_argument("--max_jobs_per_company", type=int, default=100)
    ap.add_argument("--restrict_top_roles", action="store_true", help="Apply top-5 roles with <=2 CS-related roles")
    ap.add_argument("--roles", type=str, default="", help="Comma-separated roles to target; overrides top-roles selection")
    ap.add_argument("--title_filter", type=str, default="", help="Normalized job title to include; others skipped")
    ap.add_argument("--require_basic", type=int, default=3)
    ap.add_argument("--require_bonus", type=int, default=3)
    ap.add_argument("--per_role_target", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="/home/zs7353/resume_validity/data/harvest")
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    log_path = out_dir / "harvest_log.jsonl"

    companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    posts = fetch_greenhouse_jobs(companies, max_jobs_per_company=args.max_jobs_per_company)
    if not posts:
        print("No posts fetched.")
        return

    # Local helpers to avoid heavy cross-imports
    import re
    def normalize_title(title: str) -> str:
        t = (title or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\b(senior|sr\.|staff|principal|lead|ii|iii|iv|v)\b", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t or "unknown"

    # Title-filtered mode: focus harvesting on a single normalized title
    if args.title_filter:
        target_norm = normalize_title(args.title_filter)
        def is_match(p_title: str) -> bool:
            cand = normalize_title(p_title)
            if cand == target_norm:
                return True
            # Relaxed matching: allow substring/prefix to include variants like
            # "software engineer - backend" under "software engineer"
            if cand.startswith(target_norm):
                return True
            if target_norm in cand:
                return True
            # Token subset: all target tokens must be present in candidate
            t_tokens = set(target_norm.split())
            c_tokens = set(cand.split())
            if t_tokens and t_tokens.issubset(c_tokens):
                return True
            return False
        filtered = [p for p in posts if is_match(p.title)]
        if not filtered:
            print(f"No postings matched title '{args.title_filter}' (normalized='{target_norm}').")
            return

        slug = target_norm.replace(" ", "_")
        passing_path = out_dir / f"passing_{slug}.jsonl"

        gemini = GeminiClient()
        count_written = 0
        # line-buffered for log
        log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
        writer = open(passing_path, "w", encoding="utf-8", buffering=1)
        try:
            print(f"Scanning title '{target_norm}': {len(filtered)} postings")
            for p in filtered:
                if count_written >= args.per_role_target:
                    break
                try:
                    quals = extract_qualifications(gemini, p.content_text)
                    basic = [q.text for q in (quals.get("basic") or []) if getattr(q, "text", "").strip()]
                    bonus = [q.text for q in (quals.get("bonus") or []) if getattr(q, "text", "").strip()]
                    bc, vc = len(basic), len(bonus)
                    row = {
                        "role": target_norm,
                        "title_norm": target_norm,
                        "original_role": map_title_to_role(p.title),
                        "source": p.source,
                        "company": p.company,
                        "title": p.title,
                        "url": p.url,
                        "basic": basic,
                        "bonus": bonus,
                        "basic_count": bc,
                        "validity_count": vc,
                        "pass": (bc >= args.require_basic and vc >= args.require_bonus),
                    }
                    log_fp.write(json.dumps(row) + "\n"); log_fp.flush()
                    if row["pass"]:
                        writer.write(json.dumps(row) + "\n"); writer.flush()
                        count_written += 1
                        print(f"PASS {target_norm}: bc={bc} vc={vc} url={p.url}")
                except Exception as e:
                    log_fp.write(json.dumps({
                        "title_norm": target_norm, "url": p.url, "error": str(e)
                    }) + "\n")
                    print(f"ERROR {target_norm}: {e} url={p.url}")
        finally:
            try:
                writer.close()
            except Exception:
                pass
            log_fp.close()

        summary = {
            "title_norm": target_norm,
            "per_title_target": args.per_role_target,
            "require_basic": args.require_basic,
            "require_bonus": args.require_bonus,
            "count": count_written,
        }
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)
        print("\nHarvest summary saved:")
        print(summary_path)
        print(json.dumps(summary, indent=2))
        return

    if args.roles:
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    elif args.restrict_top_roles:
        roles = pick_top_roles(posts, top_k=5, max_cs_roles=2)
        print(f"Restricting to top-5 roles (<=2 CS): {roles}")
    else:
        # Default to all roles present
        roles = sorted({map_title_to_role(p.title) for p in posts})
    buckets = bucket_by_role(posts)

    gemini = GeminiClient()
    role_counts: Dict[str, int] = defaultdict(int)
    role_targets: Dict[str, int] = {r: args.per_role_target for r in roles}
    writers: Dict[str, object] = {}
    for r in roles:
        # line-buffered to ensure logs appear while running
        writers[r] = open(out_dir / f"passing_{r.replace(' ', '_').lower()}.jsonl", "w", encoding="utf-8", buffering=1)
    # line-buffered for log as well
    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)

    try:
        for r in roles:
            if role_counts[r] >= role_targets[r]:
                continue
            bucket = buckets.get(r, [])
            if not bucket:
                print(f"No postings for role {r}")
                continue
            print(f"Scanning role {r}: {len(bucket)} postings")
            for p in bucket:
                if role_counts[r] >= role_targets[r]:
                    break
                try:
                    quals = extract_qualifications(gemini, p.content_text)
                    basic = [q.text for q in (quals.get("basic") or []) if getattr(q, "text", "").strip()]
                    bonus = [q.text for q in (quals.get("bonus") or []) if getattr(q, "text", "").strip()]
                    bc, vc = len(basic), len(bonus)
                    row = {
                        "role": r,
                        "source": p.source,
                        "company": p.company,
                        "title": p.title,
                        "url": p.url,
                        "basic": basic,
                        "bonus": bonus,
                        "basic_count": bc,
                        "validity_count": vc,
                        "pass": (bc >= args.require_basic and vc >= args.require_bonus),
                    }
                    log_fp.write(json.dumps(row) + "\n"); log_fp.flush()
                    if row["pass"]:
                        writers[r].write(json.dumps(row) + "\n"); writers[r].flush()
                        role_counts[r] += 1
                        print(f"PASS {r}: bc={bc} vc={vc} url={p.url}")
                except Exception as e:
                    log_fp.write(json.dumps({
                        "role": r, "url": p.url, "error": str(e)
                    }) + "\n")
                    print(f"ERROR {r}: {e} url={p.url}")
    finally:
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass
        log_fp.close()

    summary = {
        "roles": roles,
        "per_role_target": args.per_role_target,
        "require_basic": args.require_basic,
        "require_bonus": args.require_bonus,
        "counts": role_counts,
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)
    print("\nHarvest summary saved:")
    print(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


