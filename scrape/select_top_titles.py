#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from resume_validity.scrape.greenhouse_scraper import (
    fetch_greenhouse_jobs,
    map_title_to_role,
)


def normalize_title(title: str) -> str:
    t = (title or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\b(senior|sr\.|staff|principal|lead|ii|iii|iv|v)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or "unknown"


def title_is_cs_related(title: str) -> bool:
    role = map_title_to_role(title)
    return role in {"Software Engineer", "ML Engineer", "Data Scientist"}


def select_job_titles(posts, top_cs_limit: int = 2, top5_count: int = 5, next_count: int = 20) -> Tuple[List[str], List[str]]:
    counts: Dict[str, int] = {}
    for p in posts:
        key = normalize_title(p.title)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    top5: List[str] = []
    next20: List[str] = []
    cs_selected = 0
    for key, _ in ranked:
        is_cs = title_is_cs_related(key)
        if len(top5) < top5_count:
            if is_cs and cs_selected >= top_cs_limit:
                continue
            top5.append(key)
            if is_cs:
                cs_selected += 1
            continue
        if len(next20) < next_count and key not in top5:
            next20.append(key)
        if len(top5) >= top5_count and len(next20) >= next_count:
            break
    return top5, next20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=str, default="")
    ap.add_argument("--companies_file", type=str, default="/home/zs7353/resume_validity/data/companies_greenhouse.txt")
    ap.add_argument("--out_file", type=str, default="/home/zs7353/resume_validity/data/selected_titles_25.txt")
    ap.add_argument("--manifest", type=str, default="/home/zs7353/resume_validity/data/selected_titles_manifest.json")
    ap.add_argument("--max_jobs_per_company", type=int, default=200)
    ap.add_argument("--min_top5_count", type=int, default=20)
    ap.add_argument("--min_next20_count", type=int, default=5)
    args = ap.parse_args()

    companies: List[str]
    if args.companies:
        companies = [c.strip() for c in args.companies.split(",") if c.strip()]
    else:
        path = Path(args.companies_file)
        if not path.exists():
            raise SystemExit(f"Companies file not found: {path}")
        companies = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    posts = fetch_greenhouse_jobs(companies, max_jobs_per_company=args.max_jobs_per_company)
    if not posts:
        raise SystemExit("No posts fetched from Greenhouse; cannot select titles.")

    # Count titles
    counts: Dict[str, int] = {}
    for p in posts:
        key = normalize_title(p.title)
        counts[key] = counts.get(key, 0) + 1
    # Filter by minimum counts
    eligible_top = [t for t, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if c >= args.min_top5_count]
    # Apply CS constraint while selecting top5
    top5: List[str] = []
    cs_selected = 0
    for t in eligible_top:
        is_cs = title_is_cs_related(t)
        if len(top5) < 5:
            if is_cs and cs_selected >= 2:
                continue
            top5.append(t)
            if is_cs:
                cs_selected += 1
        if len(top5) == 5:
            break
    # If still not enough, backfill from remaining highest counts regardless of min_top5_count but respecting CS cap
    if len(top5) < 5:
        for t, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            if t in top5:
                continue
            is_cs = title_is_cs_related(t)
            if is_cs and cs_selected >= 2:
                continue
            top5.append(t)
            if is_cs:
                cs_selected += 1
            if len(top5) == 5:
                break

    # Next 20 titles with at least min_next20_count and not in top5
    candidates_next = [t for t, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if t not in top5 and c >= args.min_next20_count]
    next20 = candidates_next[:20]
    # If fewer than 20, backfill with remaining by count
    if len(next20) < 20:
        for t, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            if t in top5 or t in next20:
                continue
            next20.append(t)
            if len(next20) == 20:
                break

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in top5 + next20:
            f.write(t + "\n")

    manifest = {
        "companies": companies,
        "top5_titles": top5,
        "next20_titles": next20,
        "total": len(top5) + len(next20),
        "min_top5_count": args.min_top5_count,
        "min_next20_count": args.min_next20_count,
        "counts": counts,
    }
    with open(args.manifest, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Wrote {len(top5) + len(next20)} titles to {out_path}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()


