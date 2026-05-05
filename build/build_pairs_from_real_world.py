#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import asdict
from json import JSONDecoder
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.resume_builder import (
    build_reworded_equivalent_resume,
    build_resume_from_real_profile,
    build_underqualified_resume_from_pool_k,
    build_preferred_resume_from_pool_k,
    build_reworded_with_awards_extracurriculars,
)

from build_pairs_from_harvest import (
    DEFAULT_COMPANIES,
    Posting,
    PairRecord,
    canonicalize_role_title,
    load_demographic_indicators,
    load_names_db,
    load_schools_list,
    inject_names,
    replace_placeholders,
    DEMOG_ORDER,
    pick_demographic_name,
    ROLE_COMPANY_HINTS,
    pick_indicator,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build resume pairs from manually collected real-world resumes")
    p.add_argument("--job_desc_files", nargs="+", required=True, help="Paths to JSON/JSONL files with job descriptions")
    p.add_argument("--resume_csvs", nargs="+", required=True, help="Paths to CSVs containing Resume_str and Link columns")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to write per-job pair files")
    p.add_argument("--names_json", type=str, default="/home/zs7353/resume-fting/data/names.json")
    p.add_argument("--indicators_json", type=str, default="/home/zs7353/resume-fting/data/broad_demographic_indicators.json")
    p.add_argument("--job_url", type=str, default=None, help="If set, only process the job with this URL (normalized ignoring query params)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--toy_limit", type=int, default=0, help="Limit the number of jobs processed (for quick testing)")
    return p.parse_args()


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")

    query = parsed.query
    if "indeed.com" in netloc:
        qs = dict(parse_qsl(parsed.query))
        if "jk" in qs:
            query = urlencode({"jk": qs["jk"]})
        else:
            query = ""
    elif "linkedin.com" in netloc:
        query = ""  # job id already in the path
    else:
        query = parsed.query

    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=netloc,
        path=path or "/",
        query=query,
        fragment="",
    )
    return urlunparse(normalized)


def slugify(value: str, fallback: str = "posting") -> str:
    value = value or fallback
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def load_job_descriptions(paths: List[str]) -> List[Dict[str, Any]]:
    decoder = JSONDecoder()
    jobs: List[Dict[str, Any]] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        idx = 0
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            obj, idx = decoder.raw_decode(text, idx)
            jobs.append(obj)
    return jobs


def load_resumes(paths: List[str]) -> Dict[str, Dict[str, Any]]:
    resume_by_url: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                link = normalize_url(row.get("Link", ""))
                if not link:
                    continue
                resume_by_url[link] = row
    return resume_by_url


def make_posting(obj: Dict[str, Any]) -> Posting:
    return Posting(
        role=str(obj.get("role") or obj.get("title_norm") or obj.get("title") or "role"),
        title_norm=str(obj.get("title_norm") or obj.get("role") or ""),
        original_role=str(obj.get("original_role") or ""),
        source=str(obj.get("source") or "real_world"),
        company=str(obj.get("company") or ""),
        title=str(obj.get("title") or ""),
        url=str(obj.get("url") or ""),
        basic=list(obj.get("basic") or []),
        bonus=list(obj.get("bonus") or []),
    )


def write_pairs(
    fout,
    ferr,
    records: List[PairRecord],
) -> int:
    written = 0
    for pr in records:
        if pr.base_resume and pr.variant_resume:
            fout.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
            written += 1
        else:
            ferr.write(json.dumps({
                "job_title": pr.job_title,
                "job_source": pr.job_source,
                "demographics": pr.demographics,
                "experiment_type": pr.experiment_type,
                "error": "missing_resume_content",
            }, ensure_ascii=False) + "\n")
    fout.flush()
    ferr.flush()
    return written


def _normalize_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def _contains(md: str, phrase: str) -> bool:
    return _normalize_text(phrase) in _normalize_text(md)


def gen_realworld_validity_pairs(
    gemini: GeminiClient,
    rng: random.Random,
    post: Posting,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
    base_resume_md: str,
) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    canonical_role = canonicalize_role_title(post.role)
    # Choose demographics (same group for base/variant)
    demogs = [code for code, _ in DEMOG_ORDER]
    demog_code = rng.choice(demogs)
    base_name = pick_demographic_name(names_db, demog_code, rng, gender_key=demog_code.split("_")[1])
    var_name = pick_demographic_name(names_db, demog_code, rng, gender_key=demog_code.split("_")[1])

    # Role-consistent company/school placeholders
    role_key = canonical_role.lower()
    if role_key in ROLE_COMPANY_HINTS and ROLE_COMPANY_HINTS[role_key]:
        company_choice = rng.choice(ROLE_COMPANY_HINTS[role_key])
    else:
        company_choice = rng.choice(companies)
    school_choice = rng.choice(schools)

    # Underqualified: k in [1..min(3, len(basic))]
    max_k_basic = min(3, len(post.basic))
    for k in range(1, max_k_basic + 1):
        # Up to 3 attempts to get a variant that actually lacks k items
        built = None
        for _ in range(3):
            try:
                var_md = build_underqualified_resume_from_pool_k(gemini, canonical_role, base_resume_md, post.basic, k)
                # Naive verification: pick a random subset of basics and check at least k are now missing (approximate)
                # Since we don't know which ones Gemini removed, verify that the count of basics clearly present decreased.
                present_count = sum(1 for q in post.basic if _contains(var_md, q))
                if present_count <= max(0, len(post.basic) - k):
                    built = var_md
                    break
            except Exception as e:
                import traceback
                print(f"[WARN] build_underqualified_resume_from_pool_k failed (k={k}): {e}")
                traceback.print_exc()
                continue
        if not built:
            print(f"[WARN] underqualified k={k} failed verification; skipping")
            continue
        base_named = replace_placeholders(inject_names(base_resume_md, base_name), company_choice, school_choice)
        var_named = replace_placeholders(inject_names(built, var_name), company_choice, school_choice)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base_named,
            variant_resume=var_named,
            pair_type="underqualified",
            differed_qualifications=[],  # not enumerating exact items; Gemini chooses
            num_differed=k,
            better="first",
            demographics=(demog_code, demog_code),
            experiment_type="validity",
        ))

    # Preferred: k in [1..min(3, len(bonus))]
    max_k_bonus = min(3, len(post.bonus))
    for k in range(1, max_k_bonus + 1):
        built = None
        for _ in range(3):
            try:
                var_md = build_preferred_resume_from_pool_k(gemini, canonical_role, base_resume_md, post.bonus, k)
                # Naive verification: ensure at least k bonus phrases are now present
                add_count = sum(1 for q in post.bonus if _contains(var_md, q) and not _contains(base_resume_md, q))
                if add_count >= min(k, len(post.bonus)):
                    built = var_md
                    break
            except Exception as e:
                import traceback
                print(f"[WARN] build_preferred_resume_from_pool_k failed (k={k}): {e}")
                traceback.print_exc()
                continue
        if not built:
            print(f"[WARN] preferred k={k} failed verification; skipping")
            continue
        base_named = replace_placeholders(inject_names(base_resume_md, base_name), company_choice, school_choice)
        var_named = replace_placeholders(inject_names(built, var_name), company_choice, school_choice)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base_named,
            variant_resume=var_named,
            pair_type="preferred",
            differed_qualifications=[],
            num_differed=k,
            better="second",
            demographics=(demog_code, demog_code),
            experiment_type="validity",
        ))

    # Reworded equal (2 times)
    for _ in range(2):
        try:
            var = build_reworded_equivalent_resume(gemini, canonical_role, base_resume_md)
        except Exception as e:
            import traceback
            print(f"[WARN] build_reworded_equivalent_resume failed: {e}")
            traceback.print_exc()
            continue
        base_named = replace_placeholders(inject_names(base_resume_md, base_name), company_choice, school_choice)
        var_named = replace_placeholders(inject_names(var, var_name), company_choice, school_choice)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base_named,
            variant_resume=var_named,
            pair_type="reworded",
            differed_qualifications=[],
            num_differed=0,
            better="equal",
            demographics=(demog_code, demog_code),
            experiment_type="validity",
        ))
    return pairs


def gen_realworld_fairness_pairs_equal(
    gemini: GeminiClient,
    rng: random.Random,
    post: Posting,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
    indicators: Dict[str, Any],
    implicit: bool,
    base_resume_md: str,
    reworded_resume_md: str,
) -> List[PairRecord]:
    """Generate fairness pairs (equal qualification, different demographics) using pre-built base/reworded resumes."""
    pairs: List[PairRecord] = []
    if not post.basic:
        return pairs
    canonical_role = canonicalize_role_title(post.role)
    
    # For fairness: pairwise demographic comparisons across 4 groups (ordered)
    codes = [c for c, _ in DEMOG_ORDER]
    # 4x4 comparisons
    for a in codes:
        for b in codes:
            name_a = pick_demographic_name(names_db, a, rng, gender_key=a.split("_")[1])
            name_b = pick_demographic_name(names_db, b, rng, gender_key=b.split("_")[1])
            company_choice = rng.choice(companies)
            school_choice = rng.choice(schools)

            # Build per-pair with retries (>=3) and continue-on-error
            def try_build_pair() -> Optional[tuple]:
                for attempt in range(1, 5):  # 4 attempts total
                    try:
                        if implicit:
                            aw_a, org_a = pick_indicator(indicators, a, post.role)
                            aw_b, org_b = pick_indicator(indicators, b, post.role)
                            base_md = build_reworded_with_awards_extracurriculars(gemini, canonical_role, base_resume_md, aw_a, org_a)
                            var_md  = build_reworded_with_awards_extracurriculars(gemini, canonical_role, reworded_resume_md, aw_b, org_b)
                            base_named = replace_placeholders(base_md, company_choice, school_choice)
                            var_named  = replace_placeholders(var_md, company_choice, school_choice)
                        else:
                            base_named = replace_placeholders(inject_names(base_resume_md, name_a), company_choice, school_choice)
                            var_named  = replace_placeholders(inject_names(reworded_resume_md, name_b), company_choice, school_choice)
                        return base_named, var_named
                    except Exception as e:
                        print(f"[fairness retry] role={canonical_role} a={a} b={b} attempt={attempt} error={e}")
                        continue
                return None

            built = try_build_pair()
            if not built:
                # log skip by raising a lightweight record later in main loop
                pairs.append(PairRecord(
                    job_title=canonical_role,
                    job_source={"company": post.company, "title": post.title, "url": post.url},
                    base_resume="",
                    variant_resume="",
                    pair_type="reworded",
                    differed_qualifications=[],
                    num_differed=0,
                    better="equal",
                    demographics=(a, b),
                    experiment_type=("implicit_demographics_fairness" if implicit else "fairness"),
                ))
                continue
            base_named, var_named = built

            pairs.append(PairRecord(
                job_title=canonical_role,
                job_source={"company": post.company, "title": post.title, "url": post.url},
                base_resume=base_named,
                variant_resume=var_named,
                pair_type="reworded",
                differed_qualifications=[],
                num_differed=0,
                better="equal",
                demographics=(a, b),
                experiment_type=("implicit_demographics_fairness" if implicit else "fairness"),
            ))
    return pairs


def gen_realworld_validity_pairs_demographics(
    gemini: GeminiClient,
    rng: random.Random,
    post: Posting,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
    base_resume_md: str,
) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    canonical_role = canonicalize_role_title(post.role)
    # Independent demographics
    codes = [c for c, _ in DEMOG_ORDER]
    for k in range(1, min(3, len(post.basic)) + 1):
        try:
            var = build_underqualified_resume_from_pool_k(gemini, canonical_role, base_resume_md, post.basic, k)
        except Exception as e:
            import traceback
            print(f"[WARN] demo underqualified k={k} build failed: {e}")
            traceback.print_exc()
            continue
        a = rng.choice(codes); b = rng.choice(codes)
        name_a = pick_demographic_name(names_db, a, rng, gender_key=a.split("_")[1])
        name_b = pick_demographic_name(names_db, b, rng, gender_key=b.split("_")[1])
        role_key = canonical_role.lower()
        company_choice = rng.choice(ROLE_COMPANY_HINTS[role_key]) if role_key in ROLE_COMPANY_HINTS and ROLE_COMPANY_HINTS[role_key] else rng.choice(companies)
        school_choice = rng.choice(schools)
        base_named = replace_placeholders(inject_names(base_resume_md, name_a), company_choice, school_choice)
        var_named = replace_placeholders(inject_names(var, name_b), company_choice, school_choice)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base_named,
            variant_resume=var_named,
            pair_type="underqualified",
            differed_qualifications=[],
            num_differed=k,
            better="first",
            demographics=(a, b),
            experiment_type="validity_demographics",
        ))
    for k in range(1, min(3, len(post.bonus)) + 1):
        try:
            var = build_preferred_resume_from_pool_k(gemini, canonical_role, base_resume_md, post.bonus, k)
        except Exception as e:
            import traceback
            print(f"[WARN] demo preferred k={k} build failed: {e}")
            traceback.print_exc()
            continue
        a = rng.choice(codes); b = rng.choice(codes)
        name_a = pick_demographic_name(names_db, a, rng, gender_key=a.split("_")[1])
        name_b = pick_demographic_name(names_db, b, rng, gender_key=b.split("_")[1])
        role_key = canonical_role.lower()
        company_choice = rng.choice(ROLE_COMPANY_HINTS[role_key]) if role_key in ROLE_COMPANY_HINTS and ROLE_COMPANY_HINTS[role_key] else rng.choice(companies)
        school_choice = rng.choice(schools)
        base_named = replace_placeholders(inject_names(base_resume_md, name_a), company_choice, school_choice)
        var_named = replace_placeholders(inject_names(var, name_b), company_choice, school_choice)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base_named,
            variant_resume=var_named,
            pair_type="preferred",
            differed_qualifications=[],
            num_differed=k,
            better="second",
            demographics=(a, b),
            experiment_type="validity_demographics",
        ))
    # Reworded equal (2 times)
    for _ in range(2):
        try:
            var = build_reworded_equivalent_resume(gemini, canonical_role, base_resume_md)
        except Exception as e:
            import traceback
            print(f"[WARN] demo reworded equal build failed: {e}")
            traceback.print_exc()
            continue
        a = rng.choice(codes); b = rng.choice(codes)
        name_a = pick_demographic_name(names_db, a, rng, gender_key=a.split("_")[1])
        name_b = pick_demographic_name(names_db, b, rng, gender_key=b.split("_")[1])
        role_key = canonical_role.lower()
        company_choice = rng.choice(ROLE_COMPANY_HINTS[role_key]) if role_key in ROLE_COMPANY_HINTS and ROLE_COMPANY_HINTS[role_key] else rng.choice(companies)
        school_choice = rng.choice(schools)
        base_named = replace_placeholders(inject_names(base_resume_md, name_a), company_choice, school_choice)
        var_named = replace_placeholders(inject_names(var, name_b), company_choice, school_choice)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base_named,
            variant_resume=var_named,
            pair_type="reworded",
            differed_qualifications=[],
            num_differed=0,
            better="equal",
            demographics=(a, b),
            experiment_type="validity_demographics",
        ))
    return pairs


def main() -> None:
    args = parse_args()
    print(f"[DEBUG] args: {args}")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = load_job_descriptions(args.job_desc_files)
    print(f"[DEBUG] loaded job descriptions: {len(jobs)}")
    resumes = load_resumes(args.resume_csvs)
    print(f"[DEBUG] loaded resumes: {len(resumes)}")
    names_db = load_names_db(Path(args.names_json))
    indicators = load_demographic_indicators(Path(args.indicators_json))
    schools = load_schools_list()
    companies = DEFAULT_COMPANIES
    print("[DEBUG] creating Gemini client...")
    gemini = GeminiClient()
    print(f"[DEBUG] Gemini health: {gemini.healthcheck()}")
    rng = random.Random(args.seed)

    url_filter = normalize_url(args.job_url) if args.job_url else None
    selected: List[Dict[str, Any]] = []
    for obj in jobs:
        url_norm = normalize_url(obj.get("url", ""))
        if url_filter and url_norm != url_filter:
            continue
        selected.append(obj)
    print(f"[DEBUG] url_filter={url_filter} selected_count={len(selected)}")

    if not selected:
        print("[ERROR] No job descriptions matched the provided criteria.", file=sys.stderr)
        sys.exit(1)

    processed = 0
    for idx, obj in enumerate(selected, 1):
        url_norm = normalize_url(obj.get("url", ""))
        resume_row = resumes.get(url_norm)
        print(f"[DEBUG] processing idx={idx} title={obj.get('title')} url_norm={url_norm} resume_found={bool(resume_row)}")
        if not resume_row:
            print(f"[WARN] No resume found for {obj.get('title')} ({obj.get('url')})")
            continue
        posting = make_posting(obj)
        canonical_role = canonicalize_role_title(posting.role)
        real_resume_text = resume_row.get("Resume_str", "")
        if not real_resume_text.strip():
            print(f"[WARN] Empty Resume_str for {posting.title} ({posting.url})")
            continue

        print(f"[INFO] Processing '{posting.title}' ({posting.company})")
        try:
            base_resume_md = build_resume_from_real_profile(
                gemini,
                canonical_role,
                real_resume_text,
                posting.basic,
            )
            print("[DEBUG] built base resume from real profile (ensured required X present)")
            reworded_md = build_reworded_equivalent_resume(gemini, canonical_role, base_resume_md)
            print("[DEBUG] built reworded resume")
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed building base/reworded resumes: {e}")
            traceback.print_exc()
            continue

        role_slug = slugify(posting.title_norm or posting.role)
        comp_slug = slugify(f"{posting.company}_{posting.title}")
        job_dir = out_root / role_slug / comp_slug
        job_dir.mkdir(parents=True, exist_ok=True)
        pairs_path = job_dir / "pairs_shard_00_of_01.jsonl"
        errors_path = job_dir / "errors_shard_00_of_01.jsonl"

        with open(pairs_path, "w", encoding="utf-8") as fout, open(errors_path, "w", encoding="utf-8") as ferr:
            written = 0
            try:
                validity_pairs = gen_realworld_validity_pairs(
                    gemini,
                    rng,
                    posting,
                    names_db,
                    schools,
                    companies,
                    base_resume_md=base_resume_md,
                )
            except Exception as e:
                import traceback
                print(f"[ERROR] gen_realworld_validity_pairs failed: {e}")
                traceback.print_exc()
                validity_pairs = []
            written += write_pairs(fout, ferr, validity_pairs)

            try:
                validity_demo_pairs = gen_realworld_validity_pairs_demographics(
                    gemini,
                    rng,
                    posting,
                    names_db,
                    schools,
                    companies,
                    base_resume_md=base_resume_md,
                )
            except Exception as e:
                import traceback
                print(f"[ERROR] gen_realworld_validity_pairs_demographics failed: {e}")
                traceback.print_exc()
                validity_demo_pairs = []
            written += write_pairs(fout, ferr, validity_demo_pairs)

            try:
                fairness_pairs = gen_realworld_fairness_pairs_equal(
                    gemini,
                    rng,
                    posting,
                    names_db,
                    schools,
                    companies,
                    indicators,
                    implicit=False,
                    base_resume_md=base_resume_md,
                    reworded_resume_md=reworded_md,
                )
            except Exception as e:
                import traceback
                print(f"[ERROR] gen_realworld_fairness_pairs_equal (explicit) failed: {e}")
                traceback.print_exc()
                fairness_pairs = []
            written += write_pairs(fout, ferr, fairness_pairs)

            try:
                fairness_pairs_implicit = gen_realworld_fairness_pairs_equal(
                    gemini,
                    rng,
                    posting,
                    names_db,
                    schools,
                    companies,
                    indicators,
                    implicit=True,
                    base_resume_md=base_resume_md,
                    reworded_resume_md=reworded_md,
                )
            except Exception as e:
                import traceback
                print(f"[ERROR] gen_realworld_fairness_pairs_equal (implicit) failed: {e}")
                traceback.print_exc()
                fairness_pairs_implicit = []
            written += write_pairs(fout, ferr, fairness_pairs_implicit)

        print(f"[INFO] Wrote {written} pairs -> {pairs_path}")
        processed += 1
        if args.toy_limit and processed >= args.toy_limit:
            print("[INFO] Toy limit reached, stopping early.")
            break


if __name__ == "__main__":
    main()


