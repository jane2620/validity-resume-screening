#!/usr/bin/env python3
"""
Build resume pairs by randomly pairing Reddit resumes with job descriptions.

This script:
1. Loads real resumes from Reddit (finance/SWE)
2. Loads job descriptions from existing harvest files
3. Randomly pairs each job with N resumes
4. Uses LLM for LIGHT cleanup only (HTML to Markdown) - preserving original content
5. Generates validity/fairness pairs using the cleaned resume as base
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resume_validity.llm.openrouter_client import OpenRouterClient
from resume_validity.llm.gemini_client import GeminiClient

# Import shared utilities from build_pairs_from_harvest
from resume_validity.build.build_pairs_from_harvest import (
    DEMOG_ORDER,
    DEFAULT_COMPANIES,
    ROLE_COMPANY_HINTS,
    Posting,
    PairRecord,
    canonicalize_role_title,
    choose_k,
    inject_names,
    load_demographic_indicators,
    load_names_db,
    load_schools_list,
    pick_demographic_name,
    pick_indicator,
    replace_placeholders,
)

# Import resume builder functions for variants
from resume_validity.llm.resume_builder_claude import (
    BASIC_RESUME_SYSTEM,
    build_reworded_equivalent_resume,
    build_reworded_with_awards_extracurriculars,
)


# ---------------------------------------------------------------------------
# Light cleanup function - preserves original content, only formats
# ---------------------------------------------------------------------------

LIGHT_CLEANUP_SYSTEM = """
You convert resumes into clean Markdown format. You must preserve ALL original content exactly as-is.
Do NOT add, remove, or modify any qualifications, experiences, or skills. Only format the text.
""".strip()


LLMClient = Union[OpenRouterClient, GeminiClient]


def light_cleanup_resume(client: LLMClient, raw_resume_text: str, role_title: str) -> str:
    """
    Convert a raw resume (possibly with HTML) into clean Markdown.
    
    CRITICAL: This function ONLY does formatting cleanup. It does NOT:
    - Add any qualifications
    - Remove any qualifications
    - Modify the substance of any experience/skill
    - "Enhance" or "improve" the resume
    
    It ONLY:
    - Strips HTML tags
    - Formats into clean Markdown sections
    - Adds the {{CANDIDATE_NAME}} placeholder at the top
    - Adds {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders for anonymization
    """
    placeholder = "{{CANDIDATE_NAME}}"
    user = f"""Convert the following raw resume text into clean Markdown format.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. PRESERVE ALL ORIGINAL CONTENT EXACTLY. Do not add, remove, or modify any:
   - Job experiences or responsibilities
   - Skills or qualifications
   - Education details
   - Achievements or certifications
   - Dates or durations
   
2. ONLY do formatting cleanup:
   - Strip HTML tags (convert to plain text)
   - Organize into clear Markdown sections (Summary, Experience, Education, Skills, etc.)
   - Use consistent bullet points and headers
   
3. For anonymization:
   - Replace the candidate's name at the top with: Name: {placeholder}
   - Replace specific company names with: {{{{COMPANY_NAME}}}}
   - Replace specific school/university names with: {{{{SCHOOL_NAME}}}}
   - Keep job titles, teams, and role descriptions intact

4. Do NOT include any contact information (email, phone, LinkedIn, address).

5. Keep the resume length approximately the same - do not truncate or expand.

Target role context (for section organization only, NOT for adding content): {role_title}

Raw resume text:
{raw_resume_text.strip()}
"""
    return client.complete_text(LIGHT_CLEANUP_SYSTEM, user, temperature=0.2, max_tokens=4096)


def build_underqualified_resume_from_real(
    client: LLMClient,
    role_title: str,
    base_resume_md: str,
    qualifications_to_remove: List[str],
) -> str:
    """
    Create an underqualified variant by removing specified qualifications.
    
    Note: Since we're working with real resumes, the qualifications may not be
    explicitly present. The LLM should remove related experience/skills if present,
    or return the resume mostly unchanged if the qualification isn't there.
    """
    user = (
        f"Given the resume below, create an UNDER-QUALIFIED variant by REMOVING or DOWNGRADING "
        f"any experience/skills related to these {len(qualifications_to_remove)} qualifications:\n"
        + "\n".join([f"- {q}" for q in qualifications_to_remove]) + "\n\n"
        "Instructions:\n"
        "- If the resume has experience matching these qualifications, remove or reduce it.\n"
        "- If the resume doesn't clearly have these qualifications, make minimal changes.\n"
        "- Do NOT add any new qualifications or experience.\n"
        "- Keep 'Name: {{CANDIDATE_NAME}}' and the {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "- Do NOT introduce any contact info (no email/LinkedIn/GitHub/phone).\n"
        "- Keep overall structure and length similar to the original.\n\n"
        f"Resume:\n{base_resume_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.3, max_tokens=4096)


def build_preferred_resume_from_real(
    client: LLMClient,
    role_title: str,
    base_resume_md: str,
    qualifications_to_add: List[str],
) -> str:
    """
    Create a preferred variant by adding specified bonus qualifications.
    
    These are added as additional experience/skills that enhance the resume.
    """
    user = (
        f"Given the resume below, create a PREFERRED variant by ADDING experience/skills "
        f"related to these {len(qualifications_to_add)} bonus qualifications:\n"
        + "\n".join([f"- {q}" for q in qualifications_to_add]) + "\n\n"
        "Instructions:\n"
        "- Add these qualifications naturally into the experience or skills sections.\n"
        "- Keep all existing content intact.\n"
        "- Keep 'Name: {{CANDIDATE_NAME}}' and the {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "- Do NOT introduce any contact info (no email/LinkedIn/GitHub/phone).\n"
        "- Keep overall structure similar, slight length increase is acceptable.\n\n"
        f"Resume:\n{base_resume_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.35, max_tokens=4096)


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------

@dataclass
class RedditResume:
    id: str
    role: str
    link: str
    resume_text: str


def load_reddit_resumes(path: Path) -> List[RedditResume]:
    """Load resumes from a Reddit file (handles Python-style triple-quoted strings)."""
    import ast
    resumes = []
    
    text = path.read_text(encoding="utf-8")
    idx = 0
    while idx < len(text):
        # Skip whitespace
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        
        # Find the next opening brace
        if text[idx] != '{':
            idx += 1
            continue
            
        # Find the matching closing brace by counting braces
        brace_count = 0
        end_idx = idx
        in_string = False
        triple_quote = False
        i = idx
        while i < len(text):
            # Handle triple-quoted strings
            if text[i:i+3] == '"""':
                if not in_string:
                    in_string = True
                    triple_quote = True
                    i += 3
                    continue
                elif triple_quote:
                    in_string = False
                    triple_quote = False
                    i += 3
                    continue
            # Handle regular strings
            elif text[i] == '"' and not triple_quote:
                if not in_string:
                    in_string = True
                else:
                    # Check for escape
                    if i > 0 and text[i-1] != '\\':
                        in_string = False
            
            if not in_string:
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            i += 1
        
        if brace_count != 0:
            print(f"[WARN] Unbalanced braces at position {idx}, skipping")
            idx = end_idx if end_idx > idx else idx + 1
            continue
            
        obj_str = text[idx:end_idx]
        try:
            # Use ast.literal_eval to handle Python-style strings
            obj = ast.literal_eval(obj_str)
            resumes.append(RedditResume(
                id=str(obj.get("ID", "")),
                role=str(obj.get("role", "")),
                link=str(obj.get("link", "")),
                resume_text=str(obj.get("resume_text", "")),
            ))
        except Exception as e:
            print(f"[WARN] Failed to parse resume at position {idx}: {e}")
            traceback.print_exc()
        
        idx = end_idx
    
    return resumes


def load_job_descriptions(path: Path) -> List[Posting]:
    """Load job descriptions from a JSON/JSONL file (handles multi-line JSON objects)."""
    from json import JSONDecoder
    decoder = JSONDecoder()
    postings = []
    
    text = path.read_text(encoding="utf-8")
    idx = 0
    while idx < len(text):
        # Skip whitespace
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            idx = end_idx
            postings.append(Posting(
                role=str(obj.get("role") or obj.get("title_norm") or ""),
                title_norm=str(obj.get("title_norm") or ""),
                original_role=str(obj.get("original_role") or ""),
                source=str(obj.get("source") or ""),
                company=str(obj.get("company") or ""),
                title=str(obj.get("title") or ""),
                url=str(obj.get("url") or ""),
                basic=list(obj.get("basic") or []),
                bonus=list(obj.get("bonus") or []),
            ))
        except Exception as e:
            print(f"[WARN] Failed to parse job at position {idx}: {e}")
            traceback.print_exc()
            # Try to skip to the next opening brace
            next_brace = text.find("{", idx + 1)
            if next_brace == -1:
                break
            idx = next_brace
            continue
    return postings


# ---------------------------------------------------------------------------
# Pair generation functions
# ---------------------------------------------------------------------------

def gen_validity_pairs_from_real(
    client: LLMClient,
    rng: random.Random,
    post: Posting,
    base_resume_md: str,
    resume_id: str,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
) -> List[PairRecord]:
    """Generate validity pairs using a real resume as the base."""
    pairs: List[PairRecord] = []
    if not post.basic:
        print(f"[WARN] No basic qualifications for job {post.title}, skipping validity pairs")
        return pairs

    canonical_role = canonicalize_role_title(post.role)
    
    # For validity, both resumes have same demographic
    demogs = [code for code, _ in DEMOG_ORDER]
    demog_code = rng.choice(demogs)
    base_name = pick_demographic_name(names_db, demog_code, rng, gender_key=demog_code.split("_")[1])
    var_name = pick_demographic_name(names_db, demog_code, rng, gender_key=demog_code.split("_")[1])

    max_k_basic = min(3, len(post.basic))
    max_k_bonus = min(3, len(post.bonus))

    # Underqualified variants (remove k basic qualifications)
    for k in range(1, max_k_basic + 1):
        removed = choose_k(post.basic, k, rng)
        try:
            var = build_underqualified_resume_from_real(client, canonical_role, base_resume_md, removed)
            pairs.append(PairRecord(
                job_title=canonical_role,
                job_source={"company": post.company, "title": post.title, "url": post.url, "resume_id": resume_id},
                base_resume=base_resume_md,
                variant_resume=var,
                pair_type="underqualified",
                differed_qualifications=removed,
                num_differed=len(removed),
                better="first",
                demographics=(demog_code, demog_code),
                experiment_type="validity",
            ))
        except Exception as e:
            print(f"[ERROR] build_underqualified_resume_from_real failed (k={k}): {e}")
            traceback.print_exc()
            continue

    # Preferred variants (add k bonus qualifications)
    for k in range(1, max_k_bonus + 1):
        added = choose_k(post.bonus, k, rng)
        if not added:
            continue
        try:
            var = build_preferred_resume_from_real(client, canonical_role, base_resume_md, added)
            pairs.append(PairRecord(
                job_title=canonical_role,
                job_source={"company": post.company, "title": post.title, "url": post.url, "resume_id": resume_id},
                base_resume=base_resume_md,
                variant_resume=var,
                pair_type="preferred",
                differed_qualifications=added,
                num_differed=len(added),
                better="second",
                demographics=(demog_code, demog_code),
                experiment_type="validity",
            ))
        except Exception as e:
            print(f"[ERROR] build_preferred_resume_from_real failed (k={k}): {e}")
            traceback.print_exc()
            continue

    # Reworded equal (2 times)
    for i in range(2):
        try:
            var = build_reworded_equivalent_resume(client, canonical_role, base_resume_md)
            pairs.append(PairRecord(
                job_title=canonical_role,
                job_source={"company": post.company, "title": post.title, "url": post.url, "resume_id": resume_id},
                base_resume=base_resume_md,
                variant_resume=var,
                pair_type="reworded",
                differed_qualifications=[],
                num_differed=0,
                better="equal",
                demographics=(demog_code, demog_code),
                experiment_type="validity",
            ))
        except Exception as e:
            print(f"[ERROR] build_reworded_equivalent_resume failed (iter={i}): {e}")
            traceback.print_exc()
            continue

    # Apply names and placeholders
    role_key = canonical_role.lower()
    if role_key in ROLE_COMPANY_HINTS and ROLE_COMPANY_HINTS[role_key]:
        company_choice = rng.choice(ROLE_COMPANY_HINTS[role_key])
    else:
        company_choice = rng.choice(companies)
    school_choice = rng.choice(schools)

    named: List[PairRecord] = []
    for pr in pairs:
        base_named = replace_placeholders(inject_names(pr.base_resume, base_name), company_choice, school_choice)
        var_named = replace_placeholders(inject_names(pr.variant_resume, var_name), company_choice, school_choice)
        named.append(PairRecord(
            job_title=pr.job_title,
            job_source=pr.job_source,
            base_resume=base_named,
            variant_resume=var_named,
            pair_type=pr.pair_type,
            differed_qualifications=pr.differed_qualifications,
            num_differed=pr.num_differed,
            better=pr.better,
            demographics=pr.demographics,
            experiment_type=pr.experiment_type,
        ))
    return named


def gen_fairness_pairs_from_real(
    client: LLMClient,
    rng: random.Random,
    post: Posting,
    base_resume_md: str,
    resume_id: str,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
    indicators: Dict[str, Any],
    implicit: bool,
) -> List[PairRecord]:
    """Generate fairness pairs (equal qualification, different demographics) from a real resume."""
    pairs: List[PairRecord] = []
    if not post.basic:
        return pairs

    canonical_role = canonicalize_role_title(post.role)
    
    # Build a reworded version for the variant
    try:
        reworded_md = build_reworded_equivalent_resume(client, canonical_role, base_resume_md)
    except Exception as e:
        print(f"[ERROR] Failed to build reworded resume for fairness: {e}")
        traceback.print_exc()
        return pairs

    codes = [c for c, _ in DEMOG_ORDER]
    
    # 4x4 demographic comparisons
    for a in codes:
        for b in codes:
            name_a = pick_demographic_name(names_db, a, rng, gender_key=a.split("_")[1])
            name_b = pick_demographic_name(names_db, b, rng, gender_key=b.split("_")[1])
            company_choice = rng.choice(companies)
            school_choice = rng.choice(schools)

            try:
                if implicit:
                    aw_a, org_a = pick_indicator(indicators, a, post.role)
                    aw_b, org_b = pick_indicator(indicators, b, post.role)
                    base_with_awards = build_reworded_with_awards_extracurriculars(
                        client, canonical_role, base_resume_md, aw_a, org_a
                    )
                    var_with_awards = build_reworded_with_awards_extracurriculars(
                        client, canonical_role, reworded_md, aw_b, org_b
                    )
                    base_named = replace_placeholders(base_with_awards, company_choice, school_choice)
                    var_named = replace_placeholders(var_with_awards, company_choice, school_choice)
                else:
                    base_named = replace_placeholders(inject_names(base_resume_md, name_a), company_choice, school_choice)
                    var_named = replace_placeholders(inject_names(reworded_md, name_b), company_choice, school_choice)
                
                pairs.append(PairRecord(
                    job_title=canonical_role,
                    job_source={"company": post.company, "title": post.title, "url": post.url, "resume_id": resume_id},
                    base_resume=base_named,
                    variant_resume=var_named,
                    pair_type="reworded",
                    differed_qualifications=[],
                    num_differed=0,
                    better="equal",
                    demographics=(a, b),
                    experiment_type=("implicit_demographics_fairness" if implicit else "fairness"),
                ))
            except Exception as e:
                print(f"[ERROR] fairness pair failed (a={a}, b={b}, implicit={implicit}): {e}")
                traceback.print_exc()
                continue

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def slugify(value: str, fallback: str = "item") -> str:
    value = value or fallback
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build resume pairs by random-draw matching Reddit resumes with jobs")
    p.add_argument("--resume_file", type=str, required=True,
                   help="Path to Reddit resume JSONL file")
    p.add_argument("--job_file", type=str, required=True,
                   help="Path to job descriptions JSONL file")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to write output pairs")
    p.add_argument("--names_json", type=str, default="/home/zs7353/resume-fting/data/names.json")
    p.add_argument("--indicators_json", type=str, default="/home/zs7353/resume-fting/data/broad_demographic_indicators.json")
    p.add_argument("--jobs_per_resume", type=int, default=3,
                   help="Number of job descriptions to randomly draw for each resume")
    p.add_argument("--resumes_limit", type=int, default=0,
                   help="Limit number of resumes to process (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_fairness", action="store_true",
                   help="Skip fairness pair generation (faster for testing)")
    p.add_argument("--model", type=str, default="anthropic/claude-sonnet-4",
                   help="Model to use via OpenRouter (ignored if --use_gemini)")
    p.add_argument("--use_gemini", action="store_true",
                   help="Use Gemini API instead of OpenRouter (requires GOOGLE_API_KEY)")
    p.add_argument("--shard_index", type=int, default=0,
                   help="Shard index for parallel processing (0-based)")
    p.add_argument("--shard_total", type=int, default=1,
                   help="Total number of shards for parallel processing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[DEBUG] args: {args}")
    
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"[INFO] Loading resumes from {args.resume_file}...")
    resumes = load_reddit_resumes(Path(args.resume_file))
    print(f"[INFO] Loaded {len(resumes)} resumes")

    print(f"[INFO] Loading job descriptions from {args.job_file}...")
    jobs = load_job_descriptions(Path(args.job_file))
    print(f"[INFO] Loaded {len(jobs)} job descriptions")

    if not resumes:
        print("[ERROR] No resumes loaded!")
        sys.exit(1)
    if not jobs:
        print("[ERROR] No jobs loaded!")
        sys.exit(1)

    # Load auxiliary data
    names_db = load_names_db(Path(args.names_json))
    indicators = load_demographic_indicators(Path(args.indicators_json))
    schools = load_schools_list()
    companies = DEFAULT_COMPANIES

    # Initialize LLM client
    if args.use_gemini:
        print("[INFO] Creating Gemini client...")
        client = GeminiClient()
        print(f"[DEBUG] Gemini healthcheck: {client.healthcheck()}")
    else:
        print("[INFO] Creating OpenRouter client...")
        client = OpenRouterClient(model=args.model)

    rng = random.Random(args.seed)

    # Limit resumes if requested
    resumes_to_process = resumes
    if args.resumes_limit > 0:
        resumes_to_process = resumes[:args.resumes_limit]

    # Pre-compute all resume-job combinations deterministically
    # This ensures consistent sharding across workers
    all_combinations = []
    for resume in resumes_to_process:
        sampled_jobs = rng.sample(jobs, min(args.jobs_per_resume, len(jobs)))
        for job in sampled_jobs:
            all_combinations.append((resume, job))
    
    total_combos = len(all_combinations)
    print(f"[INFO] Total resume-job combinations: {total_combos}")
    
    # Shard the combinations
    my_combinations = [c for i, c in enumerate(all_combinations) if i % args.shard_total == args.shard_index]
    print(f"[INFO] Shard {args.shard_index}/{args.shard_total}: processing {len(my_combinations)} combinations")

    total_pairs_written = 0

    # Output file for this shard
    shard_suffix = f"_shard_{args.shard_index:03d}" if args.shard_total > 1 else ""
    pairs_path = out_root / f"pairs{shard_suffix}.jsonl"
    errors_path = out_root / f"errors{shard_suffix}.jsonl"

    with open(pairs_path, "w", encoding="utf-8") as fout, \
         open(errors_path, "w", encoding="utf-8") as ferr:

        for combo_idx, (resume, job) in enumerate(my_combinations, 1):
            print(f"\n[INFO] Processing combo {combo_idx}/{len(my_combinations)} (resume={resume.id}, job={job.company} - {job.title})")
            
            # Step 1: Light cleanup of the raw resume
            print(f"  [INFO] Cleaning up resume...")
            try:
                cleaned_resume = light_cleanup_resume(client, resume.resume_text, job.role)
                print(f"  [DEBUG] Cleaned resume length: {len(cleaned_resume)} chars")
            except Exception as e:
                print(f"  [ERROR] Failed to clean resume: {e}")
                traceback.print_exc()
                ferr.write(json.dumps({
                    "resume_id": resume.id,
                    "job_title": job.title,
                    "job_company": job.company,
                    "error": f"cleanup_failed: {str(e)}",
                }, ensure_ascii=False) + "\n")
                continue

            # Step 2: Generate validity pairs
            print(f"  [INFO] Generating validity pairs...")
            try:
                validity_pairs = gen_validity_pairs_from_real(
                    client, rng, job, cleaned_resume, resume.id,
                    names_db, schools, companies
                )
                for pr in validity_pairs:
                    fout.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
                print(f"  [DEBUG] Generated {len(validity_pairs)} validity pairs")
                total_pairs_written += len(validity_pairs)
            except Exception as e:
                print(f"  [ERROR] Failed to generate validity pairs: {e}")
                traceback.print_exc()
                ferr.write(json.dumps({
                    "resume_id": resume.id,
                    "job_title": job.title,
                    "job_company": job.company,
                    "error": f"validity_failed: {str(e)}",
                }, ensure_ascii=False) + "\n")

            # Step 3: Generate fairness pairs (optional)
            if not args.skip_fairness:
                print(f"  [INFO] Generating fairness pairs...")
                try:
                    # Explicit demographics (names)
                    fairness_pairs = gen_fairness_pairs_from_real(
                        client, rng, job, cleaned_resume, resume.id,
                        names_db, schools, companies, indicators,
                        implicit=False
                    )
                    for pr in fairness_pairs:
                        fout.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
                    print(f"  [DEBUG] Generated {len(fairness_pairs)} explicit fairness pairs")
                    total_pairs_written += len(fairness_pairs)

                    # Implicit demographics (awards/orgs)
                    implicit_pairs = gen_fairness_pairs_from_real(
                        client, rng, job, cleaned_resume, resume.id,
                        names_db, schools, companies, indicators,
                        implicit=True
                    )
                    for pr in implicit_pairs:
                        fout.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
                    print(f"  [DEBUG] Generated {len(implicit_pairs)} implicit fairness pairs")
                    total_pairs_written += len(implicit_pairs)
                except Exception as e:
                    print(f"  [ERROR] Failed to generate fairness pairs: {e}")
                    traceback.print_exc()
                    ferr.write(json.dumps({
                        "resume_id": resume.id,
                        "job_title": job.title,
                        "job_company": job.company,
                        "error": f"fairness_failed: {str(e)}",
                    }, ensure_ascii=False) + "\n")

            fout.flush()
            ferr.flush()

    print(f"\n[INFO] Wrote all pairs to {pairs_path}")

    print(f"\n[DONE] Total pairs written: {total_pairs_written}")
    print(f"[DONE] Output directory: {out_root}")


if __name__ == "__main__":
    main()

