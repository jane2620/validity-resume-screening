#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resume_validity.llm.openrouter_client import OpenRouterClient
from resume_validity.llm.resume_builder_claude import (
    build_basic_resume,
    build_underqualified_resume_multi,
    build_preferred_resume_multi,
    build_reworded_equivalent_resume,
    build_reworded_with_awards_extracurriculars,
)


# --- Utility data classes ---


@dataclass
class Posting:
    role: str
    title_norm: str
    original_role: str
    source: str
    company: str
    title: str
    url: str
    basic: List[str]
    bonus: List[str]


@dataclass
class PairRecord:
    job_title: str  # now set to canonical role title (not raw posting title)
    job_source: Dict[str, Any]
    base_resume: str
    variant_resume: str
    pair_type: str  # underqualified | preferred | reworded
    differed_qualifications: List[str]
    num_differed: int
    better: str  # first | second | equal
    demographics: Tuple[str, str]  # (demog_base, demog_variant)
    experiment_type: str  # validity | fairness | implicit_demographics_fairness


DEMOG_ORDER = [
    ("W_M", "white man"),
    ("W_W", "white woman"),
    ("B_M", "black man"),
    ("B_W", "black woman"),
]


def load_harvest_dir(harvest_dir: Path) -> Dict[str, List[Posting]]:
    buckets: Dict[str, List[Posting]] = {}
    for role_dir in sorted(harvest_dir.iterdir()):
        if not role_dir.is_dir():
            continue
        name = role_dir.name
        jsonl_files = [p for p in role_dir.glob("passing_*.jsonl")]
        if not jsonl_files:
            continue
        path = jsonl_files[0]
        rows: List[Posting] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rows.append(Posting(
                    role=str(obj.get("role") or obj.get("title_norm") or name),
                    title_norm=str(obj.get("title_norm") or name),
                    original_role=str(obj.get("original_role") or ""),
                    source=str(obj.get("source") or ""),
                    company=str(obj.get("company") or ""),
                    title=str(obj.get("title") or ""),
                    url=str(obj.get("url") or ""),
                    basic=list(obj.get("basic") or []),
                    bonus=list(obj.get("bonus") or []),
                ))
        if rows:
            buckets[name] = rows
    return buckets


def pick_top_roles_by_count(buckets: Dict[str, List[Posting]], top_k: int = 5) -> List[str]:
    ranked = sorted(((k, len(v)) for k, v in buckets.items()), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:top_k]]


def _stable_int_from_str(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def sample_postings_for_role(role: str, postings: List[Posting], top_roles: List[str]) -> List[Posting]:
    n = len(postings)
    if role in top_roles:
        k = min(20, n)
    else:
        k = min(5, n)
    # deterministic and stable shuffle per role name for sharding stability
    rng = random.Random(_stable_int_from_str(role) & 0xFFFFFFFF)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return [postings[i] for i in idxs[:k]]


def canonicalize_role_title(role_value: str) -> str:
    """Map lower-case role labels to canonical job titles used in indicators mapping."""
    role_l = (role_value or "").strip().lower()
    mapping = {
        "android engineer": "Android Engineer",
        "ios engineer": "iOS Engineer",
        "software engineer": "Software Engineer",
        "software engineer - backend": "Software Engineer - Backend",
        "software engineer, product": "Software Engineer, Product",
        "data engineer": "Data Engineer",
        "data scientist": "Data Scientist",
        "product designer": "Product Designer",
        "product manager": "Product Manager",
        "solutions architect": "Solutions Architect",
        "solutions engineer": "Solutions Engineer",
        "enterprise sales engineer": "Enterprise Sales Engineer",
        "enterprise security engineer": "Enterprise Security Engineer",
        "customer solutions engineer": "Customer Solutions Engineer",
        "customer success manager": "Customer Success Manager",
        "business development representative": "Business Development Representative",
        "business development representative - german speaking": "Business Development Representative - German Speaking",
        "sales development representative": "Sales Development Representative",
        "enterprise account executive": "Enterprise Account Executive",
        "commercial account executive": "Commercial Account Executive",
        "manager, sales development": "Manager, Sales Development",
        "manager, field sales": "Manager, Field Sales",
        "director, enterprise sales": "Director, Enterprise Sales",
        "revenue operations manager": "Revenue Operations Manager",
        "field sales representative": "Field Sales Representative",
    }
    return mapping.get(role_l, role_value.strip().title())


def choose_k(seq: List[str], k: int, rng: random.Random) -> List[str]:
    k = max(0, min(k, len(seq)))
    return rng.sample(seq, k) if k > 0 else []


def inject_names(md: str, name: str) -> str:
    return md.replace("{{CANDIDATE_NAME}}", name)


def replace_placeholders(md: str, company: str, school: str) -> str:
    out = md.replace("{{COMPANY_NAME}}", company)
    out = out.replace("{{SCHOOL_NAME}}", school)
    return out


def load_names_db(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_demographic_name(names_db: Dict[str, Any], code: str, rng: random.Random, gender_key: str) -> str:
    # code in {W_M, W_W, B_M, B_W}; map to structure in names.json
    gender_map = {"M": "MEN", "W": "WOMEN"}
    eth = code.split("_")[0]
    gen = code.split("_")[1]
    gender_bucket = gender_map[gen]
    eth_bucket = {"W": "W", "B": "B"}.get(eth, "W")
    pool = names_db[gender_bucket][eth_bucket]
    return rng.choice(pool).title()


def load_schools_list() -> List[str]:
    # Reasonable real schools; extendable
    return [
        "Princeton University", "Massachusetts Institute of Technology", "Stanford University",
        "Harvard University", "University of California, Berkeley", "Carnegie Mellon University",
        "University of Illinois Urbana-Champaign", "Georgia Institute of Technology",
        "University of Michigan", "Cornell University", "University of Washington",
        "Columbia University", "University of Texas at Austin", "University of Wisconsin–Madison",
        "Purdue University", "University of Pennsylvania", "Brown University",
        "University of Toronto", "ETH Zurich", "University of Cambridge",
    ]


DEFAULT_COMPANIES = [
    # Engineering / CS
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Nvidia", "OpenAI", "Anthropic", "Databricks",
    "Snowflake", "Stripe", "Roblox", "Square", "Block", "Cloudflare", "Salesforce", "Adobe",
    # Product / Design
    "Airbnb", "Figma", "Notion",
    # Consumer / Media
    "Netflix", "Dropbox",
    # Ride / Food / Logistics
    "Uber", "Lyft", "DoorDash",
]


ROLE_COMPANY_HINTS: Dict[str, List[str]] = {
    # Sales-related roles with plausible companies
    "commercial account executive": ["Salesforce", "HubSpot", "Oracle", "SAP", "Workday"],
    "enterprise account executive": ["Salesforce", "ServiceNow", "Datadog", "Cloudflare", "Snowflake"],
    "field sales representative": ["Procter & Gamble", "Unilever", "PepsiCo", "Coca-Cola", "Kroger"],
    "sales development representative": ["Salesforce", "ZoomInfo", "Gong", "HubSpot", "Outreach"],
    "business development representative": ["Salesforce", "Snowflake", "Databricks", "ServiceNow"],
    # Engineering / CS
    "software engineer": ["Google", "Microsoft", "Meta", "Apple", "Amazon", "Stripe", "Databricks"],
    "software engineer - backend": ["Google", "Microsoft", "Meta", "Stripe", "Snowflake"],
    "software engineer, product": ["Airbnb", "Notion", "Figma", "Dropbox"],
    "android engineer": ["Google", "Meta", "Roblox", "Square"],
    "ios engineer": ["Apple", "Airbnb", "Lyft"],
    "data engineer": ["Databricks", "Snowflake", "Google"],
    "data scientist": ["Google", "Meta", "Airbnb"],
    # Design / PM
    "product designer": ["Figma", "Airbnb", "Notion"],
    "product manager": ["Google", "Amazon", "Meta", "Stripe"],
    # Solutions/Customer
    "solutions architect": ["AWS", "Google", "Microsoft", "Datadog"],
    "solutions engineer": ["Google", "Cloudflare", "Datadog"],
    "customer solutions engineer": ["Google", "Datadog"],
    "customer success manager": ["Salesforce", "HubSpot"],
    "enterprise sales engineer": ["Salesforce", "ServiceNow", "Datadog", "Cloudflare", "Snowflake"],
    # Security
    "enterprise security engineer": ["Cloudflare", "CrowdStrike", "Okta"],
    # Revenue Ops
    "revenue operations manager": ["Salesforce", "HubSpot", "Gainsight"],
    # Regional/Language-specific BDR
    "business development representative - german speaking": ["Salesforce", "SAP", "HubSpot", "Oracle"],
    # Management
    "manager, sales development": ["Salesforce", "HubSpot", "Gong", "Outreach"],
    "manager, field sales": ["Procter & Gamble", "Coca-Cola", "PepsiCo", "Unilever"],
    "director, enterprise sales": ["Salesforce", "ServiceNow", "Oracle", "SAP"],
}


def gen_validity_pairs(
    client: OpenRouterClient,
    rng: random.Random,
    post: Posting,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    if not post.basic:
        return pairs
    # Base resume with only basic qualifications
    canonical_role = canonicalize_role_title(post.role)
    base = build_basic_resume(client, canonical_role, post.basic)

    # For each k in 1..min(3, |basic| or |bonus|)
    max_k_basic = min(3, len(post.basic))
    max_k_bonus = min(3, len(post.bonus))

    # Names and demographics: for validity, both from same demographic group
    demogs = [code for code, _ in DEMOG_ORDER]
    demog_code = rng.choice(demogs)
    base_name = pick_demographic_name(names_db, demog_code, rng, gender_key=demog_code.split("_")[1])
    var_name = pick_demographic_name(names_db, demog_code, rng, gender_key=demog_code.split("_")[1])

    # Underqualified (remove up to k basics)
    for k in range(1, max_k_basic + 1):
        removed = choose_k(post.basic, k, rng)
        var = build_underqualified_resume_multi(client, canonical_role, base, removed)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base,
            variant_resume=var,
            pair_type="underqualified",
            differed_qualifications=removed,
            num_differed=len(removed),
            better="first",
            demographics=(demog_code, demog_code),
            experiment_type="validity",
        ))

    # Preferred (add up to k bonus)
    for k in range(1, max_k_bonus + 1):
        added = choose_k(post.bonus, k, rng)
        if not added:
            continue
        var = build_preferred_resume_multi(client, canonical_role, base, added)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base,
            variant_resume=var,
            pair_type="preferred",
            differed_qualifications=added,
            num_differed=len(added),
            better="second",
            demographics=(demog_code, demog_code),
            experiment_type="validity",
        ))

    # Reworded equal (2 times)
    for _ in range(2):
        var = build_reworded_equivalent_resume(client, canonical_role, base)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base,
            variant_resume=var,
            pair_type="reworded",
            differed_qualifications=[],
            num_differed=0,
            better="equal",
            demographics=(demog_code, demog_code),
            experiment_type="validity",
        ))

    # Replace placeholders (names, companies, schools) for all pairs
    # Role-consistent company selection
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


def gen_validity_pairs_demographics(
    client: OpenRouterClient,
    rng: random.Random,
    post: Posting,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
) -> List[PairRecord]:
    """Same as validity pairs but draw names independently for base/variant from any demographic group."""
    pairs: List[PairRecord] = []
    if not post.basic:
        return pairs
    canonical_role = canonicalize_role_title(post.role)
    base = build_basic_resume(client, canonical_role, post.basic)
    max_k_basic = min(3, len(post.basic))
    max_k_bonus = min(3, len(post.bonus))
    # Random independent demographics
    codes = [c for c,_ in DEMOG_ORDER]
    base_code = rng.choice(codes)
    var_code  = rng.choice(codes)
    base_name = pick_demographic_name(names_db, base_code, rng, gender_key=base_code.split("_")[1])
    var_name  = pick_demographic_name(names_db, var_code,  rng, gender_key=var_code.split("_")[1])
    # Build variants
    for k in range(1, max_k_basic + 1):
        removed = choose_k(post.basic, k, rng)
        var = build_underqualified_resume_multi(client, canonical_role, base, removed)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base,
            variant_resume=var,
            pair_type="underqualified",
            differed_qualifications=removed,
            num_differed=len(removed),
            better="first",
            demographics=(base_code, var_code),
            experiment_type="validity_demographics",
        ))
    for k in range(1, max_k_bonus + 1):
        added = choose_k(post.bonus, k, rng)
        if not added:
            continue
        var = build_preferred_resume_multi(client, canonical_role, base, added)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base,
            variant_resume=var,
            pair_type="preferred",
            differed_qualifications=added,
            num_differed=len(added),
            better="second",
            demographics=(base_code, var_code),
            experiment_type="validity_demographics",
        ))
    for _ in range(2):
        var = build_reworded_equivalent_resume(client, canonical_role, base)
        pairs.append(PairRecord(
            job_title=canonical_role,
            job_source={"company": post.company, "title": post.title, "url": post.url},
            base_resume=base,
            variant_resume=var,
            pair_type="reworded",
            differed_qualifications=[],
            num_differed=0,
            better="equal",
            demographics=(base_code, var_code),
            experiment_type="validity_demographics",
        ))
    # Company/school placeholders
    role_key = canonical_role.lower()
    if role_key in ROLE_COMPANY_HINTS and ROLE_COMPANY_HINTS[role_key]:
        company_choice = rng.choice(ROLE_COMPANY_HINTS[role_key])
    else:
        company_choice = rng.choice(companies)
    school_choice = rng.choice(schools)
    named: List[PairRecord] = []
    for pr in pairs:
        base_named = replace_placeholders(inject_names(pr.base_resume, base_name), company_choice, school_choice)
        var_named  = replace_placeholders(inject_names(pr.variant_resume, var_name), company_choice, school_choice)
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


def load_demographic_indicators(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_indicator(indicators: Dict[str, Any], demog_code: str, job_role: str) -> Tuple[str, str]:
    """Pick award/org strings using robust role-to-field mapping.
    We map canonicalized role to the indicators' job_to_field_mapping; if missing, fall back by role family (Engineer->CS, Data->Data Science, Designer->Design, else Business)."""
    node = indicators["demographic_specific"][demog_code]
    canonical_role = canonicalize_role_title(job_role)
    # Try direct mapping first
    field = indicators["job_to_field_mapping"].get(canonical_role, "")
    if not field:
        rl = canonical_role.lower()
        if any(k in rl for k in ["engineer", "architect"]):
            field = "Computer Science"
        elif "data" in rl:
            field = "Data Science"
        elif "designer" in rl or "design" in rl:
            field = "Design"
        elif any(k in rl for k in ["manager", "sales", "account executive", "representative"]):
            field = "Business"
        else:
            field = "Business"
    award = random.choice(node["relevant_awards"]).replace("{Job Title}", canonical_role)
    org = random.choice(node["organizations"]).replace("{Field}", field)
    return award, org


def gen_fairness_pairs_equal(
    client: OpenRouterClient,
    rng: random.Random,
    post: Posting,
    names_db: Dict[str, Any],
    schools: List[str],
    companies: List[str],
    indicators: Dict[str, Any],
    implicit: bool,
) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    if not post.basic:
        return pairs
    canonical_role = canonicalize_role_title(post.role)
    base = build_basic_resume(client, canonical_role, post.basic)
    reworded = build_reworded_equivalent_resume(client, canonical_role, base)

    # For fairness: pairwise demographic comparisons across 4 groups (ordered)
    codes = [c for c, _ in DEMOG_ORDER]
    # 4x4 comparisons
    for a in codes:
        for b in codes:
            name_a = pick_demographic_name(names_db, a, rng, gender_key=a.split("_")[1])
            name_b = pick_demographic_name(names_db, b, rng, gender_key=b.split("_")[1])
            # Role-consistent company selection per pair for diversity
            role_key = canonical_role.lower()
            if role_key in ROLE_COMPANY_HINTS and ROLE_COMPANY_HINTS[role_key]:
                company_choice = rng.choice(ROLE_COMPANY_HINTS[role_key])
            else:
                company_choice = rng.choice(companies)
            school_choice = rng.choice(schools)

            # Build per-pair with retries (>=3) and continue-on-error
            def try_build_pair() -> Optional[Tuple[str, str]]:
                for attempt in range(1, 5):  # 4 attempts total
                    try:
                        if implicit:
                            aw_a, org_a = pick_indicator(indicators, a, post.role)
                            aw_b, org_b = pick_indicator(indicators, b, post.role)
                            base_md = build_reworded_with_awards_extracurriculars(client, canonical_role, base, aw_a, org_a)
                            var_md  = build_reworded_with_awards_extracurriculars(client, canonical_role, reworded, aw_b, org_b)
                            base_named = replace_placeholders(base_md, company_choice, school_choice)
                            var_named  = replace_placeholders(var_md, company_choice, school_choice)
                        else:
                            base_named = replace_placeholders(inject_names(base, name_a), company_choice, school_choice)
                            var_named  = replace_placeholders(inject_names(reworded, name_b), company_choice, school_choice)
                        return base_named, var_named
                    except Exception as e:
                        print(f"[fairness retry] role={canonical_role} a={a} b={b} attempt={attempt} error={e}")
                        continue
                return None

            built = try_build_pair()
            if not built:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build resume pairs from harvested postings (Claude via OpenRouter)")
    p.add_argument("--harvest_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--names_json", type=str, default="/home/zs7353/resume-fting/data/names.json")
    p.add_argument("--indicators_json", type=str, default="/home/zs7353/resume-fting/data/broad_demographic_indicators.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--shard_total", type=int, default=1)
    p.add_argument("--toy_limit", type=int, default=0, help="If >0, limit number of postings processed for a quick sanity run")
    p.add_argument("--model", type=str, default="anthropic/claude-sonnet-4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed + args.shard_index)

    harvest_dir = Path(args.harvest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names_db = load_names_db(Path(args.names_json))
    indicators = load_demographic_indicators(Path(args.indicators_json))
    schools = load_schools_list()
    companies = DEFAULT_COMPANIES

    data = load_harvest_dir(harvest_dir)
    if not data:
        print(f"No postings found under {harvest_dir}")
        sys.exit(1)

    # Top 5 roles by volume
    top5 = pick_top_roles_by_count(data, top_k=5)

    # Create OpenRouter client for Claude (env var must be set)
    client = OpenRouterClient(model=args.model)

    all_pairs_path = out_dir / f"pairs_shard_{args.shard_index:02d}_of_{args.shard_total:02d}.jsonl"
    written = 0
    error_log_path = out_dir / f"errors_shard_{args.shard_index:02d}_of_{args.shard_total:02d}.jsonl"
    with open(all_pairs_path, "w", encoding="utf-8") as fout, open(error_log_path, "w", encoding="utf-8") as ferr:
        processed = 0
        print(f"[shard {args.shard_index}/{args.shard_total}] Starting. Output -> {all_pairs_path}")
        for role, postings in sorted(data.items()):
            # Shard by posting to allow more shards than roles

            sampled = sample_postings_for_role(role, postings, top5)
            # Compute fairness set from the first fairness_k sampled postings (by URL), independent of shard
            fairness_k = 10 if role in top5 else len(sampled)
            fairness_urls = {p.url for p in sampled[:fairness_k]}
            print(f"[shard {args.shard_index}] Role '{role}' -> sampled={len(sampled)} fairness_k={fairness_k}")

            for idx, post in enumerate(sampled):
                if args.shard_total > 1:
                    post_bucket = _stable_int_from_str(f"{role}|{post.url}") % args.shard_total
                    if post_bucket != args.shard_index:
                        continue
                print(f"[shard {args.shard_index}] ({idx+1}/{len(sampled)}) {post.company} | {post.title} -> start")
                # Validity set
                try:
                    validity_pairs = gen_validity_pairs(client, rng, post, names_db, schools, companies)
                except Exception as e:
                    print(f"[ERROR] gen_validity_pairs failed for role={role} url={post.url}: {e}")
                    import traceback as _tb
                    print(_tb.format_exc())
                    validity_pairs = []
                for pr in validity_pairs:
                    fout.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
                    written += 1
                fout.flush()
                print(f"[shard {args.shard_index}] ({idx+1}/{len(sampled)}) validity -> wrote {len(validity_pairs)} pairs (total={written})")

                # Validity_demographics set
                try:
                    validity_demo_pairs = gen_validity_pairs_demographics(client, rng, post, names_db, schools, companies)
                except Exception as e:
                    print(f"[ERROR] gen_validity_pairs_demographics failed for role={role} url={post.url}: {e}")
                    import traceback as _tb
                    print(_tb.format_exc())
                    validity_demo_pairs = []
                for pr in validity_demo_pairs:
                    fout.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
                    written += 1
                fout.flush()
                print(f"[shard {args.shard_index}] ({idx+1}/{len(sampled)}) validity_demographics -> wrote {len(validity_demo_pairs)} pairs (total={written})")

                # Fairness equal pairs: generate if this posting is in the fairness set
                if post.url in fairness_urls:
                    fairness_pairs_name = gen_fairness_pairs_equal(
                        client, rng, post, names_db, schools, companies, indicators, implicit=False
                    )
                    fairness_pairs_implicit = gen_fairness_pairs_equal(
                        client, rng, post, names_db, schools, companies, indicators, implicit=True
                    )
                    for pr in fairness_pairs_name + fairness_pairs_implicit:
                        if pr.base_resume and pr.variant_resume:
                            fout.write(json.dumps(asdict(pr), ensure_ascii=False) + "\n")
                            written += 1
                        else:
                            # log failed pair attempt
                            ferr.write(json.dumps({
                                "job_title": pr.job_title,
                                "job_source": pr.job_source,
                                "demographics": pr.demographics,
                                "experiment_type": pr.experiment_type,
                                "error": "failed_to_build_pair_after_retries"
                            }, ensure_ascii=False) + "\n")
                    ferr.flush()
                    fout.flush()
                    print(f"[shard {args.shard_index}] ({idx+1}/{len(sampled)}) fairness+implicit -> wrote {len(fairness_pairs_name)+len(fairness_pairs_implicit)} pairs (total={written})")

                processed += 1
                if args.toy_limit and processed >= args.toy_limit:
                    print(f"Toy limit reached at {processed} postings")
                    print(f"Wrote {written} pairs -> {all_pairs_path}")
                    return
        print(f"[shard {args.shard_index}] DONE -> wrote {written} pairs -> {all_pairs_path}")

    print(f"Wrote {written} pairs -> {all_pairs_path}")


if __name__ == "__main__":
    main()


