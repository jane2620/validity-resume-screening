#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

from resume_validity.llm.openrouter_client import OpenRouterClient
import time


# Reuse prompt content from resume_builder, but call via OpenRouter GPT-5
BASIC_RESUME_SYSTEM = (
    "You write realistic professional resumes in Markdown. Use concise, credible content with clean, readable formatting."
)


def build_basic_resume(client: OpenRouterClient, role_title: str, basic_qualifications: List[str]) -> str:
    placeholder = "{{CANDIDATE_NAME}}"
    user = (
        f"Construct a realistic-looking resume in Markdown for the role: {role_title}.\n"
        f"Include ALL of these required qualifications and do not include ANY other qualifications beyond reasonable elaborations (no extras not implied by the list).\n"
        + "\n".join([f"- {q}" for q in basic_qualifications]) + "\n\n"
        "Rules:\n"
        f"- Begin with 'Name: {placeholder}'.\n"
        "- Replace company names with the placeholder {{COMPANY_NAME}}. Replace school names with the placeholder {{SCHOOL_NAME}}. Do not invent specific company or school names.\n"
        "- Do NOT include any contact information or placeholder contact lines (no email, phone, LinkedIn, GitHub, websites).\n"
        "- Keep within one page, concise and non-verbose. Prefer 1–2 roles in Experience; 2–3 bullets per role.\n"
        "- Do not add extra qualifications beyond the required list.\n"
        "- Use beautiful, clean Markdown formatting: clear section headers, subtle separators, consistent bullets.\n"
        "- Sections: Summary, Experience, Education, Skills, Certifications (if implied).\n"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.4, max_tokens=2048)


def build_underqualified_multi(client: OpenRouterClient, role_title: str, base_md: str, removed: List[str]) -> str:
    user = (
        f"Given the basic resume below, create an UNDER-QUALIFIED variant by REMOVING EXACTLY these {len(removed)} qualifications.\n"
        + "\n".join([f"- {q}" for q in removed]) + "\n"
        "Do not remove anything else and do not add new qualifications.\n"
        "When the removed qualification is about years of experience, ensure that every other part of the resume remains generally unchanged except the years of experience.\n"
        "Keep 'Name: {{CANDIDATE_NAME}}' and the {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful Markdown formatting. Keep overall length roughly equal to the base (±10%), maintaining the same number of roles and similar bullet counts.\n\n"
        f"Basic resume:\n{base_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.3, max_tokens=2048)


def build_preferred_multi(client: OpenRouterClient, role_title: str, base_md: str, added: List[str]) -> str:
    user = (
        f"Given the basic resume below, create a PREFERRED variant by ADDING EXACTLY these {len(added)} preferred qualifications.\n"
        + "\n".join([f"- {q}" for q in added]) + "\n"
        "Do not add any other new qualifications.\n"
        "When the added qualification is about years of experience, ensure that every other part of the resume remains generally unchanged except the years of experience.\n"
        "Keep 'Name: {{CANDIDATE_NAME}}' and the {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful Markdown formatting. Keep overall length roughly equal to the base (±10%), maintaining the same number of roles and similar bullet counts.\n\n"
        f"Basic resume:\n{base_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.35, max_tokens=2048)


def build_reworded_equivalent(client: OpenRouterClient, role_title: str, base_md: str) -> str:
    user = (
        "Reword and restructure the following resume while keeping qualifications equivalent.\n"
        "Do not add or remove qualifications beyond rephrasing and reordering. Keep 'Name: {{CANDIDATE_NAME}}' at top and preserve {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful formatting. Keep overall length roughly equal to the base (±10%), with the same number of roles and similar bullet counts.\n\n"
        f"Resume:\n{base_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.5, max_tokens=2048)


def _retry_call(fn, max_attempts: int = 4, initial_wait: float = 1.0, backoff: float = 1.8, desc: str = ""):
    """Call fn() with retries and simple exponential backoff; print errors for visibility.

    Never hides errors — on repeated failure, re-raises the last exception.
    """
    attempt = 0
    wait = initial_wait
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            print(f"[ERROR] LLM call failed (attempt {attempt}/{max_attempts}) {desc}: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            if attempt >= max_attempts:
                raise
            time.sleep(wait)
            wait *= backoff


def _call_text_with_validation(call_fn, desc: str, min_chars: int = 100, attempts: int = 4) -> str:
    """Call an LLM text function with retries on exceptions and empty/too-short outputs.

    - Retries on thrown exceptions (via _retry_call)
    - Retries again if the returned text is empty/too short
    """
    last_txt: str = ""
    for i in range(1, attempts + 1):
        try:
            txt = _retry_call(call_fn, max_attempts=1, desc=f"{desc} try={i}")
        except Exception as e:
            print(f"[ERROR] {desc} raised: {e}")
            traceback.print_exc()
            if i >= attempts:
                raise
            time.sleep(0.5 * i)
            continue
        last_txt = txt if isinstance(txt, str) else ""
        if isinstance(last_txt, str) and len(last_txt.strip()) >= min_chars:
            return last_txt
        print(f"[WARN] {desc} returned empty/too-short text (len={len(last_txt.strip()) if last_txt else 0}); retry {i}/{attempts}")
        time.sleep(0.5 * i)
    return last_txt


@dataclass
class PairRecord:
    role: str
    job_title: str
    base_resume: str
    variant_resume: str
    pair_type: str
    differed_qualifications: List[str]
    num_differed: int
    better: str
    experiment_type: str
    demographics: List[str]
    job_description: str
    url: str


def load_harvest_dir(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    data: Dict[str, List[Dict[str, Any]]] = {}
    for role_dir in sorted(root.iterdir()):
        if not role_dir.is_dir():
            continue
        role = role_dir.name
        posts: List[Dict[str, Any]] = []
        for fp in sorted(role_dir.glob("passing_*.jsonl")):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    posts.append(row)
        if posts:
            data[role] = posts
    return data


def pick_samples_for_role(posts: List[Dict[str, Any]], k: int = 2, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return rng.sample(posts, min(k, len(posts)))


def choose_k_items(lst: List[str], k: int) -> List[str]:
    if not lst:
        return []
    k = max(0, min(k, len(lst)))
    idx = list(range(len(lst)))
    random.shuffle(idx)
    return [lst[i] for i in idx[:k]]


def sample_demographic_pair(names_db: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Sample a name and normalized demographic label from nested names.json.

    names.json structure:
      {
        "MEN": {"W": [...], "B": [...], "A": [...], "H": [...]},
        "WOMEN": {"W": [...], "B": [...], "A": [...], "H": [...]} 
      }
    We normalize labels to like "W_M", "B_W", "A_M", "H_W".
    """
    sex_key = random.choice([k for k in ("MEN", "WOMEN") if k in names_db])
    pools = names_db[sex_key]
    dem_key = random.choice([k for k in pools.keys() if isinstance(pools.get(k), list) and pools.get(k)])
    name = random.choice(pools[dem_key])
    label = f"{dem_key}_{'M' if sex_key == 'MEN' else 'W'}"
    return name, label, [label, label]


def sample_single_name(names_db: Dict[str, Any]) -> Tuple[str, str]:
    """Sample a single name and normalized demographic label.

    Returns (name, label) where label is like "W_M", "B_W".
    """
    sex_key = random.choice([k for k in ("MEN", "WOMEN") if k in names_db])
    pools = names_db[sex_key]
    dem_key = random.choice([k for k in pools.keys() if isinstance(pools.get(k), list) and pools.get(k)])
    name = random.choice(pools[dem_key])
    label = f"{dem_key}_{'M' if sex_key == 'MEN' else 'W'}"
    return name, label


def instantiate_pair(base_md: str, variant_md: str, name: str, companies: List[str], schools: List[str]) -> Tuple[str, str]:
    company = random.choice(companies)
    school = random.choice(schools)
    def _inst(txt: str) -> str:
        return (
            txt.replace("{{CANDIDATE_NAME}}", name)
               .replace("{{COMPANY_NAME}}", company)
               .replace("{{SCHOOL_NAME}}", school)
        )
    return _inst(base_md), _inst(variant_md)


def instantiate_pair_two_names(base_md: str, variant_md: str, name_base: str, name_variant: str, companies: List[str], schools: List[str]) -> Tuple[str, str]:
    """Instantiate placeholders using the SAME company/school across the pair,
    but different names for base and variant resumes (independent naming)."""
    company = random.choice(companies)
    school = random.choice(schools)
    def _inst(txt: str, nm: str) -> str:
        return (
            txt.replace("{{CANDIDATE_NAME}}", nm)
               .replace("{{COMPANY_NAME}}", company)
               .replace("{{SCHOOL_NAME}}", school)
        )
    return _inst(base_md, name_base), _inst(variant_md, name_variant)


def main():
    ap = argparse.ArgumentParser("Build GPT-5 validity_demographics pairs for roles with configurable sampling")
    ap.add_argument("--harvest_root", type=str, default="/home/zs7353/resume_validity/data/harvest_pool_t25")
    ap.add_argument("--names_path", type=str, default="/home/zs7353/resume-fting/data/names.json")
    ap.add_argument("--schools", type=str, nargs="*", default=[
        "MIT","Stanford","Princeton","Harvard","Berkeley","CMU","Caltech","UCLA","UT Austin","Georgia Tech"
    ])
    ap.add_argument("--companies", type=str, nargs="*", default=[
        "Google","OpenAI","Anthropic","Meta","Microsoft","Amazon","Apple","NVIDIA","Databricks","Salesforce","Uber","Stripe","Airbnb"
    ])
    ap.add_argument("--output", type=str, default="/home/zs7353/resume_validity/data/pairs_from_harvest_gpt5/validity_demographics.jsonl")
    ap.add_argument("--role", type=str, default="", help="If set, only build for this role directory name")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_posts", type=int, default=2, help="Number of postings to sample per role")
    ap.add_argument("--independent_names", action="store_true", help="If set, sample independent names for base and variant within a pair")
    args = ap.parse_args()

    random.seed(args.seed)

    # Load names
    with open(args.names_path, "r", encoding="utf-8") as f:
        names_db = json.load(f)

    # OpenRouter GPT-5 client
    client = OpenRouterClient(model="openai/gpt-5")

    roles_to_posts = load_harvest_dir(Path(args.harvest_root))
    if args.role:
        roles_to_posts = {k: v for k, v in roles_to_posts.items() if k == args.role}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for role, posts in roles_to_posts.items():
            print(f"[INFO] Processing role={role} with {len(posts)} postings; sampling {args.num_posts}...")
            samples = pick_samples_for_role(posts, k=args.num_posts, seed=args.seed)
            for idx, post in enumerate(samples):
                try:
                    role_title = post.get("role") or post.get("title") or role
                    basics = post.get("basic") or []
                    bonuses = post.get("bonus") or []
                    url = post.get("url", "")
                    jd = post.get("job_description", "")
                    if not basics:
                        print("[WARN] Skipping posting with empty basics")
                        continue
                    # Build placeholder basic resume first
                    print(f"[INFO] Building base + variants for role={role} sample_idx={idx} url={url}")
                    print(f"[STEP] base role={role} idx={idx} -> basic resume")
                    basic_md = _call_text_with_validation(lambda: build_basic_resume(client, role_title, basics), desc="basic", min_chars=200, attempts=4)
                    if not isinstance(basic_md, str) or not basic_md.strip():
                        print("[WARN] Empty basic resume; skipping posting")
                        continue
                    # Reworded variant from placeholder base
                    print(f"[STEP] base role={role} idx={idx} -> reworded")
                    re_md = _call_text_with_validation(lambda: build_reworded_equivalent(client, role_title, basic_md), desc="reworded", min_chars=200, attempts=4)
                    if not isinstance(re_md, str) or not re_md.strip():
                        print("[WARN] Empty reworded resume; skipping posting")
                        continue
                    # Instantiate placeholders for this pair
                    if args.independent_names:
                        name_b, lab_b = sample_single_name(names_db)
                        name_v, lab_v = sample_single_name(names_db)
                        base_inst, re_inst = instantiate_pair_two_names(basic_md, re_md, name_b, name_v, args.companies, args.schools)
                        demos = [lab_b, lab_v]
                    else:
                        name, group, demos = sample_demographic_pair(names_db)
                        base_inst, re_inst = instantiate_pair(basic_md, re_md, name, args.companies, args.schools)
                    re_rec = PairRecord(
                        role=role,
                        job_title=role_title,
                        base_resume=base_inst,
                        variant_resume=re_inst,
                        pair_type="reworded",
                        differed_qualifications=[],
                        num_differed=0,
                        better="equal",
                        experiment_type="validity_demographics",
                        demographics=demos,
                        job_description=jd,
                        url=url,
                    )
                    fout.write(json.dumps(asdict(re_rec)) + "\n")
                    fout.flush()
                    print("[OK] Wrote reworded pair")

                    # Underqualified and preferred for k=1,2,3 (cap by availability)
                    for k in [1,2,3]:
                        rem = choose_k_items(basics, k)
                        if rem:
                            print(f"[STEP] base role={role} idx={idx} -> underqualified k={len(rem)}")
                            uq_md = _call_text_with_validation(lambda: build_underqualified_multi(client, role_title, basic_md, rem), desc=f"underqualified k={len(rem)}", min_chars=200, attempts=4)
                            if isinstance(uq_md, str) and uq_md.strip():
                                if args.independent_names:
                                    name_b, lab_b = sample_single_name(names_db)
                                    name_v, lab_v = sample_single_name(names_db)
                                    base_inst, uq_inst = instantiate_pair_two_names(basic_md, uq_md, name_b, name_v, args.companies, args.schools)
                                    demos = [lab_b, lab_v]
                                else:
                                    name, group, demos = sample_demographic_pair(names_db)
                                    base_inst, uq_inst = instantiate_pair(basic_md, uq_md, name, args.companies, args.schools)
                                uq_rec = PairRecord(
                                    role=role,
                                    job_title=role_title,
                                    base_resume=base_inst,
                                    variant_resume=uq_inst,
                                    pair_type="underqualified",
                                    differed_qualifications=rem,
                                    num_differed=len(rem),
                                    better="first",
                                    experiment_type="validity_demographics",
                                    demographics=demos,
                                    job_description=jd,
                                    url=url,
                                )
                                fout.write(json.dumps(asdict(uq_rec)) + "\n")
                                fout.flush()
                                print(f"[OK] Wrote underqualified k={len(rem)}")
                            else:
                                print(f"[WARN] Empty underqualified k={k}; skipping")

                        add = choose_k_items(bonuses, k)
                        if add:
                            print(f"[STEP] base role={role} idx={idx} -> preferred k={len(add)}")
                            pf_md = _call_text_with_validation(lambda: build_preferred_multi(client, role_title, basic_md, add), desc=f"preferred k={len(add)}", min_chars=200, attempts=4)
                            if isinstance(pf_md, str) and pf_md.strip():
                                if args.independent_names:
                                    name_b, lab_b = sample_single_name(names_db)
                                    name_v, lab_v = sample_single_name(names_db)
                                    base_inst, pf_inst = instantiate_pair_two_names(basic_md, pf_md, name_b, name_v, args.companies, args.schools)
                                    demos = [lab_b, lab_v]
                                else:
                                    name, group, demos = sample_demographic_pair(names_db)
                                    base_inst, pf_inst = instantiate_pair(basic_md, pf_md, name, args.companies, args.schools)
                                pf_rec = PairRecord(
                                    role=role,
                                    job_title=role_title,
                                    base_resume=base_inst,
                                    variant_resume=pf_inst,
                                    pair_type="preferred",
                                    differed_qualifications=add,
                                    num_differed=len(add),
                                    better="second",
                                    experiment_type="validity_demographics",
                                    demographics=demos,
                                    job_description=jd,
                                    url=url,
                                )
                                fout.write(json.dumps(asdict(pf_rec)) + "\n")
                                fout.flush()
                                print(f"[OK] Wrote preferred k={len(add)}")
                            else:
                                print(f"[WARN] Empty preferred k={k}; skipping")
                except Exception:
                    print("[ERROR] Failure building pairs for one posting; continuing.")
                    print(traceback.format_exc())
                    continue

    print(f"Wrote GPT-5 validity_demographics pairs to {out_path}")


if __name__ == "__main__":
    main()


