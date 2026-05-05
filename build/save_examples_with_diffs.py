import json, re, random
from pathlib import Path
from typing import Optional, Dict, Any

from resume_validity.llm.gemini_client import GeminiClient
from resume_validity.llm.resume_builder import (
    build_basic_resume,
    build_preferred_resume,
    build_underqualified_resume,
    build_reworded_equivalent_resume,
)
from resume_validity.build.assign_names import (
    inject_name,
    inject_company_and_school,
    COMPANIES,
    SCHOOLS,
)


def sanitize(md: str) -> str:
    # remove any contact-like lines
    return re.sub(r"^.*(email|linkedin|github|phone)\s*[:@.].*$", "", md, flags=re.I | re.M)


def make_all_variants_by_quals(
    gemini: GeminiClient,
    role: str,
    url: str,
    basic: list[str],
    bonus: list[str],
    name_basic: str,
    name_variants: str,
    company: str,
    school: str,
) -> Dict[str, Any]:
    base_md = build_basic_resume(gemini, role, basic)
    base_md = sanitize(base_md)
    base_rendered = inject_company_and_school(inject_name(base_md, name_basic), company, school)

    preferred = []
    for added in bonus[: min(3, len(bonus))]:
        md = build_preferred_resume(gemini, role, base_md, added)
        md = sanitize(md)
        rendered = inject_company_and_school(inject_name(md, name_variants), company, school)
        preferred.append({"resume": rendered, "added_qualification": added})

    under = []
    for removed in basic[: min(3, len(basic))]:
        md = build_underqualified_resume(gemini, role, base_md, removed)
        md = sanitize(md)
        rendered = inject_company_and_school(inject_name(md, name_variants), company, school)
        under.append({"resume": rendered, "removed_qualification": removed})

    equal = []
    for _ in range(2):
        md = build_reworded_equivalent_resume(gemini, role, base_md)
        md = sanitize(md)
        rendered = inject_company_and_school(inject_name(md, name_variants), company, school)
        equal.append(rendered)

    return {
        "role": role,
        "url": url,
        "company": company,
        "school": school,
        "name_basic": name_basic,
        "name_variants": name_variants,
        "basic": base_rendered,
        "basic_qualifications": basic,
        "bonus_qualifications": bonus,
        "preferred": preferred,
        "under": under,
        "equal": equal,
    }


from typing import Optional, Tuple, List


def pick_from_harvest(base_dir: Path, role_slug: str) -> Optional[Tuple[str, List[str], List[str]]]:
    fp = base_dir / f"{role_slug}/passing_{role_slug}.jsonl"
    if not fp.exists():
        return None
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            basic = row.get("basic") or []
            bonus = row.get("bonus") or []
            bc = row.get("basic_count", len(basic))
            vc = row.get("validity_count", len(bonus))
            if bc >= 3 and vc >= 3:
                return row.get("url", ""), basic, bonus
    return None


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cs_url", type=str, default="")
    ap.add_argument("--non_url", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="/home/zs7353/resume_validity/data/constructed_examples")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If URLs not provided, attempt to pick from harvested data
    base = Path("/home/zs7353/resume_validity/data/harvest_top10")
    cs_url = args.cs_url
    non_url = args.non_url
    if not cs_url:
        pick = pick_from_harvest(base, "software_engineer")
        if pick:
            cs_url = pick[0]
        if not cs_url:
            cs_url = "https://stripe.com/jobs/search?gh_jid=7089532"

    if not non_url:
        pick = pick_from_harvest(base, "sales_representative")
        if pick:
            non_url = pick[0]

    g = GeminiClient()
    company = random.choice(COMPANIES)
    school = random.choice(SCHOOLS)
    # Load quals for CS
    cs_pick = pick_from_harvest(base, "software_engineer")
    if cs_pick and cs_pick[0] == cs_url:
        cs_basic, cs_bonus = cs_pick[1], cs_pick[2]
    else:
        raise SystemExit("CS quals not found in harvest")
    cs_json = make_all_variants_by_quals(g, "Software Engineer", cs_url, cs_basic, cs_bonus, "Alex Taylor", "Jordan Lee", company, school)
    with open(out_dir / "cs_example.json", "w", encoding="utf-8") as f:
        json.dump(cs_json, f, ensure_ascii=False, indent=2)

    # Load quals for Non-CS if available
    non_pick = pick_from_harvest(base, "sales_representative") if non_url else None
    if non_pick and non_pick[0] == non_url:
        company2 = random.choice(COMPANIES)
        school2 = random.choice(SCHOOLS)
        non_basic, non_bonus = non_pick[1], non_pick[2]
        non_json = make_all_variants_by_quals(g, "Sales Representative", non_url, non_basic, non_bonus, "Alex Taylor", "Jordan Lee", company2, school2)
        with open(out_dir / "non_example.json", "w", encoding="utf-8") as f:
            json.dump(non_json, f, ensure_ascii=False, indent=2)

    print("Wrote:", str(out_dir / "cs_example.json"))
    if non_pick and non_pick[0] == non_url:
        print("Wrote:", str(out_dir / "non_example.json"))


if __name__ == "__main__":
    main()


