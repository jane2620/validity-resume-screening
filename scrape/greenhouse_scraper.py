import json, time, html, re, traceback
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup


@dataclass
class JobPosting:
    source: str
    company: str
    title: str
    url: str
    content_html: str
    content_text: str


GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true"


def _http_get_json(url: str, timeout_sec: float = 20.0, retries: int = 3, backoff_sec: float = 1.5) -> Optional[Dict]:
    attempt = 0
    while attempt <= retries:
        try:
            r = requests.get(url, timeout=timeout_sec)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"HTTP error: {e} for {url}")
            traceback.print_exc()
            attempt += 1
            if attempt <= retries:
                wait = backoff_sec * (2 ** (attempt - 1))
                print(f"Retrying in {wait:.1f}s ...")
                try:
                    time.sleep(wait)
                except Exception as se:
                    print(f"Sleep error: {se}")
    return None


def strip_html(text: str) -> str:
    try:
        soup = BeautifulSoup(text or "", "lxml")
        for tag in soup(["script","style","noscript"]):
            try:
                tag.decompose()
            except Exception:
                pass
        clean = soup.get_text(" ", strip=True)
        clean = html.unescape(clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    except Exception as e:
        print(f"Failed to strip html: {e}")
        traceback.print_exc()
        return text or ""


def fetch_greenhouse_jobs(companies: List[str], max_jobs_per_company: int = 50, timeout_sec: float = 20.0, retries: int = 3, backoff_sec: float = 1.5) -> List[JobPosting]:
    postings: List[JobPosting] = []
    for company in companies:
        url = GREENHOUSE_API.format(company=company)
        print(f"Fetching Greenhouse jobs for {company} ...")
        data = _http_get_json(url, timeout_sec=timeout_sec, retries=retries, backoff_sec=backoff_sec)
        if not data or "jobs" not in data:
            print(f"No data or 'jobs' missing for {company}")
            continue
        jobs = data.get("jobs", [])[:max_jobs_per_company]
        for j in jobs:
            try:
                title = str(j.get("title", ""))
                abs_url = str(j.get("absolute_url", ""))
                content = str(j.get("content", ""))
                postings.append(JobPosting(
                    source="greenhouse",
                    company=company,
                    title=title,
                    url=abs_url,
                    content_html=content,
                    content_text=strip_html(content),
                ))
            except Exception as e:
                print(f"Failed to parse job for {company}: {e}")
                traceback.print_exc()
                continue
    return postings


ROLE_FAMILIES = {
    "Software Engineer": ["frontend","front-end","backend","back-end","full stack","software engineer","developer"],
    "Data Scientist": ["data scientist","research scientist","applied scientist","quant","data analyst","analytics"],
    "ML Engineer": ["machine learning","ml engineer","applied ml","ai engineer","ai research","research engineer","llm","nlp","deep"],
    "DevOps Engineer": ["devops","sre","site reliability","platform","infrastructure","cloud engineer","reliability"],
    "Product Manager": ["product manager","product management","pm "],
    "Financial Analyst": ["financial analyst","finance analyst","investment banking","credit analyst","risk analyst","treasury"],
    "HR Specialist": ["human resources","hr ","recruiter","talent acquisition","people ops","benefits","payroll"],
    "Retail Associate": ["retail","store","associate","cashier","customer service"],
    "Sales Representative": ["sales","account executive","business development","bdr","sdr","account manager"],
    "Customer Support": ["customer support","support specialist","help desk","service desk"],
}


def map_title_to_role(title: str) -> str:
    t = (title or "").lower()
    for role, keys in ROLE_FAMILIES.items():
        if any(k in t for k in keys):
            return role
    return "Software Engineer"


def pick_top_roles(posts: List[JobPosting], top_k: int = 10, max_cs_roles: int = 4) -> List[str]:
    counts: Dict[str, int] = {}
    for p in posts:
        r = map_title_to_role(p.title)
        counts[r] = counts.get(r, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    selected: List[str] = []
    cs_roles = {"Software Engineer","ML Engineer","Data Scientist"}
    cs_count = 0
    for r, _ in ranked:
        if r in cs_roles:
            if cs_count >= max_cs_roles:
                continue
            cs_count += 1
        selected.append(r)
        if len(selected) >= top_k:
            break
    print(f"Selected top roles (<= {max_cs_roles} CS-related): {selected[:top_k]}")
    return selected


def bucket_by_role(posts: List[JobPosting]) -> Dict[str, List[JobPosting]]:
    buckets: Dict[str, List[JobPosting]] = {r: [] for r in ROLE_FAMILIES.keys()}
    for p in posts:
        r = map_title_to_role(p.title)
        buckets.setdefault(r, []).append(p)
    return buckets


