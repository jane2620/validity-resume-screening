from typing import Dict, List, Any
from dataclasses import dataclass
from .gemini_client import GeminiClient


@dataclass
class Qualification:
    text: str
    kind: str  # "basic" or "bonus"


def extract_qualifications(gemini: GeminiClient, job_page_text: str) -> Dict[str, List[Qualification]]:
    schema_hint = '{"basic": [{"text": "", "kind": "basic"}], "bonus": [{"text": "", "kind": "bonus"}]}'
    system = (
        "Extract qualifications from a job page, grouping similar or scattered points into single consolidated items.\n"
        "Group technologies and close variants (e.g., Python/PyTorch/TensorFlow) under one consolidated point if they express the same underlying requirement.\n"
        "Classify each as 'basic' (required) or 'bonus' (preferred).\n"
        "Return JSON with 'basic' and 'bonus' lists. Each list item: {\"text\": <consolidated requirement>, \"kind\": \"basic\"|\"bonus\"}.\n"
        "Keep items distinct enough so that removing or adding one item changes the candidate meaningfully."
    )
    user = (
        "Job Description Page (entire text follows):\n" + job_page_text + "\n\n"
        "Return only minified JSON with keys 'basic' and 'bonus'."
    )
    data = gemini.complete_json(system, user, schema_hint=schema_hint, temperature=0.1)
    basics: List[Qualification] = [Qualification(text=i.get('text','').strip(), kind='basic') for i in data.get('basic', []) if i.get('text')]
    bonuses: List[Qualification] = [Qualification(text=i.get('text','').strip(), kind='bonus') for i in data.get('bonus', []) if i.get('text')]
    return {"basic": basics, "bonus": bonuses}


