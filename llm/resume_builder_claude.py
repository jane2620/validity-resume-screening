from typing import List
from .openrouter_client import OpenRouterClient


BASIC_RESUME_SYSTEM = """
You write realistic professional resumes in Markdown. Use concise, credible content with clean, readable formatting.
""".strip()


def build_basic_resume(client: OpenRouterClient, role_title: str, basic_qualifications: List[str]) -> str:
    # Placeholder for name that can be easily replaced later
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
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.4, max_tokens=4096)


def build_underqualified_resume_multi(client: OpenRouterClient, role_title: str, basic_resume_md: str, removed_qualifications: List[str]) -> str:
    user = (
        f"Given the basic resume below, create an UNDER-QUALIFIED variant by REMOVING EXACTLY these {len(removed_qualifications)} qualifications.\n"
        + "\n".join([f"- {q}" for q in removed_qualifications]) + "\n"
        "Do not remove anything else and do not add new qualifications.\n"
        "When the removed qualification is about years of experience, ensure that every other part of the resume remains generally unchanged except the years of experience.\n"
        "Keep 'Name: {{CANDIDATE_NAME}}' and the {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful Markdown formatting. Keep overall length roughly equal to the base (±10%), maintaining the same number of roles and similar bullet counts.\n\n"
        f"Basic resume:\n{basic_resume_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.3, max_tokens=4096)


def build_preferred_resume_multi(client: OpenRouterClient, role_title: str, basic_resume_md: str, added_bonus_qualifications: List[str]) -> str:
    user = (
        f"Given the basic resume below, create a PREFERRED variant by ADDING EXACTLY these {len(added_bonus_qualifications)} preferred qualifications.\n"
        + "\n".join([f"- {q}" for q in added_bonus_qualifications]) + "\n"
        "Do not add any other new qualifications.\n"
        "When the added qualification is about years of experience, ensure that every other part of the resume remains generally unchanged except the years of experience.\n"
        "Keep 'Name: {{CANDIDATE_NAME}}' and the {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful Markdown formatting. Keep overall length roughly equal to the base (±10%), maintaining the same number of roles and similar bullet counts.\n\n"
        f"Basic resume:\n{basic_resume_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.35, max_tokens=4096)


def build_reworded_equivalent_resume(client: OpenRouterClient, role_title: str, basic_resume_md: str) -> str:
    user = (
        "Reword and restructure the following resume while keeping qualifications equivalent.\n"
        "Do not add or remove qualifications beyond rephrasing and reordering. Keep 'Name: {{CANDIDATE_NAME}}' at top and preserve {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful formatting. Keep overall length roughly equal to the base (±10%), with the same number of roles and similar bullet counts.\n\n"
        f"Resume:\n{basic_resume_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.5, max_tokens=4096)


def build_reworded_with_awards_extracurriculars(
    client: OpenRouterClient,
    role_title: str,
    basic_resume_md: str,
    award_text: str,
    org_text: str,
) -> str:
    user = (
        "Reword and restructure the following resume while keeping qualifications equivalent.\n"
        "Do not add or remove qualifications beyond rephrasing and reordering. Keep 'Name: {{CANDIDATE_NAME}}' at top and preserve {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful formatting. Keep overall length roughly equal to the base (±10%), with the same number of roles and similar bullet counts.\n"
        "At the end, add a new section 'Awards & Extracurriculars' with exactly two bullets: one 'Award: ...' and one 'Extracurricular: ...' using the provided texts verbatim.\n\n"
        f"Award: {award_text}\n"
        f"Extracurricular: {org_text}\n\n"
        f"Resume:\n{basic_resume_md}"
    )
    return client.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.45, max_tokens=4096)


