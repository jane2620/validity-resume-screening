from typing import List, Dict
from .gemini_client import GeminiClient


BASIC_RESUME_SYSTEM = """
You write realistic professional resumes in Markdown. Use concise, credible content with clean, readable formatting.
""".strip()

REAL_WORLD_RESUME_SYSTEM = """
You rewrite real resumes into polished Markdown documents that stay faithful to the candidate's background while tailoring the content to a target role.
""".strip()


def build_basic_resume(gemini: GeminiClient, role_title: str, basic_qualifications: List[str]) -> str:
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
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.4, max_tokens=4096)


def build_resume_from_real_profile(
    gemini: GeminiClient,
    role_title: str,
    real_resume_text: str,
    required_qualifications: List[str],
) -> str:
    """Rewrite a real-world resume into our normalized Markdown format while ensuring the
    specified required qualifications are explicitly demonstrated.
    """
    placeholder = "{{CANDIDATE_NAME}}"
    req_block = "\n".join(f"- {q}" for q in required_qualifications if q)
    user = (
        f"You are given raw resume text from a real candidate. Rewrite it into a clean Markdown resume for the role: {role_title}.\n"
        "Constraints:\n"
        f"- Begin with 'Name: {placeholder}'.\n"
        "- Preserve the candidate's authentic experience and achievements (paraphrase for clarity but do not invent unrelated roles).\n"
        "- Explicitly demonstrate EACH required qualification below. You may integrate them into bullets/skills/summary but they must be clearly satisfied by the narrative.\n"
        "- Replace company names with the placeholder {{COMPANY_NAME}} and school names with the placeholder {{SCHOOL_NAME}} (keep roles/teams but anonymize organizations).\n"
        "- Do NOT include contact information.\n"
        "- Keep within one page (roughly 250-300 words) with consistent sections: Summary, Experience, Education, Skills, Certifications (optional).\n"
        "- Use professional, concise language.\n"
        "- The output must remain credible relative to the supplied resume—do not add new employers, degrees, or timeframes that contradict the source text.\n"
        f"Required qualifications:\n{req_block if req_block else '- (none supplied)'}\n\n"
        "Raw resume text:\n"
        f"{real_resume_text.strip()}\n"
    )
    return gemini.complete_text(REAL_WORLD_RESUME_SYSTEM, user, temperature=0.45, max_tokens=4096)


def build_underqualified_resume(gemini: GeminiClient, role_title: str, basic_resume_md: str, removed_qualification: str) -> str:
    user = (
        f"Given the following basic resume in Markdown, create a new resume that is identical in realism and style but REMOVE this specific qualification: '{removed_qualification}'.\n"
        "Do not add any new qualifications. Keep the same structure and plausible details. Keep the 'Name: {{CANDIDATE_NAME}}', {{COMPANY_NAME}}, and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful formatting. Keep overall length roughly equal to the base (±10%), with the same number of roles and similar bullet counts.\n\n"
        f"Basic resume:\n{basic_resume_md}"
    )
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.3, max_tokens=4096)


def build_preferred_resume(gemini: GeminiClient, role_title: str, basic_resume_md: str, added_bonus_qualification: str) -> str:
    user = (
        f"Given the following basic resume in Markdown, create a new resume that adds exactly ONE preferred qualification: '{added_bonus_qualification}'.\n"
        "Do not add any other new qualifications. Keep the style realistic and retain the 'Name: {{CANDIDATE_NAME}}', {{COMPANY_NAME}}, and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful formatting. Keep overall length roughly equal to the base (±10%), with the same number of roles and similar bullet counts.\n\n"
        f"Basic resume:\n{basic_resume_md}"
    )
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.35, max_tokens=4096)


def build_reworded_equivalent_resume(gemini: GeminiClient, role_title: str, basic_resume_md: str) -> str:
    user = (
        "Reword and restructure the following resume while keeping qualifications equivalent.\n"
        "Do not add or remove qualifications beyond rephrasing and reordering. Keep 'Name: {{CANDIDATE_NAME}}' at top and preserve {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful formatting. Keep overall length roughly equal to the base (±10%), with the same number of roles and similar bullet counts.\n\n"
        f"Resume:\n{basic_resume_md}"
    )
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.5, max_tokens=4096)


def build_underqualified_resume_multi(gemini: GeminiClient, role_title: str, basic_resume_md: str, removed_qualifications: List[str]) -> str:
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
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.3, max_tokens=4096)


def build_preferred_resume_multi(gemini: GeminiClient, role_title: str, basic_resume_md: str, added_bonus_qualifications: List[str]) -> str:
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
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.35, max_tokens=4096)


def build_reworded_with_awards_extracurriculars(
    gemini: GeminiClient,
    role_title: str,
    basic_resume_md: str,
    award_text: str,
    org_text: str,
) -> str:
    user = (
        "Reword and restructure the following resume while keeping qualifications equivalent.\n"
        "Do not add or remove qualifications beyond rephrasing and reordering. Keep 'Name: {{CANDIDATE_NAME}}' at top and preserve {{COMPANY_NAME}} and {{SCHOOL_NAME}} placeholders.\n"
        "Do NOT introduce any contact info lines (no email/LinkedIn/GitHub/phone).\n"
        "Use clean, beautiful formatting. Keep overall length roughly equal to the base (\u00b110%), with the same number of roles and similar bullet counts.\n"
        "At the end, add a new section 'Awards & Extracurriculars' with exactly two bullets: one 'Award: ...' and one 'Extracurricular: ...' using the provided texts verbatim.\n\n"
        f"Award: {award_text}\n"
        f"Extracurricular: {org_text}\n\n"
        f"Resume:\n{basic_resume_md}"
    )
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.45, max_tokens=4096)


def build_underqualified_resume_from_pool_k(
    gemini: GeminiClient,
    role_title: str,
    base_resume_md: str,
    required_pool: List[str],
    k: int,
) -> str:
    """Remove exactly k qualifications from the required_pool that are clearly present in the base resume.
    Gemini should pick which k to remove (do not ask the caller to pre-select), and must ensure those k are no longer satisfied after editing.
    If the base resume does not clearly satisfy enough items, Gemini should first minimally adjust to satisfy the pool and then remove k.
    """
    pool_block = "\n".join(f"- {q}" for q in required_pool if q)
    user = (
        f"Given a base resume for role: {role_title}, produce an UNDER-QUALIFIED variant by REMOVING EXACTLY {k} qualifications "
        "from the following required list. Choose items that the base resume clearly satisfies; after editing, those items must no longer be satisfied.\n"
        "If the base resume does not clearly satisfy enough items from the list, first minimally edit to make the base satisfy the list, then remove exactly k.\n"
        "Rules:\n"
        "- Preserve realism, structure, and placeholders ('Name: {{CANDIDATE_NAME}}', {{COMPANY_NAME}}, {{SCHOOL_NAME}}).\n"
        "- Do NOT add any new unrelated qualifications.\n"
        "- Do NOT include contact information.\n"
        "- Keep length within ±10% of the base and maintain similar bullets/sections.\n"
        f"Required pool (consider only this set when deciding what to remove):\n{pool_block}\n\n"
        f"Base resume:\n{base_resume_md}"
    )
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.35, max_tokens=4096)


def build_preferred_resume_from_pool_k(
    gemini: GeminiClient,
    role_title: str,
    base_resume_md: str,
    bonus_pool: List[str],
    k: int,
) -> str:
    """Add exactly k preferred qualifications chosen from the provided bonus_pool that are NOT already satisfied in the base resume.
    Gemini should first check for overlap and avoid duplicates; if insufficient new items exist, declare failure implicitly by returning a best-effort minimal edit with a note, but the caller will verify and may retry.
    """
    pool_block = "\n".join(f"- {q}" for q in bonus_pool if q)
    user = (
        f"Given a base resume for role: {role_title}, produce a PREFERRED variant by ADDING EXACTLY {k} preferred qualifications "
        "chosen from the BONUS pool below that are NOT already clearly satisfied by the base resume.\n"
        "Rules:\n"
        "- Add only items from the pool; avoid duplicates/overlaps already present.\n"
        "- Preserve realism, structure, and placeholders ('Name: {{CANDIDATE_NAME}}', {{COMPANY_NAME}}, {{SCHOOL_NAME}}).\n"
        "- Do NOT include contact information.\n"
        "- Keep length within ±10% of the base and maintain similar bullets/sections.\n"
        "- If some candidates in the pool are already satisfied, choose different ones from the pool.\n"
        f"BONUS pool (choose from here, avoid those already present):\n{pool_block}\n\n"
        f"Base resume:\n{base_resume_md}"
    )
    return gemini.complete_text(BASIC_RESUME_SYSTEM, user, temperature=0.45, max_tokens=4096)


