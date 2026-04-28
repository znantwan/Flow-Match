"""FlowMatch – Hybrid Matching and Recommendation Engine."""

import json
import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Matching methodologies
# ---------------------------------------------------------------------------

METHODOLOGIES = {
    "hybrid": {
        "id": "hybrid",
        "name": "Hybrid (Default)",
        "emoji": "⚡",
        "description": (
            "Balances baseline competence with growth alignment, biased toward "
            "developmental stretch. Penalises 'clone' matches where the employee "
            "already does this exact work every day."
        ),
        "fit_weight": 0.40,
        "stretch_weight": 0.60,
        "clone_threshold": 0.85,
        "clone_penalty_factor": 2.0,
        "availability_weight": 0.00,
    },
    "efficiency": {
        "id": "efficiency",
        "name": "Efficiency First",
        "emoji": "🚀",
        "description": (
            "Prioritises employees who can hit the ground running with minimal "
            "ramp-up. Best for urgent deadlines where training time is costly. "
            "Clone matches are acceptable — experience is the point."
        ),
        "fit_weight": 0.80,
        "stretch_weight": 0.20,
        "clone_threshold": 0.95,
        "clone_penalty_factor": 0.5,
        "availability_weight": 0.00,
    },
    "growth": {
        "id": "growth",
        "name": "Growth First",
        "emoji": "🌱",
        "description": (
            "Maximises career development value. Accepts more ramp-up in exchange "
            "for strong skill-building alignment. Best for retention and long-term "
            "talent investment. Heavily penalises clone matches."
        ),
        "fit_weight": 0.20,
        "stretch_weight": 0.80,
        "clone_threshold": 0.80,
        "clone_penalty_factor": 3.0,
        "availability_weight": 0.00,
    },
    "balanced": {
        "id": "balanced",
        "name": "Balanced",
        "emoji": "⚖️",
        "description": (
            "Equal weight between competence and growth with no philosophical bias. "
            "A neutral baseline that neither over-indexes on speed nor development."
        ),
        "fit_weight": 0.50,
        "stretch_weight": 0.50,
        "clone_threshold": 0.85,
        "clone_penalty_factor": 2.0,
        "availability_weight": 0.00,
    },
    "availability": {
        "id": "availability",
        "name": "Availability-Weighted",
        "emoji": "📅",
        "description": (
            "Hybrid scoring with a bonus for employees who have more bandwidth "
            "than the project requires. Surfaces people who can give the project "
            "their full attention rather than squeezing it in."
        ),
        "fit_weight": 0.35,
        "stretch_weight": 0.50,
        "clone_threshold": 0.85,
        "clone_penalty_factor": 2.0,
        "availability_weight": 0.15,
    },
}


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_embedding(client, text, model="text-embedding-3-small"):
    return client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=model,
    ).data[0].embedding


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def _employee_current_text(emp):
    return (
        f"Role: {emp.get('role', '')}. "
        f"Department: {emp.get('department', '')}. "
        f"Skills: {emp.get('current_skills', '')}. "
        f"Experience: {emp.get('years_experience', '')} years."
    )


def _employee_target_text(emp):
    return (
        f"Career growth targets: {emp.get('target_skills', '')}. "
        f"Current role: {emp.get('role', '')} in {emp.get('department', '')}."
    )


def _project_requirement_text(proj):
    return (
        f"Project: {proj.get('project_name', '')}. "
        f"Description: {proj.get('description', '')}. "
        f"Required skills: {proj.get('required_skills', '')}. "
        f"Industry context: {proj.get('industry_context', '')}."
    )


def _project_stretch_text(proj):
    skills = proj.get('desired_skills') or proj.get('required_skills', '')
    return (
        f"Skills involved: {skills}. "
        f"Context: {proj.get('description', '')}."
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _weekly_capacity(emp):
    try:
        return float(emp.get("available_hours_per_week", 0))
    except (ValueError, TypeError):
        return 0.0


def _hours_per_week_needed(proj):
    try:
        total = float(proj.get("hours_needed", 0))
        weeks = max(float(proj.get("duration_weeks", 1)), 1)
        return total / weeks
    except (ValueError, TypeError):
        return 0.0


def compute_match_scores(
    emp_embed, emp_target_embed, proj_embed, proj_stretch_embed,
    methodology=None, emp_capacity=0.0, needed_weekly=0.0,
):
    if methodology is None:
        methodology = METHODOLOGIES["hybrid"]

    fit = cosine_similarity(emp_embed, proj_embed)
    stretch = cosine_similarity(emp_target_embed, proj_stretch_embed)

    clone_penalty = (
        max(0.0, fit - methodology["clone_threshold"])
        * methodology["clone_penalty_factor"]
    )
    base = methodology["fit_weight"] * fit + methodology["stretch_weight"] * stretch - clone_penalty

    # Availability bonus: reward employees with excess bandwidth
    avail_bonus = 0.0
    avail_weight = methodology.get("availability_weight", 0.0)
    if avail_weight > 0 and needed_weekly > 0:
        excess_ratio = min(max(emp_capacity - needed_weekly, 0.0) / needed_weekly, 1.0)
        avail_bonus = excess_ratio * avail_weight

    composite = base + avail_bonus

    return {
        "fit_score": round(fit, 4),
        "stretch_score": round(stretch, 4),
        "clone_penalty": round(clone_penalty, 4),
        "availability_bonus": round(avail_bonus, 4),
        "composite_score": round(composite, 4),
        "methodology_id": methodology["id"],
    }


# ---------------------------------------------------------------------------
# LLM rationale generation
# ---------------------------------------------------------------------------

_RATIONALE_PROMPT = """\
You are an internal talent marketplace advisor at a large company.

Given an employee and a short-term project opportunity, write a concise match rationale.

Employee
  Name:              {name}
  Role:              {role} — {department}
  Current Skills:    {current_skills}
  Target Skills:     {target_skills}
  Availability:      {available_hours_per_week} hrs/week until {availability_end_date}

Project
  Name:              {project_name}
  Manager:           {manager_name}
  Description:       {description}
  Required Skills:   {required_skills}
  Hours Needed:      {hours_needed} hrs over {duration_weeks} weeks
  Industry Context:  {industry_context}

Matching Methodology: {methodology_name}
  Fit Score (baseline competence match): {fit_score:.1%}
  Stretch Score (growth alignment):      {stretch_score:.1%}
  Composite Score:                       {composite_score:.1%}

Return ONLY a JSON object with exactly these two keys:
{{
  "fit_rationale": "2-3 sentences: why this employee has enough baseline knowledge to contribute meaningfully.",
  "stretch_rationale": "2-3 sentences: how this project advances the employee's stated career goals and builds skills toward promotion."
}}"""


def generate_rationale(client, employee, project, scores, methodology_name="Hybrid (Default)"):
    prompt = _RATIONALE_PROMPT.format(
        **{**employee, **project},
        **scores,
        methodology_name=methodology_name,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# Embedding cache builder
# ---------------------------------------------------------------------------

def build_employee_embeddings(client, employees):
    """Return two dicts keyed by employee_id: current-skill embeds and target embeds."""
    current_embeds, target_embeds = {}, {}
    for emp in employees:
        eid = emp["employee_id"]
        current_embeds[eid] = get_embedding(client, _employee_current_text(emp))
        target_embeds[eid] = get_embedding(client, _employee_target_text(emp))
    return current_embeds, target_embeds


# ---------------------------------------------------------------------------
# Core matching function
# ---------------------------------------------------------------------------

def rank_employees_for_project(
    client,
    project,
    employees,
    current_embeds,
    target_embeds,
    top_n=3,
    methodology_id="hybrid",
):
    """Return the top-N employee matches for a project, with LLM rationales."""
    methodology = METHODOLOGIES.get(methodology_id, METHODOLOGIES["hybrid"])

    proj_embed        = get_embedding(client, _project_requirement_text(project))
    proj_stretch_embed = get_embedding(client, _project_stretch_text(project))
    needed_weekly     = _hours_per_week_needed(project)

    scored = []
    for emp in employees:
        eid = emp["employee_id"]
        cap = _weekly_capacity(emp)
        scores = compute_match_scores(
            current_embeds[eid],
            target_embeds[eid],
            proj_embed,
            proj_stretch_embed,
            methodology=methodology,
            emp_capacity=cap,
            needed_weekly=needed_weekly,
        )
        if cap < needed_weekly:
            scores = dict(scores)
            scores["composite_score"] = round(scores["composite_score"] - 0.05, 4)
            scores["capacity_warning"] = True

        scored.append((emp, scores))

    scored.sort(key=lambda x: x[1]["composite_score"], reverse=True)

    results = []
    for emp, scores in scored[:top_n]:
        rationale = generate_rationale(
            client, emp, project, scores,
            methodology_name=methodology["name"],
        )
        results.append({
            "employee_id":            emp["employee_id"],
            "name":                   emp["name"],
            "role":                   emp["role"],
            "department":             emp.get("department", ""),
            "available_hours_per_week": emp.get("available_hours_per_week"),
            "availability_end_date":  emp.get("availability_end_date"),
            "scores":                 scores,
            "fit_rationale":          rationale["fit_rationale"],
            "stretch_rationale":      rationale["stretch_rationale"],
        })

    return results


# ---------------------------------------------------------------------------
# Multi-methodology comparison
# ---------------------------------------------------------------------------

def rank_employees_no_rationale(
    project, employees, current_embeds, target_embeds,
    proj_embed, proj_stretch_embed, methodology_id="hybrid",
):
    """Score all employees for a project without calling the LLM. Used for comparison."""
    methodology   = METHODOLOGIES.get(methodology_id, METHODOLOGIES["hybrid"])
    needed_weekly = _hours_per_week_needed(project)

    scored = []
    for emp in employees:
        eid = emp["employee_id"]
        cap = _weekly_capacity(emp)
        scores = compute_match_scores(
            current_embeds[eid],
            target_embeds[eid],
            proj_embed,
            proj_stretch_embed,
            methodology=methodology,
            emp_capacity=cap,
            needed_weekly=needed_weekly,
        )
        if cap < needed_weekly:
            scores = dict(scores)
            scores["composite_score"] = round(scores["composite_score"] - 0.05, 4)
            scores["capacity_warning"] = True
        scored.append((emp, scores))

    scored.sort(key=lambda x: x[1]["composite_score"], reverse=True)
    return [
        {
            "employee_id": emp["employee_id"],
            "name":        emp["name"],
            "role":        emp["role"],
            "scores":      scores,
        }
        for emp, scores in scored[:3]
    ]


_COMPARISON_INSIGHT_PROMPT = """\
You are an internal talent advisor helping a manager understand why different matching \
algorithms recommended different employees for the same project.

PROJECT
  Name:            {project_name}
  Required Skills: {required_skills}
  Industry Context:{industry_context}
  Hours Needed:    {hours_needed} hrs over {duration_weeks} weeks

RESULTS BY METHODOLOGY
{methodology_summaries}

CONSENSUS PICKS (appear in 3 or more methodologies' top-3):
{consensus}

METHODOLOGY-SPECIFIC PICKS (appear in only 1 methodology's top-3):
{unique_picks}

Write a concise analysis in 3–4 paragraphs covering:
1. Who the "safe" consensus picks are and why most algorithms agree on them.
2. The key trade-off each methodology is making — what it optimises for and what it sacrifices.
3. Which methodology and candidate you would recommend for this specific project given its context, and why.

Be concrete and reference specific employees and scores. Do not use bullet points — write in prose."""


def generate_comparison_insight(client, project, all_method_results):
    """Call GPT-4o-mini to explain why different methodologies diverged."""
    # Build methodology summaries text
    summaries = []
    for mid, matches in all_method_results.items():
        m_info = METHODOLOGIES[mid]
        lines = [f"  {m_info['emoji']} {m_info['name']}:"]
        for rank, m in enumerate(matches, 1):
            s = m["scores"]
            lines.append(
                f"    #{rank} {m['name']} ({m['role']}) — "
                f"Fit {s['fit_score']:.0%}, Stretch {s['stretch_score']:.0%}, "
                f"Composite {s['composite_score']:.0%}"
            )
        summaries.append("\n".join(lines))

    # Identify consensus and unique picks
    appearance_count: dict[str, int] = {}
    appearance_methods: dict[str, list] = {}
    for mid, matches in all_method_results.items():
        for m in matches:
            name = m["name"]
            appearance_count[name] = appearance_count.get(name, 0) + 1
            appearance_methods.setdefault(name, []).append(METHODOLOGIES[mid]["name"])

    consensus  = [n for n, c in appearance_count.items() if c >= 3]
    unique     = [n for n, c in appearance_count.items() if c == 1]

    prompt = _COMPARISON_INSIGHT_PROMPT.format(
        project_name=project.get("project_name", ""),
        required_skills=project.get("required_skills", ""),
        industry_context=project.get("industry_context", ""),
        hours_needed=project.get("hours_needed", ""),
        duration_weeks=project.get("duration_weeks", ""),
        methodology_summaries="\n\n".join(summaries),
        consensus=", ".join(consensus) if consensus else "None — all methodologies diverged",
        unique_picks=", ".join(
            f"{n} (only via {appearance_methods[n][0]})" for n in unique
        ) if unique else "None",
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return resp.choices[0].message.content
