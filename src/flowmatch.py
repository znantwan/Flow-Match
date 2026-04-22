"""FlowMatch – Hybrid Matching and Recommendation Engine."""

import json
import numpy as np
from openai import OpenAI


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
# Text builders – convert structured rows into prose for embedding
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

_STRETCH_WEIGHT = 0.60   # 60 % weight on growth alignment (per design brief)
_FIT_WEIGHT = 0.40
_CLONE_THRESHOLD = 0.85  # penalise employees who are an exact copy of requirements
_CLONE_PENALTY_FACTOR = 2.0


def compute_match_scores(emp_embed, emp_target_embed, proj_embed, proj_stretch_embed):
    fit = cosine_similarity(emp_embed, proj_embed)
    stretch = cosine_similarity(emp_target_embed, proj_stretch_embed)

    clone_penalty = max(0.0, fit - _CLONE_THRESHOLD) * _CLONE_PENALTY_FACTOR
    composite = _FIT_WEIGHT * fit + _STRETCH_WEIGHT * stretch - clone_penalty

    return {
        "fit_score": round(fit, 4),
        "stretch_score": round(stretch, 4),
        "clone_penalty": round(clone_penalty, 4),
        "composite_score": round(composite, 4),
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

Match Scores
  Fit Score (baseline competence match): {fit_score:.1%}
  Stretch Score (growth alignment):      {stretch_score:.1%}
  Composite Score:                       {composite_score:.1%}

Return ONLY a JSON object with exactly these two keys:
{{
  "fit_rationale": "2-3 sentences: why this employee has enough baseline knowledge to contribute meaningfully.",
  "stretch_rationale": "2-3 sentences: how this project advances the employee's stated career goals and builds skills toward promotion."
}}"""


def generate_rationale(client, employee, project, scores):
    prompt = _RATIONALE_PROMPT.format(
        **{**employee, **project},
        **scores,
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


def rank_employees_for_project(
    client,
    project,
    employees,
    current_embeds,
    target_embeds,
    top_n=3,
):
    """Return the top-N employee matches for a project, with LLM rationales."""
    proj_embed = get_embedding(client, _project_requirement_text(project))
    proj_stretch_embed = get_embedding(client, _project_stretch_text(project))

    scored = []
    needed_weekly = _hours_per_week_needed(project)

    for emp in employees:
        eid = emp["employee_id"]
        scores = compute_match_scores(
            current_embeds[eid],
            target_embeds[eid],
            proj_embed,
            proj_stretch_embed,
        )
        # Soft penalty when employee lacks the bandwidth
        if _weekly_capacity(emp) < needed_weekly:
            scores = dict(scores)
            scores["composite_score"] = round(scores["composite_score"] - 0.05, 4)
            scores["capacity_warning"] = True

        scored.append((emp, scores))

    scored.sort(key=lambda x: x[1]["composite_score"], reverse=True)

    results = []
    for emp, scores in scored[:top_n]:
        rationale = generate_rationale(client, emp, project, scores)
        results.append({
            "employee_id": emp["employee_id"],
            "name": emp["name"],
            "role": emp["role"],
            "department": emp["department"],
            "available_hours_per_week": emp.get("available_hours_per_week"),
            "availability_end_date": emp.get("availability_end_date"),
            "scores": scores,
            "fit_rationale": rationale["fit_rationale"],
            "stretch_rationale": rationale["stretch_rationale"],
        })

    return results
