import streamlit as st
import pandas as pd
import sys, os, json, io
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from flowmatch import (
    METHODOLOGIES,
    build_employee_embeddings,
    rank_employees_for_project,
    rank_employees_no_rationale,
    generate_comparison_insight,
    get_embedding,
    _project_requirement_text,
    _project_stretch_text,
)

st.set_page_config(
    page_title="FlowMatch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data helpers ───────────────────────────────────────────────────────────────

def normalise_employees(df):
    df = df.rename(columns={
        "Name":                  "name",
        "Role":                  "role",
        "Current Skills":        "current_skills",
        "Target Skills":         "target_skills",
        "Capacity (Hours/Week)": "available_hours_per_week",
        "Capacity End Date":     "availability_end_date",
    })
    if "employee_id" not in df.columns:
        df.insert(0, "employee_id", [f"EMP{i+1:03d}" for i in range(len(df))])
    if "department" not in df.columns:
        df["department"] = ""
    df["availability_end_date"] = (
        pd.to_datetime(df["availability_end_date"], errors="coerce")
        .dt.strftime("%Y-%m-%d").fillna("")
    )
    return df.fillna("")


def normalise_projects(df):
    df = df.rename(columns={
        "Project Name":                 "project_name",
        "Required Skills":              "required_skills",
        "Nice to Have Skills":          "desired_skills",
        "Industry Context":             "industry_context",
        "Capacity Needed (Hours/Week)": "hours_per_week",
    })
    df["Start Date"] = pd.to_datetime(df.get("Start Date"), errors="coerce")
    df["End Date"]   = pd.to_datetime(df.get("End Date"),   errors="coerce")
    df["duration_weeks"] = (
        ((df["End Date"] - df["Start Date"]).dt.days / 7)
        .round().clip(lower=1).fillna(2).astype(int)
    )
    df["hours_needed"] = df["hours_per_week"] * df["duration_weeks"]
    if "project_id" not in df.columns:
        df.insert(0, "project_id", [f"PROJ{i+1:03d}" for i in range(len(df))])
    for col in ["manager_name", "department", "description"]:
        if col not in df.columns:
            df[col] = ""
    return df.fillna("")


@st.cache_data(show_spinner=False)
def load_from_upload(emp_bytes, proj_bytes):
    emp_df  = pd.read_excel(io.BytesIO(emp_bytes))
    proj_df = pd.read_excel(io.BytesIO(proj_bytes))
    return normalise_employees(emp_df), normalise_projects(proj_df)


@st.cache_data(show_spinner=False)
def load_from_repo():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    def find(keywords):
        for ext, reader in [(".xlsx", pd.read_excel), (".xls", pd.read_excel), (".csv", pd.read_csv)]:
            for fname in sorted(os.listdir(data_dir)):
                if fname.lower().endswith(ext) and any(k in fname.lower() for k in keywords):
                    return reader(os.path.join(data_dir, fname))
        raise FileNotFoundError(f"No file matching {keywords} in {data_dir}")
    return normalise_employees(find(["employee"])), normalise_projects(find(["project"]))


# ── UI helpers ─────────────────────────────────────────────────────────────────

def score_bar(label, value, color):
    pct = int(value * 100)
    return f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="font-size:0.78rem;color:#6b7280">{label}</span>
        <span style="font-size:0.82rem;font-weight:700;color:{color}">{pct}%</span>
      </div>
      <div style="background:#e5e7eb;border-radius:6px;height:7px">
        <div style="width:{pct}%;background:{color};border-radius:6px;height:7px"></div>
      </div>
    </div>"""


MEDALS = ["🥇", "🥈", "🥉"]
METHOD_COLORS = {
    "hybrid":       "#4C8BF5",
    "efficiency":   "#E8453C",
    "growth":       "#34A853",
    "balanced":     "#9C27B0",
    "availability": "#F5A623",
}


def render_match_cards(matches, project):
    for i, m in enumerate(matches):
        s = m["scores"]
        cap_warn = "  ⚠️ *Capacity may be tight*" if s.get("capacity_warning") else ""
        with st.container(border=True):
            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"### {MEDALS[i]} {m['name']}")
                role_line = m["role"]
                if m.get("department"):
                    role_line += f" · {m['department']}"
                st.markdown(f"**{role_line}**")
                st.caption(
                    f"Available **{m['available_hours_per_week']} hrs / week** "
                    f"until {m['availability_end_date']}{cap_warn}"
                )
            with right:
                st.markdown(
                    score_bar("Fit — baseline competence", s["fit_score"], "#4C8BF5")
                    + score_bar("Stretch — growth alignment", s["stretch_score"], "#F5A623")
                    + score_bar("Composite score", s["composite_score"], "#34A853"),
                    unsafe_allow_html=True,
                )
            st.markdown("---")
            fc, sc = st.columns(2)
            with fc:
                st.markdown("**Fit Rationale**")
                st.markdown(m["fit_rationale"])
            with sc:
                st.markdown("**Stretch Rationale**")
                st.markdown(m["stretch_rationale"])

    result_json = json.dumps(
        {"project_id": project["project_id"],
         "project_name": project["project_name"],
         "matches": matches},
        indent=2,
    )
    st.download_button(
        "⬇️  Download results (JSON)", result_json,
        file_name=f"flowmatch_{project['project_id']}.json",
        mime="application/json",
        key=f"dl_single_{project['project_id']}",
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ FlowMatch")
    st.caption("Internal Capacity & Growth Marketplace")
    st.divider()

    api_key = st.text_input(
        "OpenAI API Key", type="password", placeholder="sk-…",
        help="Used only for this session. Never stored.",
    )

    st.divider()
    st.markdown("**Data Source**")
    use_upload = st.toggle("Upload my own files", value=False)

    emp_file = proj_file = None
    if use_upload:
        emp_file  = st.file_uploader("Employee profiles (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
        proj_file = st.file_uploader("Project profiles (.xlsx / .csv)",  type=["xlsx", "xls", "csv"])
    else:
        st.caption("Reading from the repo's `data/` folder.")

    st.divider()
    st.markdown("**Matching Methodologies**")
    for mid, m in METHODOLOGIES.items():
        st.caption(f"{m['emoji']} **{m['name']}** — {m['description'][:80]}…")


# ── Load data ──────────────────────────────────────────────────────────────────

employees_df = projects_df = None
load_error = None
try:
    if use_upload and emp_file and proj_file:
        employees_df, projects_df = load_from_upload(emp_file.read(), proj_file.read())
    elif not use_upload:
        employees_df, projects_df = load_from_repo()
except Exception as e:
    load_error = str(e)

# ── Page header ────────────────────────────────────────────────────────────────

st.markdown("## ⚡ FlowMatch")
st.markdown(
    "Match employees who have capacity and growth goals with projects that need them. "
    "Every recommendation includes a **Fit Rationale** and a **Stretch Rationale**."
)
st.divider()

if load_error:
    st.error(f"Data load failed: {load_error}")
    st.stop()

if employees_df is None:
    st.info("Upload your files in the sidebar, or toggle off 'Upload my own files' to use the repo data.")
    st.stop()

employees = employees_df.to_dict(orient="records")
projects  = projects_df.to_dict(orient="records")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", len(employees))
c2.metric("Open Projects", len(projects))
c3.metric("Methodologies", len(METHODOLOGIES))
embeds_ready = "current_embeds" in st.session_state
c4.metric("Engine Status", "Ready ✅" if embeds_ready else "Needs setup ⚙️")

st.divider()

# ── Embed step ─────────────────────────────────────────────────────────────────

if not embeds_ready:
    st.markdown("### Step 1 — Build employee embeddings")
    st.markdown(
        "Calls the OpenAI embedding API once for all employees and caches the results "
        "in memory. Only needed once per session."
    )
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to continue.")
        st.stop()
    if st.button("Build embeddings", type="primary", use_container_width=True):
        client = OpenAI(api_key=api_key)
        with st.spinner(f"Embedding {len(employees)} employees… ~30 seconds"):
            cur, tgt = build_employee_embeddings(client, employees)
        st.session_state.update({
            "current_embeds": cur,
            "target_embeds":  tgt,
            "api_key":        api_key,
        })
        st.success("Done — embeddings cached for this session.")
        st.rerun()
    st.stop()

with st.expander("⚙️  Re-embed / reset session"):
    st.caption("Use this if you've swapped data files or want to change the API key.")
    if st.button("Clear everything and restart"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_match, tab_compare, tab_batch, tab_data = st.tabs([
    "🎯  Match a Project",
    "🔬  Compare All Methods",
    "📊  Batch Run",
    "🗂  View Data",
])

# ── Tab 1: Single match ────────────────────────────────────────────────────────

with tab_match:
    st.markdown("### Select a project and matching methodology")

    proj_options = {f"{p['project_id']} — {p['project_name']}": p for p in projects}
    t1_col1, t1_col2 = st.columns([2, 3])

    with t1_col1:
        selected_label = st.selectbox("Project", list(proj_options.keys()))
        project = proj_options[selected_label]

    with t1_col2:
        method_labels = [
            f"{m['emoji']}  {m['name']}" for m in METHODOLOGIES.values()
        ]
        method_ids = list(METHODOLOGIES.keys())
        selected_method_label = st.radio(
            "Methodology", method_labels, horizontal=False,
            help="Choose how to weight fit vs. growth when ranking candidates.",
        )
        selected_method_id = method_ids[method_labels.index(selected_method_label)]

    selected_method = METHODOLOGIES[selected_method_id]
    st.caption(f"*{selected_method['description']}*")

    with st.expander("Project details"):
        d1, d2, d3 = st.columns(3)
        d1.markdown(f"**Manager:** {project.get('manager_name') or '—'}")
        d2.markdown(f"**Hours needed:** {int(project.get('hours_needed', 0))} hrs total")
        d3.markdown(f"**Duration:** {project.get('duration_weeks')} weeks")
        st.markdown(f"**Required skills:** {project['required_skills']}")
        if project.get("desired_skills"):
            st.markdown(f"**Nice to have:** {project['desired_skills']}")
        if project.get("industry_context"):
            st.markdown(f"**Industry context:** {project['industry_context']}")

    run_col, _ = st.columns([1, 3])
    find_clicked = run_col.button(
        f"Find top 3 matches", type="primary", use_container_width=True
    )

    cache_key = f"matches_{project['project_id']}_{selected_method_id}"

    if find_clicked:
        client = OpenAI(api_key=st.session_state["api_key"])
        with st.spinner(f"Running {selected_method['name']} — scoring + generating rationales…"):
            matches = rank_employees_for_project(
                client, project, employees,
                st.session_state["current_embeds"],
                st.session_state["target_embeds"],
                top_n=3,
                methodology_id=selected_method_id,
            )
        st.session_state[cache_key] = matches

    if cache_key in st.session_state:
        st.markdown(
            f"#### {selected_method['emoji']} {selected_method['name']} — "
            f"Top 3 matches for **{project['project_name']}**"
        )
        render_match_cards(st.session_state[cache_key], project)


# ── Tab 2: Compare all methodologies ──────────────────────────────────────────

with tab_compare:
    st.markdown("### Compare all 5 methodologies side-by-side")
    st.markdown(
        "Runs every methodology against the same project and shows where they agree "
        "and disagree — then explains *why* in plain language."
    )

    proj_options_c = {f"{p['project_id']} — {p['project_name']}": p for p in projects}
    selected_label_c = st.selectbox(
        "Project to compare", list(proj_options_c.keys()), key="compare_proj_select"
    )
    project_c = proj_options_c[selected_label_c]

    compare_cache_key  = f"compare_{project_c['project_id']}"
    insight_cache_key  = f"insight_{project_c['project_id']}"

    run_compare = st.button("Run all 5 methodologies", type="primary")

    if run_compare:
        client = OpenAI(api_key=st.session_state["api_key"])

        # Embed the project once — reused across all methodology runs
        with st.spinner("Embedding project…"):
            proj_embed        = get_embedding(client, _project_requirement_text(project_c))
            proj_stretch_embed = get_embedding(client, _project_stretch_text(project_c))

        all_method_results = {}
        bar = st.progress(0, text="Scoring…")
        for i, mid in enumerate(METHODOLOGIES):
            bar.progress(i / len(METHODOLOGIES), text=f"Running {METHODOLOGIES[mid]['name']}…")
            all_method_results[mid] = rank_employees_no_rationale(
                project_c, employees,
                st.session_state["current_embeds"],
                st.session_state["target_embeds"],
                proj_embed, proj_stretch_embed,
                methodology_id=mid,
            )
        bar.progress(1.0, text="Scoring complete.")

        with st.spinner("Generating AI insight…"):
            insight = generate_comparison_insight(client, project_c, all_method_results)

        st.session_state[compare_cache_key] = all_method_results
        st.session_state[insight_cache_key] = insight

    if compare_cache_key in st.session_state:
        all_method_results = st.session_state[compare_cache_key]
        insight            = st.session_state.get(insight_cache_key, "")

        st.markdown(f"#### Results for **{project_c['project_name']}**")

        # ── Ranking comparison table ───────────────────────────────────────────
        # Collect all unique employees that appear in any top-3
        all_names: dict[str, dict] = {}
        for mid, matches in all_method_results.items():
            for rank, m in enumerate(matches, 1):
                if m["name"] not in all_names:
                    all_names[m["name"]] = {"role": m["role"]}

        # Build rows
        table_rows = []
        for name, info in all_names.items():
            row = {"Employee": name, "Role": info["role"]}
            appearances = 0
            for mid, matches in all_method_results.items():
                m_name = METHODOLOGIES[mid]["name"]
                rank_found = next(
                    (r + 1 for r, m in enumerate(matches) if m["name"] == name), None
                )
                row[m_name] = MEDALS[rank_found - 1] if rank_found else "—"
                if rank_found:
                    appearances += 1
            row["Consensus"] = f"{appearances}/{len(METHODOLOGIES)}"
            row["_appearances"] = appearances
            table_rows.append(row)

        table_rows.sort(key=lambda r: r["_appearances"], reverse=True)
        for r in table_rows:
            del r["_appearances"]

        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── Per-methodology score breakdown ───────────────────────────────────
        st.markdown("#### Score breakdown by methodology")
        method_cols = st.columns(len(METHODOLOGIES))
        for col, (mid, matches) in zip(method_cols, all_method_results.items()):
            m_info = METHODOLOGIES[mid]
            color  = METHOD_COLORS[mid]
            with col:
                st.markdown(
                    f"<div style='border-top:3px solid {color};padding-top:8px'>"
                    f"<b>{m_info['emoji']} {m_info['name']}</b></div>",
                    unsafe_allow_html=True,
                )
                for rank, m in enumerate(matches, 1):
                    s = m["scores"]
                    st.markdown(
                        f"**{MEDALS[rank-1]} {m['name']}**  \n"
                        f"<span style='font-size:0.8rem;color:#6b7280'>"
                        f"Fit {s['fit_score']:.0%} · "
                        f"Stretch {s['stretch_score']:.0%} · "
                        f"**{s['composite_score']:.0%}**</span>",
                        unsafe_allow_html=True,
                    )

        # ── Consensus summary chips ────────────────────────────────────────────
        st.markdown("#### Consensus at a glance")
        consensus_counts: dict[str, int] = {}
        for mid, matches in all_method_results.items():
            for m in matches:
                consensus_counts[m["name"]] = consensus_counts.get(m["name"], 0) + 1

        unanimous = [n for n, c in consensus_counts.items() if c == len(METHODOLOGIES)]
        majority  = [n for n, c in consensus_counts.items() if 2 < c < len(METHODOLOGIES)]
        minority  = [n for n, c in consensus_counts.items() if c <= 2]

        chip_col1, chip_col2, chip_col3 = st.columns(3)
        with chip_col1:
            st.markdown("**All 5 agree**")
            if unanimous:
                for n in unanimous:
                    st.success(n)
            else:
                st.caption("No unanimous picks")
        with chip_col2:
            st.markdown("**Majority (3-4)**")
            if majority:
                for n in majority:
                    st.info(n)
            else:
                st.caption("None")
        with chip_col3:
            st.markdown("**Minority (1-2)**")
            if minority:
                for n in minority:
                    st.warning(n)
            else:
                st.caption("None")

        # ── AI insight ────────────────────────────────────────────────────────
        st.markdown("#### AI Insight — why the methods diverged")
        st.markdown(insight)

        # Drill-down: run full match (with rationales) for any methodology
        st.divider()
        st.markdown("#### Drill into a methodology")
        st.caption("Get the full Fit & Stretch rationales for any methodology's top-3.")
        drill_label = st.selectbox(
            "Choose methodology",
            [f"{m['emoji']} {m['name']}" for m in METHODOLOGIES.values()],
            key="drill_select",
        )
        drill_id = list(METHODOLOGIES.keys())[
            [f"{m['emoji']} {m['name']}" for m in METHODOLOGIES.values()].index(drill_label)
        ]
        drill_cache = f"drill_{project_c['project_id']}_{drill_id}"

        if st.button("Get full rationales", key="drill_btn"):
            client = OpenAI(api_key=st.session_state["api_key"])
            with st.spinner(f"Generating rationales for {METHODOLOGIES[drill_id]['name']}…"):
                drill_matches = rank_employees_for_project(
                    client, project_c, employees,
                    st.session_state["current_embeds"],
                    st.session_state["target_embeds"],
                    top_n=3,
                    methodology_id=drill_id,
                )
            st.session_state[drill_cache] = drill_matches

        if drill_cache in st.session_state:
            render_match_cards(st.session_state[drill_cache], project_c)


# ── Tab 3: Batch run ───────────────────────────────────────────────────────────

with tab_batch:
    st.markdown("### Batch run across all projects")

    batch_method_label = st.radio(
        "Methodology",
        [f"{m['emoji']}  {m['name']}" for m in METHODOLOGIES.values()],
        horizontal=True,
        key="batch_method",
    )
    batch_method_id = list(METHODOLOGIES.keys())[
        [f"{m['emoji']}  {m['name']}" for m in METHODOLOGIES.values()].index(batch_method_label)
    ]
    st.caption(f"*{METHODOLOGIES[batch_method_id]['description']}*")
    st.markdown(
        f"Generates top-3 matches for all {len(projects)} projects using the selected "
        f"methodology. Approx. **{len(projects) * 5} API calls** — 3–5 minutes."
    )

    if st.button("Run batch", type="primary"):
        client = OpenAI(api_key=st.session_state["api_key"])
        all_results = []
        bar = st.progress(0, text="Starting…")
        for i, proj in enumerate(projects):
            bar.progress(i / len(projects), text=f"Matching: {proj['project_name']}")
            proj_matches = rank_employees_for_project(
                client, proj, employees,
                st.session_state["current_embeds"],
                st.session_state["target_embeds"],
                top_n=3,
                methodology_id=batch_method_id,
            )
            all_results.append({
                "project_id":   proj["project_id"],
                "project_name": proj["project_name"],
                "methodology":  batch_method_id,
                "matches":      proj_matches,
            })
        bar.progress(1.0, text="Complete!")
        st.session_state["batch_results"] = all_results

    if "batch_results" in st.session_state:
        all_results = st.session_state["batch_results"]
        used_method = all_results[0].get("methodology", "hybrid")
        st.success(
            f"{len(all_results)} projects matched using "
            f"{METHODOLOGIES[used_method]['emoji']} {METHODOLOGIES[used_method]['name']}."
        )
        rows = []
        for r in all_results:
            for rank, m in enumerate(r["matches"], 1):
                rows.append({
                    "Project":   r["project_name"],
                    "Rank":      rank,
                    "Employee":  m["name"],
                    "Role":      m["role"],
                    "Fit":       f"{m['scores']['fit_score']:.0%}",
                    "Stretch":   f"{m['scores']['stretch_score']:.0%}",
                    "Composite": f"{m['scores']['composite_score']:.0%}",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️  Download full results (JSON)",
            json.dumps(all_results, indent=2),
            file_name=f"flowmatch_batch_{used_method}.json",
            mime="application/json",
        )


# ── Tab 4: Data preview ────────────────────────────────────────────────────────

with tab_data:
    st.markdown("### Employees")
    st.dataframe(
        employees_df[["employee_id", "name", "role", "current_skills",
                       "target_skills", "available_hours_per_week", "availability_end_date"]],
        use_container_width=True, hide_index=True,
    )
    st.markdown("### Projects")
    st.dataframe(
        projects_df[["project_id", "project_name", "required_skills",
                      "desired_skills", "hours_per_week", "duration_weeks", "industry_context"]],
        use_container_width=True, hide_index=True,
    )
