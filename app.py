import streamlit as st
import pandas as pd
import sys, os, json, io
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from flowmatch import build_employee_embeddings, rank_employees_for_project

st.set_page_config(
    page_title="FlowMatch",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ FlowMatch")
    st.caption("Internal Capacity & Growth Marketplace")
    st.divider()

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-…",
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
    st.info("Upload your employee and project files in the sidebar, or toggle off 'Upload my own files' to use the repo data.")
    st.stop()

employees = employees_df.to_dict(orient="records")
projects  = projects_df.to_dict(orient="records")

# Stats strip
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", len(employees))
c2.metric("Open Projects", len(projects))
c3.metric("Possible Pairs", len(employees) * len(projects))
embeds_ready = "current_embeds" in st.session_state
c4.metric("Engine Status", "Ready ✅" if embeds_ready else "Needs setup ⚙️")

st.divider()


# ── Embed step ─────────────────────────────────────────────────────────────────

if not embeds_ready:
    st.markdown("### Step 1 — Build employee embeddings")
    st.markdown(
        "This calls the OpenAI embedding API once for all employees and caches the "
        "results in memory. You only need to do this once per session."
    )
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    if st.button("Build embeddings", type="primary", use_container_width=True):
        client = OpenAI(api_key=api_key)
        with st.spinner(f"Embedding {len(employees)} employees… this takes about 30 seconds"):
            cur, tgt = build_employee_embeddings(client, employees)
        st.session_state["current_embeds"] = cur
        st.session_state["target_embeds"]  = tgt
        st.session_state["api_key"]        = api_key
        st.success("Done — embeddings cached for this session.")
        st.rerun()
    st.stop()

# Allow re-embedding
with st.expander("⚙️  Re-embed / reset"):
    st.caption("Use this if you've changed the data files or want to use a different API key.")
    if st.button("Clear embeddings and restart"):
        for k in ("current_embeds", "target_embeds", "api_key", "last_matches", "last_project_id", "batch_results"):
            st.session_state.pop(k, None)
        st.rerun()

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_match, tab_batch, tab_data = st.tabs(["🎯  Match a Project", "📊  Batch All Projects", "🗂  View Data"])


# ── Tab 1: Single match ────────────────────────────────────────────────────────

with tab_match:
    st.markdown("### Select a project to find the best employee matches")

    proj_options = {f"{p['project_id']} — {p['project_name']}": p for p in projects}
    selected_label = st.selectbox("Project", list(proj_options.keys()), label_visibility="collapsed")
    project = proj_options[selected_label]

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
    find_clicked = run_col.button("Find top 3 matches", type="primary", use_container_width=True)

    if find_clicked:
        client = OpenAI(api_key=st.session_state["api_key"])
        with st.spinner("Scoring all employees and generating rationales…"):
            matches = rank_employees_for_project(
                client, project, employees,
                st.session_state["current_embeds"],
                st.session_state["target_embeds"],
                top_n=3,
            )
        st.session_state["last_matches"]    = matches
        st.session_state["last_project_id"] = project["project_id"]

    if (
        "last_matches" in st.session_state
        and st.session_state.get("last_project_id") == project["project_id"]
    ):
        matches = st.session_state["last_matches"]
        st.markdown(f"#### Top 3 matches for **{project['project_name']}**")

        medals = ["🥇", "🥈", "🥉"]
        for i, m in enumerate(matches):
            s = m["scores"]
            cap_warn = "  ⚠️ *Capacity may be tight*" if s.get("capacity_warning") else ""

            with st.container(border=True):
                left, right = st.columns([3, 2])

                with left:
                    st.markdown(f"### {medals[i]} {m['name']}")
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
            "⬇️  Download results (JSON)",
            result_json,
            file_name=f"flowmatch_{project['project_id']}.json",
            mime="application/json",
        )


# ── Tab 2: Batch ───────────────────────────────────────────────────────────────

with tab_batch:
    st.markdown("### Run the matching engine across all projects")
    st.markdown(
        f"Generates top-3 matches for every project in the dataset. "
        f"Approx. **{len(projects) * 5} API calls** — expect 3–5 minutes."
    )

    if st.button("Run batch match", type="primary"):
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
            )
            all_results.append({
                "project_id":   proj["project_id"],
                "project_name": proj["project_name"],
                "matches":      proj_matches,
            })

        bar.progress(1.0, text="Complete!")
        st.session_state["batch_results"] = all_results

    if "batch_results" in st.session_state:
        all_results = st.session_state["batch_results"]
        st.success(f"Matched {len(all_results)} projects.")

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
            file_name="flowmatch_batch_results.json",
            mime="application/json",
        )


# ── Tab 3: Data preview ────────────────────────────────────────────────────────

with tab_data:
    st.markdown("### Employees")
    st.dataframe(
        employees_df[["employee_id", "name", "role", "current_skills",
                       "target_skills", "available_hours_per_week", "availability_end_date"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Projects")
    st.dataframe(
        projects_df[["project_id", "project_name", "required_skills",
                      "desired_skills", "hours_per_week", "duration_weeks", "industry_context"]],
        use_container_width=True,
        hide_index=True,
    )
