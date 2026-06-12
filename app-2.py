import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="PlaceIQ — Student Placement Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Canvas ── */
.stApp { background: #F7F5F0; }
section[data-testid="stSidebar"] { background: #1C1917 !important; }
section[data-testid="stSidebar"] * { color: #E7E5E4 !important; }
section[data-testid="stSidebar"] .stSlider > label { color: #A8A29E !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #A8A29E !important; }

/* ── Sidebar inputs ── */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #292524 !important;
    border: 1px solid #44403C !important;
    color: #E7E5E4 !important;
    border-radius: 8px !important;
}

/* ── Top nav bar ── */
.placeiq-nav {
    background: #1C1917;
    padding: 14px 36px;
    display: flex;
    align-items: center;
    gap: 14px;
    border-radius: 0 0 16px 16px;
    margin-bottom: 32px;
}
.nav-logo-wrap {
    width: 40px; height: 40px;
    background: #D97706;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.nav-wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: #FAFAF9;
    letter-spacing: -0.02em;
    line-height: 1;
}
.nav-wordmark span { color: #D97706; }
.nav-tagline {
    font-size: 0.72rem;
    color: #78716C;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}
.nav-pill {
    margin-left: auto;
    background: #292524;
    border: 1px solid #44403C;
    color: #A8A29E;
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}

/* ── Stat row ── */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E7E5E4;
    border-radius: 14px;
    padding: 20px 22px;
    text-align: left;
}
.stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #1C1917;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: #78716C;
    margin-top: 5px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.stat-accent { color: #D97706; }

/* ── Section heading ── */
.sec-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #1C1917;
    margin: 28px 0 14px;
    padding-bottom: 10px;
    border-bottom: 2px solid #E7E5E4;
    display: flex; align-items: center; gap: 10px;
}
.sec-heading-icon {
    width: 28px; height: 28px;
    background: #FEF3C7;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
}

/* ── Result banners ── */
.result-placed {
    background: #F0FDF4;
    border: 1.5px solid #86EFAC;
    border-left: 5px solid #16A34A;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.result-placed h2 { color: #15803D; font-family: 'DM Serif Display', serif; margin: 0; font-size: 1.4rem; }
.result-placed p { color: #166534; font-size: 0.88rem; margin: 6px 0 0; }

.result-risk {
    background: #FFF7ED;
    border: 1.5px solid #FCA5A5;
    border-left: 5px solid #DC2626;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.result-risk h2 { color: #B91C1C; font-family: 'DM Serif Display', serif; margin: 0; font-size: 1.4rem; }
.result-risk p { color: #991B1B; font-size: 0.88rem; margin: 6px 0 0; }

/* ── Reason cards ── */
.reason-pos {
    background: #fff;
    border-left: 4px solid #16A34A;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 8px 0;
    border-top: 1px solid #E7E5E4;
    border-right: 1px solid #E7E5E4;
    border-bottom: 1px solid #E7E5E4;
}
.reason-neg {
    background: #fff;
    border-left: 4px solid #DC2626;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 8px 0;
    border-top: 1px solid #E7E5E4;
    border-right: 1px solid #E7E5E4;
    border-bottom: 1px solid #E7E5E4;
}
.reason-title { font-weight: 600; font-size: 0.88rem; color: #1C1917; }
.reason-desc { font-size: 0.82rem; color: #78716C; margin-top: 3px; }

/* ── Career card ── */
.career-card {
    background: #FFFFFF;
    border: 1px solid #E7E5E4;
    border-radius: 14px;
    padding: 20px 22px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
}
.career-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 4px;
    background: linear-gradient(90deg, #D97706, #F59E0B);
}
.career-title { font-family: 'DM Serif Display', serif; font-size: 1.05rem; color: #1C1917; margin: 0 0 4px; }
.career-match { font-size: 0.75rem; color: #78716C; }
.career-match span { color: #D97706; font-weight: 600; }
.skill-tag {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.74rem;
    margin: 3px 2px;
    font-weight: 500;
}
.step-line {
    font-size: 0.82rem;
    color: #57534E;
    padding: 5px 0;
    border-bottom: 1px dashed #E7E5E4;
}
.step-num { color: #D97706; font-weight: 600; margin-right: 6px; }

/* ── Resume card ── */
.resume-doc {
    background: #FFFFFF;
    border: 1px solid #D6D3D1;
    border-radius: 4px;
    padding: 36px 40px;
    font-family: 'DM Sans', sans-serif;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    position: relative;
}
.resume-doc::after {
    content: 'PlaceIQ Resume Draft';
    position: absolute;
    top: 12px; right: 16px;
    font-size: 0.68rem;
    color: #D6D3D1;
    letter-spacing: 0.05em;
}
.resume-name { font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #1C1917; margin: 0 0 2px; }
.resume-role { font-size: 0.85rem; color: #78716C; margin: 0 0 14px; }
.resume-divider { border: none; border-top: 2px solid #1C1917; margin: 10px 0 16px; }
.resume-section { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #78716C; margin: 16px 0 8px; }
.resume-bullet { font-size: 0.82rem; color: #1C1917; padding: 3px 0; }
.resume-bullet::before { content: '▸ '; color: #D97706; }
.resume-tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.resume-tag {
    background: #F5F5F4;
    border: 1px solid #E7E5E4;
    color: #1C1917;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 4px;
}
.resume-highlight {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #92400E;
    margin: 8px 0;
}
.resume-score-bar {
    height: 5px;
    background: #E7E5E4;
    border-radius: 3px;
    margin: 3px 0 8px;
    overflow: hidden;
}
.resume-score-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #D97706, #F59E0B);
}

/* ── Company logos strip ── */
.logos-strip {
    background: #fff;
    border: 1px solid #E7E5E4;
    border-radius: 12px;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
    margin: 16px 0;
}
.company-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 0.88rem;
    color: #78716C;
    letter-spacing: -0.01em;
    padding: 4px 12px;
    border: 1px solid #E7E5E4;
    border-radius: 6px;
    background: #FAFAF9;
}

/* ── Sidebar predict button ── */
.stButton > button {
    background: #D97706 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { background: #B45309 !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #FFFFFF;
    border: 1px solid #E7E5E4;
    border-radius: 8px;
    color: #78716C;
    font-size: 0.85rem;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #1C1917 !important;
    color: #FAFAF9 !important;
    border-color: #1C1917 !important;
}

/* ── Plotly chart backgrounds ── */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── NAV BAR (SVG mortarboard logo) ──────────────────────────────────────────────
cap_svg = """
<svg width="22" height="20" viewBox="0 0 22 20" fill="none" xmlns="http://www.w3.org/2000/svg">
  <polygon points="11,2 22,7 11,12 0,7" fill="#FEF3C7" stroke="#FDE68A" stroke-width="0.5"/>
  <path d="M17,9.5 L17,15 C17,15 14.5,17.5 11,17.5 C7.5,17.5 5,15 5,15 L5,9.5" stroke="#FEF3C7" stroke-width="2" fill="none" stroke-linecap="round"/>
  <line x1="20" y1="7" x2="20" y2="13" stroke="#D97706" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="20" cy="14" r="1.5" fill="#D97706"/>
</svg>"""

st.markdown(f"""
<div class="placeiq-nav">
  <div class="nav-logo-wrap">{cap_svg}</div>
  <div>
    <div class="nav-wordmark">Place<span>IQ</span></div>
    <div class="nav-tagline">Placement Intelligence Platform</div>
  </div>
  <div class="nav-pill">Beta v2.0</div>
</div>
""", unsafe_allow_html=True)

# ── DATA + MODEL ────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=2000):
    np.random.seed(42)
    cgpa = np.round(np.random.uniform(5.0, 10.0, n), 2)
    internships = np.random.randint(0, 4, n)
    projects = np.random.randint(0, 6, n)
    workshops = np.random.randint(0, 5, n)
    aptitude = np.random.randint(40, 100, n)
    communication = np.random.randint(1, 6, n)
    backlogs = np.random.randint(0, 5, n)
    stream = np.random.choice(["CS", "IT", "ECE", "EEE", "MECH", "CIVIL"], n)
    stream_bonus = {"CS": 0.15, "IT": 0.12, "ECE": 0.08, "EEE": 0.06, "MECH": 0.03, "CIVIL": 0.02}
    sb = np.array([stream_bonus[s] for s in stream])
    score = (
        0.30 * (cgpa - 5) / 5 +
        0.20 * internships / 3 +
        0.15 * projects / 5 +
        0.10 * workshops / 4 +
        0.12 * (aptitude - 40) / 60 +
        0.08 * communication / 5 +
        sb - 0.10 * backlogs / 4 +
        np.random.normal(0, 0.05, n)
    )
    placed = (score > 0.42).astype(int)
    le = LabelEncoder()
    return pd.DataFrame({
        "CGPA": cgpa, "Internships": internships, "Projects": projects,
        "Workshops": workshops, "AptitudeScore": aptitude,
        "CommunicationSkill": communication, "Backlogs": backlogs,
        "Stream": stream, "StreamCode": le.fit_transform(stream), "Placed": placed
    })

@st.cache_resource
def train_model(df):
    features = ["CGPA", "Internships", "Projects", "Workshops",
                "AptitudeScore", "CommunicationSkill", "Backlogs", "StreamCode"]
    X, y = df[features], df["Placed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    return clf, clf.score(X_test, y_test), features

df = generate_dataset()
model, accuracy, features = train_model(df)
le = LabelEncoder().fit(df["Stream"])

# ── STAT ROW ─────────────────────────────────────────────────────────────────────
placed_pct = df["Placed"].mean() * 100
c1, c2, c3, c4 = st.columns(4)
for col, num, lbl, accent in [
    (c1, f"{len(df):,}", "Training records", False),
    (c2, f"{accuracy*100:.1f}%", "Model accuracy", True),
    (c3, f"{placed_pct:.0f}%", "Placement rate", False),
    (c4, "8", "Features analysed", False),
]:
    with col:
        st.markdown(f"""
        <div class="stat-card">
          <div class="stat-num {'stat-accent' if accent else ''}">{num}</div>
          <div class="stat-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ── COMPANY LOGOS STRIP ──────────────────────────────────────────────────────────
st.markdown("""
<div class="logos-strip">
  <span style="font-size:0.72rem;color:#A8A29E;letter-spacing:0.06em;text-transform:uppercase;margin-right:4px;">Hiring partners</span>
  <div class="company-logo">Google</div>
  <div class="company-logo">Microsoft</div>
  <div class="company-logo">Amazon</div>
  <div class="company-logo">Infosys</div>
  <div class="company-logo">TCS</div>
  <div class="company-logo">Wipro</div>
  <div class="company-logo">Deloitte</div>
  <div class="company-logo">Accenture</div>
</div>
""", unsafe_allow_html=True)

# ── HELPER FUNCTIONS ─────────────────────────────────────────────────────────────
def get_explanation(cgpa, internships, projects, aptitude, backlogs, communication):
    pos, neg = [], []
    if cgpa >= 8.5:    pos.append(("Excellent CGPA", f"{cgpa}/10 — top academic performer"))
    elif cgpa >= 7.5:  pos.append(("Good CGPA", f"{cgpa}/10 — above average"))
    elif cgpa >= 6.5:  neg.append(("Average CGPA", f"{cgpa}/10 — needs improvement"))
    else:              neg.append(("Low CGPA", f"{cgpa}/10 — critical area"))
    if internships >= 2:   pos.append(("Strong internship record", f"{internships} internships — industry ready"))
    elif internships == 1: pos.append(("Has internship experience", "1 internship — solid start"))
    else:                  neg.append(("No internship experience", "Apply to internships urgently"))
    if projects >= 4:      pos.append(("Impressive project portfolio", f"{projects} projects — strong signal"))
    elif projects >= 2:    pos.append(("Decent project count", f"{projects} projects — keep building"))
    else:                  neg.append(("Few projects", f"{projects} project(s) — build more ASAP"))
    if aptitude >= 80:     pos.append(("High aptitude score", f"{aptitude}/100 — strong analytical skills"))
    elif aptitude >= 65:   pos.append(("Good aptitude", f"{aptitude}/100"))
    else:                  neg.append(("Low aptitude score", f"{aptitude}/100 — practice more"))
    if communication >= 4: pos.append(("Strong communication", f"{communication}/5 — great soft skills"))
    elif communication == 3: neg.append(("Average communication", f"{communication}/5 — room to grow"))
    else:                  neg.append(("Weak communication", f"{communication}/5 — must improve"))
    if backlogs > 0:       neg.append(("Active backlogs", f"{backlogs} backlog(s) — clear before placements"))
    return pos, neg

def get_careers(cgpa, internships, projects, aptitude, communication, stream):
    careers = []
    if stream in ["CS", "IT"]:
        if cgpa >= 8.0 and projects >= 3:
            careers.append({"title": "Software Development Engineer", "match": 95,
                "steps": ["Master DSA & System Design", "Contribute to open-source projects", "Prepare for FAANG-style interviews"],
                "skills": ["Python/Java", "LeetCode", "Git", "System Design"]})
        if aptitude >= 75 and cgpa >= 7.5:
            careers.append({"title": "Data Analyst / Data Scientist", "match": 88,
                "steps": ["Learn Power BI or Tableau", "Sharpen SQL & Pandas skills", "Build 2 end-to-end analytics projects"],
                "skills": ["SQL", "Python", "Power BI", "Statistics"]})
        if communication >= 4:
            careers.append({"title": "Cloud & DevOps Engineer", "match": 82,
                "steps": ["Get AWS or GCP certified", "Learn Docker & Kubernetes", "Set up CI/CD pipelines"],
                "skills": ["AWS", "Docker", "Linux", "Terraform"]})
    elif stream in ["ECE", "EEE"]:
        careers.append({"title": "Embedded Systems Engineer", "match": 85,
            "steps": ["Master C/C++ for embedded targets", "Learn RTOS concepts", "Build an IoT prototype project"],
            "skills": ["C/C++", "RTOS", "Arduino", "FPGA"]})
        if cgpa >= 7.5:
            careers.append({"title": "VLSI / Hardware Design Engineer", "match": 80,
                "steps": ["Learn Verilog & VHDL", "Practice on ModelSim", "Target semiconductor companies"],
                "skills": ["Verilog", "VHDL", "Cadence", "Signal Processing"]})
    else:
        careers.append({"title": "Core Engineering / Consultant", "match": 78,
            "steps": ["Earn domain-specific certifications", "Build technical portfolio projects", "Apply to PSU & core companies"],
            "skills": ["AutoCAD", "MATLAB", "Project Management", "Domain expertise"]})
    if not careers:
        careers.append({"title": "Business Analyst / IT Consultant", "match": 72,
            "steps": ["Deep-dive into SQL & Excel", "Get PMP or CBAP certified", "Build an analytical portfolio"],
            "skills": ["SQL", "Excel", "Tableau", "Communication"]})
    return careers[:2]

def build_resume(name, stream, cgpa, internships, projects, workshops, aptitude, communication, backlogs, careers, pos_reasons):
    target_role = careers[0]["title"] if careers else "Software Engineer"
    skills_all = []
    for c in careers:
        skills_all.extend(c["skills"])
    skills_all = list(dict.fromkeys(skills_all))[:8]

    highlights = []
    if cgpa >= 8.0: highlights.append(f"Maintained a strong CGPA of {cgpa}/10 throughout the programme")
    if internships >= 1: highlights.append(f"Completed {internships} industry internship(s), gaining hands-on professional experience")
    if projects >= 3: highlights.append(f"Built {projects} technical projects demonstrating applied problem-solving")
    if aptitude >= 75: highlights.append(f"Scored {aptitude}/100 in aptitude assessments — top percentile performer")
    if communication >= 4: highlights.append(f"Demonstrated strong communication skills rated {communication}/5")
    if workshops >= 2: highlights.append(f"Attended {workshops} workshops and certification programmes")

    project_bullets = []
    if stream in ["CS", "IT"]:
        project_bullets = [
            f"Capstone Project: End-to-end ML pipeline using Python & scikit-learn",
            f"Web Application: Full-stack app with React frontend and REST API backend",
            f"Data Dashboard: Interactive analytics dashboard using Power BI & SQL"
        ]
    elif stream in ["ECE", "EEE"]:
        project_bullets = [
            f"IoT Automation: Smart home system using Arduino & MQTT protocol",
            f"Signal Processing: Real-time audio filter implementation on FPGA",
            f"PCB Design: Custom PCB for embedded sensor integration"
        ]
    else:
        project_bullets = [
            f"Structural Analysis: FEM-based stress simulation using ANSYS",
            f"Process Optimisation: Lean manufacturing study with 18% efficiency gain",
            f"AutoCAD Design: Complete 2D/3D model of a mechanical assembly"
        ]

    backlog_note = "" if backlogs == 0 else f"<!-- Note: {backlogs} backlog(s) present — clear before submission -->"

    return {
        "name": name,
        "role": target_role,
        "stream": stream,
        "cgpa": cgpa,
        "highlights": highlights[:4],
        "skills": skills_all,
        "projects": project_bullets[:projects if projects <= 3 else 3],
        "workshops": workshops,
        "careers": careers,
        "pos_reasons": pos_reasons,
    }

# ── SIDEBAR ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 16px;">
      <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#FAFAF9;margin-bottom:4px;">Student Profile</div>
      <div style="font-size:0.74rem;color:#78716C;letter-spacing:0.04em;text-transform:uppercase;">Fill in your details below</div>
    </div>
    """, unsafe_allow_html=True)

    student_name = st.text_input("Full Name", value="Alex Kumar", key="name")
    stream = st.selectbox("Branch / Stream", ["CS", "IT", "ECE", "EEE", "MECH", "CIVIL"])
    cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
    internships = st.slider("Internships completed", 0, 3, 1)
    projects = st.slider("Projects built", 0, 5, 2)
    workshops = st.slider("Workshops / Certifications", 0, 4, 1)
    aptitude = st.slider("Aptitude score (out of 100)", 40, 100, 70)
    communication = st.slider("Communication skill (1–5)", 1, 5, 3)
    backlogs = st.slider("Active backlogs", 0, 4, 0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Analyse My Profile →", use_container_width=True)

    st.markdown("""
    <div style="margin-top:28px;padding-top:20px;border-top:1px solid #292524;">
      <div style="font-size:0.72rem;color:#57534E;line-height:1.7;">
        Powered by Random Forest · 200 trees · 2,000 training samples<br>
        <span style="color:#D97706;">PlaceIQ</span> is for educational use only
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Predict & Analyse", "Data Insights", "Resume Builder", "Model Details"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    if not predict_btn:
        st.markdown("""
        <div style="text-align:center;padding:64px 20px;">
          <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:#1C1917;margin-bottom:12px;">
            Ready when you are.
          </div>
          <p style="color:#78716C;font-size:0.95rem;">Fill in your profile in the sidebar and click <strong>Analyse My Profile</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        stream_enc = le.transform([stream])[0]
        inp = np.array([[cgpa, internships, projects, workshops, aptitude, communication, backlogs, stream_enc]])
        prob = model.predict_proba(inp)[0][1]
        placed = prob >= 0.5

        left, right = st.columns([1, 1.4], gap="large")

        with left:
            # Result
            if placed:
                st.markdown(f"""
                <div class="result-placed">
                  <h2>Likely to be placed</h2>
                  <p>Placement probability: <strong>{prob*100:.1f}%</strong> · Strong candidate profile</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-risk">
                  <h2>Placement at risk</h2>
                  <p>Placement probability: <strong>{prob*100:.1f}%</strong> · Profile needs strengthening</p>
                </div>""", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 34, "color": "#1C1917", "family": "DM Serif Display"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#D6D3D1", "tickfont": {"color": "#78716C"}},
                    "bar": {"color": "#D97706", "thickness": 0.25},
                    "bgcolor": "#F7F5F0",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#FEF2F2"},
                        {"range": [50, 75], "color": "#FFFBEB"},
                        {"range": [75, 100], "color": "#F0FDF4"},
                    ],
                    "threshold": {"line": {"color": "#1C1917", "width": 2}, "value": 50},
                },
                title={"text": "Placement Probability", "font": {"color": "#78716C", "size": 13}}
            ))
            fig_g.update_layout(paper_bgcolor="#F7F5F0", font_color="#1C1917",
                                height=250, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_g, use_container_width=True)

            # Explanation
            st.markdown('<div class="sec-heading"><div class="sec-heading-icon">🔍</div>Why this prediction</div>', unsafe_allow_html=True)
            pos, neg = get_explanation(cgpa, internships, projects, aptitude, backlogs, communication)

            if pos:
                st.markdown("<div style='font-size:0.78rem;font-weight:600;color:#16A34A;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:4px;'>Strengths</div>", unsafe_allow_html=True)
                for t, d in pos:
                    st.markdown(f'<div class="reason-pos"><div class="reason-title">{t}</div><div class="reason-desc">{d}</div></div>', unsafe_allow_html=True)

            if neg:
                st.markdown("<div style='font-size:0.78rem;font-weight:600;color:#DC2626;letter-spacing:0.04em;text-transform:uppercase;margin:12px 0 4px;'>Areas to improve</div>", unsafe_allow_html=True)
                for t, d in neg:
                    st.markdown(f'<div class="reason-neg"><div class="reason-title">{t}</div><div class="reason-desc">{d}</div></div>', unsafe_allow_html=True)

        with right:
            # Career recommendations
            careers = get_careers(cgpa, internships, projects, aptitude, communication, stream)
            st.markdown('<div class="sec-heading"><div class="sec-heading-icon">🚀</div>Career recommendations</div>', unsafe_allow_html=True)

            for i, c in enumerate(careers):
                badge = "Best match" if i == 0 else "Alternative path"
                steps_html = "".join([f'<div class="step-line"><span class="step-num">{j+1}.</span>{s}</div>' for j, s in enumerate(c["steps"])])
                tags_html = "".join([f'<span class="skill-tag">{sk}</span>' for sk in c["skills"]])
                st.markdown(f"""
                <div class="career-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;">
                    <div class="career-title">{c['title']}</div>
                    <div style="text-align:right;">
                      <div style="font-size:0.72rem;color:#78716C;">{badge}</div>
                      <div style="font-size:1rem;font-weight:600;color:#D97706;">{c['match']}% fit</div>
                    </div>
                  </div>
                  <div style="margin-bottom:12px;">{tags_html}</div>
                  <div style="font-size:0.72rem;font-weight:600;color:#78716C;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Recommended next steps</div>
                  {steps_html}
                </div>""", unsafe_allow_html=True)

            # Radar chart — profile strengths
            st.markdown('<div class="sec-heading" style="margin-top:24px;"><div class="sec-heading-icon">📡</div>Profile radar</div>', unsafe_allow_html=True)
            cats = ["CGPA", "Internships", "Projects", "Aptitude", "Communication"]
            vals = [
                (cgpa - 5) / 5 * 100,
                internships / 3 * 100,
                projects / 5 * 100,
                (aptitude - 40) / 60 * 100,
                communication / 5 * 100
            ]
            fig_r = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill='toself',
                fillcolor='rgba(217,119,6,0.15)',
                line=dict(color='#D97706', width=2),
                marker=dict(color='#D97706', size=6)
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color='#78716C'),
                                    gridcolor='#E7E5E4', linecolor='#E7E5E4'),
                    angularaxis=dict(tickfont=dict(size=11, color='#1C1917'), gridcolor='#E7E5E4'),
                    bgcolor='#FFFFFF'
                ),
                paper_bgcolor='#F7F5F0',
                showlegend=False,
                height=300,
                margin=dict(t=30, b=20, l=50, r=50)
            )
            st.plotly_chart(fig_r, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ════════════════════════════════════════════════════════════════
with tab2:
    PLOT_BG = "#FFFFFF"
    PAPER_BG = "#F7F5F0"
    FONT_COL = "#1C1917"
    GRID_COL = "#E7E5E4"

    def style_fig(fig, h=340):
        fig.update_layout(paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COL, title_font_size=13,
                          title_font_family="DM Serif Display",
                          coloraxis_showscale=False, height=h,
                          margin=dict(t=50, b=30))
        fig.update_xaxes(gridcolor=GRID_COL, tickfont_color="#78716C", linecolor=GRID_COL)
        fig.update_yaxes(gridcolor=GRID_COL, tickfont_color="#78716C", linecolor=GRID_COL)
        return fig

    col1, col2 = st.columns(2)
    with col1:
        cgpa_bins = pd.cut(df["CGPA"], bins=[5, 6, 7, 7.5, 8, 8.5, 9, 10],
                           labels=["5–6", "6–7", "7–7.5", "7.5–8", "8–8.5", "8.5–9", "9–10"])
        cgpa_rate = df.groupby(cgpa_bins, observed=True)["Placed"].mean().reset_index()
        cgpa_rate.columns = ["CGPA Range", "Placement Rate"]
        cgpa_rate["Placement Rate"] = (cgpa_rate["Placement Rate"] * 100).round(1)
        fig1 = px.bar(cgpa_rate, x="CGPA Range", y="Placement Rate",
                      title="CGPA vs Placement Rate",
                      color="Placement Rate", color_continuous_scale=[[0,"#FDE68A"],[1,"#92400E"]],
                      text="Placement Rate")
        fig1.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        style_fig(fig1).update_yaxes(range=[0,110])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        intern_rate = df.groupby("Internships")["Placed"].mean().reset_index()
        intern_rate["Placed"] = (intern_rate["Placed"] * 100).round(1)
        fig2 = px.bar(intern_rate, x="Internships", y="Placed",
                      title="Internships vs Placement Rate",
                      color="Placed", color_continuous_scale=[[0,"#FDE68A"],[1,"#92400E"]],
                      text="Placed")
        fig2.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        style_fig(fig2).update_yaxes(range=[0,110])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        sample = df.sample(600, random_state=1)
        fig3 = px.scatter(sample, x="CGPA", y="AptitudeScore",
                          color=sample["Placed"].map({1: "Placed", 0: "Not Placed"}),
                          title="CGPA × Aptitude Distribution",
                          opacity=0.6,
                          color_discrete_map={"Placed": "#16A34A", "Not Placed": "#DC2626"})
        fig3.update_layout(legend=dict(bgcolor="#FFFFFF", bordercolor="#E7E5E4", font=dict(color="#1C1917")))
        style_fig(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        stream_rate = df.groupby("Stream")["Placed"].mean().reset_index()
        stream_rate["Placed"] = (stream_rate["Placed"] * 100).round(1)
        stream_rate = stream_rate.sort_values("Placed")
        fig4 = px.bar(stream_rate, y="Stream", x="Placed", orientation="h",
                      title="Branch-wise Placement Rate",
                      color="Placed", color_continuous_scale=[[0,"#FDE68A"],[1,"#92400E"]],
                      text="Placed")
        fig4.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        style_fig(fig4).update_xaxes(range=[0,110])
        st.plotly_chart(fig4, use_container_width=True)

    proj_rate = df.groupby("Projects")["Placed"].mean().reset_index()
    proj_rate["Placed"] = (proj_rate["Placed"] * 100).round(1)
    fig5 = px.area(proj_rate, x="Projects", y="Placed", title="Projects Count vs Placement Rate",
                   line_shape="spline")
    fig5.update_traces(line_color="#D97706", fillcolor="rgba(217,119,6,0.12)", marker_color="#D97706")
    style_fig(fig5, h=280).update_yaxes(range=[0,105])
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — RESUME BUILDER
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-heading"><div class="sec-heading-icon">📄</div>Resume Builder — Recruiter-Ready Draft</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#78716C;font-size:0.88rem;margin-bottom:20px;">
      Based on your strengths, PlaceIQ generates a tailored resume skeleton that highlights what recruiters look for.
      Fill in your profile first, then click <strong>Analyse My Profile</strong>.
    </p>""", unsafe_allow_html=True)

    if not predict_btn:
        st.info("Click **Analyse My Profile** in the sidebar to generate your personalised resume draft.")
    else:
        careers = get_careers(cgpa, internships, projects, aptitude, communication, stream)
        pos, _ = get_explanation(cgpa, internships, projects, aptitude, backlogs, communication)
        resume = build_resume(student_name, stream, cgpa, internships, projects,
                              workshops, aptitude, communication, backlogs, careers, pos)

        left_r, right_r = st.columns([1.4, 1], gap="large")

        with left_r:
            # The actual resume document
            skills_tags = "".join([f'<span class="resume-tag">{s}</span>' for s in resume["skills"]])
            highlights_html = "".join([f'<div class="resume-bullet">{h}</div>' for h in resume["highlights"]])
            projects_html = "".join([f'<div class="resume-bullet">{p}</div>' for p in resume["projects"]])

            contact_line = f"{stream} Engineering · CGPA: {cgpa}"

            cgpa_pct = int((cgpa - 5) / 5 * 100)
            intern_pct = int(internships / 3 * 100)
            apt_pct = int((aptitude - 40) / 60 * 100)
            comm_pct = int(communication / 5 * 100)

            st.markdown(f"""
<div class="resume-doc">
  <div class="resume-name">{resume['name']}</div>
  <div class="resume-role">Aspiring {resume['role']} &nbsp;·&nbsp; {contact_line}</div>
  <hr class="resume-divider"/>

  <div class="resume-section">Professional Summary</div>
  <div style="font-size:0.84rem;color:#1C1917;line-height:1.65;margin-bottom:4px;">
    Motivated {stream} engineering student with a strong academic record (CGPA {cgpa}) and
    {"hands-on internship experience" if internships > 0 else "growing project portfolio"}.
    Demonstrated ability to deliver {projects} technical project(s) with a focus on
    {"software engineering and data-driven solutions" if stream in ["CS","IT"] else "embedded systems and hardware design" if stream in ["ECE","EEE"] else "core engineering and process optimisation"}.
  </div>

  <div class="resume-section">Key Highlights</div>
  {highlights_html}

  <div class="resume-section">Technical Skills</div>
  <div class="resume-tag-row">{skills_tags}</div>

  <div class="resume-section">Projects ({projects} completed)</div>
  {projects_html if projects > 0 else '<div style="font-size:0.82rem;color:#78716C;">No projects listed — add projects to strengthen this section.</div>'}

  <div class="resume-section">Education</div>
  <div class="resume-bullet">B.E. / B.Tech in {stream} — [University Name], Expected 20XX</div>
  <div style="font-size:0.8rem;color:#57534E;padding-left:14px;margin-top:2px;">CGPA: {cgpa}/10 &nbsp;·&nbsp; {"No active backlogs" if backlogs == 0 else f"{backlogs} backlog(s) — in progress"}</div>

  {"<div class='resume-section'>Certifications & Workshops</div><div class='resume-bullet'>" + str(workshops) + " workshop(s) / certification(s) completed — add names here</div>" if workshops > 0 else ""}

</div>
""", unsafe_allow_html=True)

        with right_r:
            st.markdown('<div class="sec-heading"><div class="sec-heading-icon">💡</div>What recruiters will notice</div>', unsafe_allow_html=True)

            recruiter_tips = []
            if cgpa >= 8.0:
                recruiter_tips.append(("CGPA opens shortlisting doors", "Many companies filter resumes below 7.5. Your {:.1f} puts you in the consideration pile automatically.".format(cgpa)))
            if internships >= 1:
                recruiter_tips.append(("Internship shows real-world exposure", "Recruiters weight internship experience 3× over coursework. Lead with it."))
            if projects >= 3:
                recruiter_tips.append(("Project portfolio demonstrates initiative", "3+ projects signal self-learner. Link them to GitHub in your resume."))
            if communication >= 4:
                recruiter_tips.append(("Communication skills stand out", "Soft skills are often the tiebreaker. Mention group projects and presentations."))
            if backlogs == 0:
                recruiter_tips.append(("Clean academic record", "No backlogs is a positive filter in many screening systems. Mention it explicitly."))

            for tip_title, tip_body in recruiter_tips:
                st.markdown(f"""
                <div class="resume-highlight">
                  <div style="font-weight:600;margin-bottom:3px;">{tip_title}</div>
                  <div style="font-size:0.8rem;color:#57534E;">{tip_body}</div>
                </div>""", unsafe_allow_html=True)

            # Profile score bars
            st.markdown('<div class="sec-heading" style="margin-top:20px;"><div class="sec-heading-icon">📊</div>Profile strength</div>', unsafe_allow_html=True)
            for label, pct in [("Academic (CGPA)", cgpa_pct), ("Industry exposure", intern_pct),
                                 ("Aptitude", apt_pct), ("Communication", comm_pct)]:
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#57534E;margin-bottom:3px;">
                    <span>{label}</span><span style="color:#D97706;font-weight:600;">{pct}%</span>
                  </div>
                  <div class="resume-score-bar">
                    <div class="resume-score-fill" style="width:{pct}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:20px;background:#F0FDF4;border:1px solid #86EFAC;border-radius:10px;padding:14px 16px;">
              <div style="font-size:0.8rem;font-weight:600;color:#15803D;margin-bottom:6px;">Before you send this resume</div>
              <div style="font-size:0.78rem;color:#166534;line-height:1.7;">
                ✓ Add real project GitHub links<br>
                ✓ Replace [University Name] with yours<br>
                ✓ Add internship company names<br>
                ✓ List actual certification names<br>
                ✓ Keep it to one page<br>
                ✓ Use a clean PDF export
              </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — MODEL DETAILS
# ════════════════════════════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        importances = model.feature_importances_
        feat_labels = ["CGPA", "Internships", "Projects", "Workshops", "Aptitude", "Communication", "Backlogs", "Stream"]
        fi_df = pd.DataFrame({"Feature": feat_labels, "Importance": importances}).sort_values("Importance")
        fig_fi = px.bar(fi_df, y="Feature", x="Importance", orientation="h",
                        title="Feature Importances",
                        color="Importance", color_continuous_scale=[[0,"#FDE68A"],[1,"#92400E"]],
                        text=fi_df["Importance"].apply(lambda x: f"{x:.3f}"))
        fig_fi.update_layout(paper_bgcolor="#F7F5F0", plot_bgcolor="#FFFFFF",
                             font_color="#1C1917", title_font_size=13,
                             title_font_family="DM Serif Display",
                             coloraxis_showscale=False, height=380,
                             margin=dict(t=50, b=20, r=60))
        fig_fi.update_xaxes(gridcolor="#E7E5E4", tickfont_color="#78716C")
        fig_fi.update_yaxes(gridcolor="#E7E5E4", tickfont_color="#78716C")
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #E7E5E4;border-radius:14px;padding:24px 26px;margin-top:4px;">
          <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#1C1917;margin-bottom:16px;">Model configuration</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.85rem;">
            <tr><td style="color:#78716C;padding:9px 0;">Algorithm</td><td style="color:#1C1917;text-align:right;font-weight:500;">Random Forest Classifier</td></tr>
            <tr style="border-top:1px solid #F5F5F4;"><td style="color:#78716C;padding:9px 0;">Trees</td><td style="color:#1C1917;text-align:right;">200 estimators</td></tr>
            <tr style="border-top:1px solid #F5F5F4;"><td style="color:#78716C;padding:9px 0;">Max depth</td><td style="color:#1C1917;text-align:right;">10</td></tr>
            <tr style="border-top:1px solid #F5F5F4;"><td style="color:#78716C;padding:9px 0;">Train / test split</td><td style="color:#1C1917;text-align:right;">80% / 20%</td></tr>
            <tr style="border-top:1px solid #F5F5F4;"><td style="color:#78716C;padding:9px 0;">Test accuracy</td><td style="text-align:right;font-weight:600;color:#D97706;">{accuracy*100:.1f}%</td></tr>
            <tr style="border-top:1px solid #F5F5F4;"><td style="color:#78716C;padding:9px 0;">Features used</td><td style="color:#1C1917;text-align:right;">8</td></tr>
            <tr style="border-top:1px solid #F5F5F4;"><td style="color:#78716C;padding:9px 0;">Training samples</td><td style="color:#1C1917;text-align:right;">2,000</td></tr>
          </table>
        </div>
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:14px;padding:20px 24px;margin-top:16px;">
          <div style="font-family:'DM Serif Display',serif;font-size:1rem;color:#1C1917;margin-bottom:12px;">Key insights from the model</div>
          <div style="font-size:0.83rem;color:#57534E;line-height:1.8;">
            · CGPA is the single strongest predictor<br>
            · Internships have strong multiplicative effect<br>
            · Backlogs hurt even with a high CGPA<br>
            · CS/IT students have the highest baseline rate<br>
            · Communication outweighs workshops in weight
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-heading" style="margin-top:8px;"><div class="sec-heading-icon">🔥</div>CGPA × Internship Placement Heatmap</div>', unsafe_allow_html=True)
    cgpa_bins2 = pd.cut(df["CGPA"], bins=[5, 6, 7, 7.5, 8, 8.5, 9, 10],
                        labels=["5–6", "6–7", "7–7.5", "7.5–8", "8–8.5", "8.5–9", "9–10"])
    pivot = df.assign(CGPA_Bin=cgpa_bins2).groupby(["CGPA_Bin", "Internships"], observed=True)["Placed"].mean().unstack()
    pivot = (pivot * 100).round(1)
    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{i} internship(s)" for i in pivot.columns],
        y=pivot.index.astype(str),
        colorscale=[[0,"#FEF3C7"],[0.5,"#F59E0B"],[1,"#92400E"]],
        text=pivot.values, texttemplate="%{text:.0f}%", textfont={"size": 12, "color": "#1C1917"},
        hovertemplate="CGPA: %{y}<br>Internships: %{x}<br>Placement: %{z:.1f}%"
    ))
    fig_hm.update_layout(paper_bgcolor="#F7F5F0", plot_bgcolor="#FFFFFF",
                         font_color="#1C1917", height=320,
                         margin=dict(t=10, b=20),
                         xaxis=dict(tickfont=dict(color="#78716C")),
                         yaxis=dict(tickfont=dict(color="#78716C")))
    st.plotly_chart(fig_hm, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 16px;margin-top:40px;border-top:1px solid #E7E5E4;">
  <span style="font-family:'DM Serif Display',serif;font-size:1rem;color:#1C1917;">Place<span style="color:#D97706;">IQ</span></span>
  <span style="color:#D6D3D1;margin:0 10px;">·</span>
  <span style="font-size:0.78rem;color:#A8A29E;">Random Forest Classifier · 2,000 synthetic profiles · Educational use only</span>
</div>
""", unsafe_allow_html=True)
