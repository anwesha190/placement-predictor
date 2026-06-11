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

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f1117; }

.hero-card {
    background: linear-gradient(135deg, #1e2235 0%, #252d45 100%);
    border: 1px solid #2e3650;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
}

.metric-card {
    background: #1a1f30;
    border: 1px solid #2a3050;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 100%;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #7c6af7;
    line-height: 1.1;
}

.metric-label {
    font-size: 0.78rem;
    color: #8892aa;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.placed-banner {
    background: linear-gradient(135deg, #0f3d2b, #1a5c3a);
    border: 1px solid #2ecc71;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
}

.not-placed-banner {
    background: linear-gradient(135deg, #3d1a1a, #5c2222);
    border: 1px solid #e74c3c;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 20px 0;
}

.reason-card {
    background: #1a1f30;
    border-left: 4px solid #7c6af7;
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    margin: 10px 0;
}

.career-card {
    background: linear-gradient(135deg, #1e1a35, #28203d);
    border: 1px solid #3d3060;
    border-radius: 12px;
    padding: 22px;
    margin: 10px 0;
}

.step-item {
    background: #1a1f30;
    border: 1px solid #2a3050;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    align-items: center;
}

.tag {
    display: inline-block;
    background: #2a2050;
    color: #9d8ff0;
    border: 1px solid #4a3880;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px;
}

.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c8d0e8;
    margin: 20px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a3050;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA GENERATION ────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=2000):
    np.random.seed(42)
    cgpa        = np.round(np.random.uniform(5.0, 10.0, n), 2)
    internships = np.random.randint(0, 4, n)
    projects    = np.random.randint(0, 6, n)
    workshops   = np.random.randint(0, 5, n)
    aptitude    = np.random.randint(40, 100, n)
    communication = np.random.randint(1, 6, n)
    backlogs    = np.random.randint(0, 5, n)
    stream      = np.random.choice(["CS", "IT", "ECE", "EEE", "MECH", "CIVIL"], n)

    stream_bonus = {"CS": 0.15, "IT": 0.12, "ECE": 0.08, "EEE": 0.06, "MECH": 0.03, "CIVIL": 0.02}
    sb = np.array([stream_bonus[s] for s in stream])

    score = (
        0.30 * (cgpa - 5) / 5 +
        0.20 * internships / 3 +
        0.15 * projects / 5 +
        0.10 * workshops / 4 +
        0.12 * (aptitude - 40) / 60 +
        0.08 * communication / 5 +
        sb -
        0.10 * backlogs / 4 +
        np.random.normal(0, 0.05, n)
    )

    placed = (score > 0.42).astype(int)
    le = LabelEncoder()

    return pd.DataFrame({
        "CGPA": cgpa, "Internships": internships, "Projects": projects,
        "Workshops": workshops, "AptitudeScore": aptitude,
        "CommunicationSkill": communication, "Backlogs": backlogs,
        "Stream": stream, "StreamCode": le.fit_transform(stream),
        "Placed": placed
    })

@st.cache_resource
def train_model(df):
    features = ["CGPA", "Internships", "Projects", "Workshops",
                "AptitudeScore", "CommunicationSkill", "Backlogs", "StreamCode"]
    X = df[features]
    y = df["Placed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, acc, features

# ─── CAREER SUGGESTIONS ─────────────────────────────────────────────────────────
def get_career_suggestion(cgpa, internships, projects, aptitude, communication, stream):
    careers = []

    if stream in ["CS", "IT"]:
        if cgpa >= 8.0 and projects >= 3:
            careers.append({
                "title": "🧑‍💻 Software Development Engineer",
                "match": 95,
                "steps": ["Master DSA & System Design", "Contribute to open-source", "Prepare for FAANG interviews"],
                "skills": ["Python/Java", "LeetCode", "Git", "System Design"]
            })
        if aptitude >= 75 and cgpa >= 7.5:
            careers.append({
                "title": "📊 Data Analyst / Data Scientist",
                "match": 88,
                "steps": ["Learn Power BI / Tableau", "Improve SQL & Python Pandas", "Build 2 end-to-end projects"],
                "skills": ["SQL", "Python", "Power BI", "Statistics"]
            })
        if communication >= 4:
            careers.append({
                "title": "☁️ Cloud & DevOps Engineer",
                "match": 82,
                "steps": ["Get AWS/GCP certification", "Learn Docker & Kubernetes", "Build CI/CD pipelines"],
                "skills": ["AWS", "Docker", "Linux", "Terraform"]
            })
    elif stream in ["ECE", "EEE"]:
        careers.append({
            "title": "🔌 Embedded Systems Engineer",
            "match": 85,
            "steps": ["Master C/C++ for embedded", "Learn RTOS concepts", "Build IoT projects"],
            "skills": ["C/C++", "RTOS", "Arduino", "FPGA"]
        })
        if cgpa >= 7.5:
            careers.append({
                "title": "📡 VLSI / Hardware Design",
                "match": 80,
                "steps": ["Learn Verilog/VHDL", "Practice on ModelSim", "Apply to semiconductor companies"],
                "skills": ["Verilog", "VHDL", "Cadence", "Signal Processing"]
            })
    else:
        careers.append({
            "title": "🏗️ Core Engineering / Consulting",
            "match": 78,
            "steps": ["Get domain certifications", "Build technical projects", "Apply to PSU/core companies"],
            "skills": ["AutoCAD", "MATLAB", "Domain expertise", "Project Management"]
        })

    if not careers:
        careers.append({
            "title": "🌐 Business Analyst / IT Consultant",
            "match": 72,
            "steps": ["Learn SQL & Excel deeply", "Get PMP or CBAP certified", "Build analytical portfolio"],
            "skills": ["SQL", "Excel", "Tableau", "Communication"]
        })

    return careers[:2]

# ─── EXPLANATION ────────────────────────────────────────────────────────────────
def get_explanation(cgpa, internships, projects, aptitude, backlogs, communication, placed_prob):
    positive, negative = [], []

    if cgpa >= 8.5:    positive.append(("🎯 Excellent CGPA", f"{cgpa}/10 — Top academic performer"))
    elif cgpa >= 7.5:  positive.append(("✅ Good CGPA", f"{cgpa}/10 — Above average academics"))
    elif cgpa >= 6.5:  negative.append(("⚠️ Average CGPA", f"{cgpa}/10 — Needs improvement"))
    else:              negative.append(("❌ Low CGPA", f"{cgpa}/10 — Critical area to work on"))

    if internships >= 2:   positive.append(("💼 Strong Internship Record", f"{internships} internships — Industry ready"))
    elif internships == 1: positive.append(("✅ Has Internship Experience", "1 internship — Good start"))
    else:                  negative.append(("⚠️ No Internship Experience", "0 internships — Apply urgently"))

    if projects >= 4:      positive.append(("🚀 Impressive Project Portfolio", f"{projects} projects — Well demonstrated skills"))
    elif projects >= 2:    positive.append(("✅ Decent Projects", f"{projects} projects — Keep building"))
    else:                  negative.append(("⚠️ Few Projects", f"{projects} project(s) — Build more ASAP"))

    if aptitude >= 80:     positive.append(("🧠 High Aptitude Score", f"{aptitude}/100 — Strong analytical skills"))
    elif aptitude >= 65:   positive.append(("✅ Good Aptitude", f"{aptitude}/100"))
    else:                  negative.append(("⚠️ Low Aptitude Score", f"{aptitude}/100 — Practice more"))

    if communication >= 4: positive.append(("🗣️ Strong Communication", f"{communication}/5 — Great soft skills"))
    elif communication == 3: negative.append(("⚠️ Average Communication", f"{communication}/5 — Room to grow"))
    else:                  negative.append(("❌ Weak Communication", f"{communication}/5 — Must improve urgently"))

    if backlogs > 0:       negative.append(("❌ Active Backlogs", f"{backlogs} backlog(s) — Clear before placements"))

    return positive, negative

# ─── MAIN APP ───────────────────────────────────────────────────────────────────
df = generate_dataset()
model, accuracy, features = train_model(df)
le = LabelEncoder().fit(df["Stream"])

# Header
st.markdown("""
<div class="hero-card">
    <h1 style="color:#e8ecf4;margin:0;font-size:2rem;font-weight:700;">🎓 Student Placement Predictor</h1>
    <p style="color:#8892aa;margin:8px 0 0 0;font-size:0.95rem;">
        AI-powered placement probability using Random Forest · Trained on 2,000 synthetic student profiles
    </p>
</div>
""", unsafe_allow_html=True)

# Dataset metrics
c1, c2, c3, c4 = st.columns(4)
placed_pct = df["Placed"].mean() * 100
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Training Samples</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{accuracy*100:.1f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{placed_pct:.0f}%</div><div class="metric-label">Placement Rate</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">8</div><div class="metric-label">Features Used</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Data Insights", "🤖 Model Info"])

# ═══════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown('<p class="section-header">Student Profile</p>', unsafe_allow_html=True)

        stream = st.selectbox("🎓 Branch / Stream", ["CS", "IT", "ECE", "EEE", "MECH", "CIVIL"])
        cgpa = st.slider("📈 CGPA", 5.0, 10.0, 7.5, 0.1)
        internships = st.slider("💼 Internships Completed", 0, 3, 1)
        projects = st.slider("🛠️ Projects Built", 0, 5, 2)
        workshops = st.slider("🧩 Workshops / Certifications", 0, 4, 1)
        aptitude = st.slider("🧠 Aptitude Score (out of 100)", 40, 100, 70)
        communication = st.slider("🗣️ Communication Skill (1–5)", 1, 5, 3)
        backlogs = st.slider("📋 Active Backlogs", 0, 4, 0)

        predict_btn = st.button("✨ Predict Placement", use_container_width=True, type="primary")

    with right:
        if predict_btn:
            stream_enc = le.transform([stream])[0]
            input_vec = np.array([[cgpa, internships, projects, workshops,
                                   aptitude, communication, backlogs, stream_enc]])
            prob = model.predict_proba(input_vec)[0][1]
            prediction = int(prob >= 0.5)

            # Result banner
            if prediction == 1:
                st.markdown(f"""
                <div class="placed-banner">
                    <h2 style="color:#2ecc71;margin:0;font-size:1.6rem;">🎉 Likely to be Placed!</h2>
                    <p style="color:#a8f0c0;margin:6px 0 0 0;font-size:0.95rem;">
                        Placement probability: <strong>{prob*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="not-placed-banner">
                    <h2 style="color:#e74c3c;margin:0;font-size:1.6rem;">⚠️ Placement at Risk</h2>
                    <p style="color:#f0a0a0;margin:6px 0 0 0;font-size:0.95rem;">
                        Placement probability: <strong>{prob*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 36, "color": "#e8ecf4"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8892aa"},
                    "bar": {"color": "#7c6af7"},
                    "steps": [
                        {"range": [0, 40], "color": "#2a1a1a"},
                        {"range": [40, 65], "color": "#2a2a1a"},
                        {"range": [65, 100], "color": "#1a2a1a"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 3}, "value": 50},
                },
                title={"text": "Placement Probability", "font": {"color": "#8892aa", "size": 14}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#1a1f30", font_color="#e8ecf4",
                height=260, margin=dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Explanation
            st.markdown('<p class="section-header">🔍 Why This Prediction?</p>', unsafe_allow_html=True)
            pos, neg = get_explanation(cgpa, internships, projects, aptitude, backlogs, communication, prob)

            if pos:
                st.markdown("**✅ Strengths**")
                for title, desc in pos:
                    st.markdown(f'<div class="reason-card"><strong style="color:#9d8ff0;">{title}</strong><br><span style="color:#8892aa;font-size:0.88rem;">{desc}</span></div>', unsafe_allow_html=True)

            if neg:
                st.markdown("**⚠️ Areas to Improve**")
                for title, desc in neg:
                    st.markdown(f'<div class="reason-card" style="border-left-color:#e74c3c;"><strong style="color:#e88;">{title}</strong><br><span style="color:#8892aa;font-size:0.88rem;">{desc}</span></div>', unsafe_allow_html=True)

            # Career Suggestions
            st.markdown('<p class="section-header">🚀 Career Recommendations</p>', unsafe_allow_html=True)
            careers = get_career_suggestion(cgpa, internships, projects, aptitude, communication, stream)

            for i, career in enumerate(careers):
                badge = "🥇 Best Match" if i == 0 else "🥈 Alternative"
                st.markdown(f"""
                <div class="career-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <h3 style="color:#c8d0e8;margin:0;font-size:1.05rem;">{career['title']}</h3>
                        <span style="color:#7c6af7;font-size:0.8rem;font-weight:600;">{badge} · {career['match']}% fit</span>
                    </div>
                    <div style="margin:10px 0 6px 0;">
                        {''.join([f'<span class="tag">{s}</span>' for s in career["skills"]])}
                    </div>
                    <p style="color:#6a7490;font-size:0.82rem;margin:8px 0 4px 0;font-weight:600;">NEXT STEPS:</p>
                    {''.join([f'<div style="color:#a0aac0;font-size:0.88rem;padding:4px 0;">{"①②③④"[j]} {step}</div>' for j, step in enumerate(career["steps"])])}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#4a5570;">
                <div style="font-size:4rem;margin-bottom:16px;">🎓</div>
                <p style="font-size:1.1rem;color:#6a7490;">Fill in your profile on the left<br>and click <strong style="color:#7c6af7;">Predict Placement</strong></p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">📊 How Features Affect Placement</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # CGPA vs Placement
        cgpa_bins = pd.cut(df["CGPA"], bins=[5, 6, 7, 7.5, 8, 8.5, 9, 10],
                           labels=["5–6", "6–7", "7–7.5", "7.5–8", "8–8.5", "8.5–9", "9–10"])
        cgpa_rate = df.groupby(cgpa_bins, observed=True)["Placed"].mean().reset_index()
        cgpa_rate.columns = ["CGPA Range", "Placement Rate"]
        cgpa_rate["Placement Rate"] = (cgpa_rate["Placement Rate"] * 100).round(1)

        fig1 = px.bar(cgpa_rate, x="CGPA Range", y="Placement Rate",
                      title="📈 CGPA vs Placement Rate (%)",
                      color="Placement Rate", color_continuous_scale="Viridis",
                      text="Placement Rate")
        fig1.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig1.update_layout(paper_bgcolor="#1a1f30", plot_bgcolor="#1a1f30",
                           font_color="#e8ecf4", title_font_size=14,
                           coloraxis_showscale=False, height=340,
                           margin=dict(t=50, b=30))
        fig1.update_xaxes(gridcolor="#2a3050", tickfont_color="#8892aa")
        fig1.update_yaxes(gridcolor="#2a3050", tickfont_color="#8892aa", range=[0, 110])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Internships vs Placement
        intern_rate = df.groupby("Internships")["Placed"].mean().reset_index()
        intern_rate["Placed"] = (intern_rate["Placed"] * 100).round(1)

        fig2 = px.bar(intern_rate, x="Internships", y="Placed",
                      title="💼 Internships vs Placement Rate (%)",
                      color="Placed", color_continuous_scale="Plasma",
                      text="Placed")
        fig2.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig2.update_layout(paper_bgcolor="#1a1f30", plot_bgcolor="#1a1f30",
                           font_color="#e8ecf4", title_font_size=14,
                           coloraxis_showscale=False, height=340,
                           margin=dict(t=50, b=30))
        fig2.update_xaxes(gridcolor="#2a3050", tickfont_color="#8892aa",
                          tickvals=[0,1,2,3], ticktext=["0","1","2","3+"])
        fig2.update_yaxes(gridcolor="#2a3050", tickfont_color="#8892aa", range=[0, 110])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # CGPA scatter by placed
        sample = df.sample(600, random_state=1)
        fig3 = px.scatter(sample, x="CGPA", y="AptitudeScore",
                          color=sample["Placed"].map({1: "Placed", 0: "Not Placed"}),
                          title="🎯 CGPA × Aptitude Score Distribution",
                          opacity=0.65, size_max=8,
                          color_discrete_map={"Placed": "#7c6af7", "Not Placed": "#e74c3c"})
        fig3.update_layout(paper_bgcolor="#1a1f30", plot_bgcolor="#1a1f30",
                           font_color="#e8ecf4", title_font_size=14,
                           legend=dict(bgcolor="#252d45", bordercolor="#2e3650"),
                           height=340, margin=dict(t=50, b=30))
        fig3.update_xaxes(gridcolor="#2a3050")
        fig3.update_yaxes(gridcolor="#2a3050")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Stream placement rate
        stream_rate = df.groupby("Stream")["Placed"].mean().reset_index()
        stream_rate["Placed"] = (stream_rate["Placed"] * 100).round(1)
        stream_rate = stream_rate.sort_values("Placed", ascending=True)

        fig4 = px.bar(stream_rate, y="Stream", x="Placed",
                      orientation="h",
                      title="🎓 Branch-wise Placement Rate (%)",
                      color="Placed", color_continuous_scale="Teal",
                      text="Placed")
        fig4.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig4.update_layout(paper_bgcolor="#1a1f30", plot_bgcolor="#1a1f30",
                           font_color="#e8ecf4", title_font_size=14,
                           coloraxis_showscale=False, height=340,
                           margin=dict(t=50, b=30, r=60))
        fig4.update_xaxes(gridcolor="#2a3050", range=[0, 110])
        fig4.update_yaxes(gridcolor="#2a3050")
        st.plotly_chart(fig4, use_container_width=True)

    # Projects effect
    proj_rate = df.groupby("Projects")["Placed"].mean().reset_index()
    proj_rate["Placed"] = (proj_rate["Placed"] * 100).round(1)

    fig5 = px.line(proj_rate, x="Projects", y="Placed",
                   title="🛠️ Number of Projects vs Placement Rate (%)",
                   markers=True, line_shape="spline")
    fig5.update_traces(line_color="#7c6af7", marker_color="#9d8ff0",
                       marker_size=10, line_width=3)
    fig5.add_hrect(y0=50, y1=100, fillcolor="#7c6af7", opacity=0.05,
                   annotation_text="Placement zone", annotation_position="top right")
    fig5.update_layout(paper_bgcolor="#1a1f30", plot_bgcolor="#1a1f30",
                       font_color="#e8ecf4", title_font_size=14,
                       height=300, margin=dict(t=50, b=30))
    fig5.update_xaxes(gridcolor="#2a3050", dtick=1)
    fig5.update_yaxes(gridcolor="#2a3050", range=[0, 105])
    st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">🤖 Random Forest Model Details</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Feature importances
        importances = model.feature_importances_
        feat_labels = ["CGPA", "Internships", "Projects", "Workshops",
                       "Aptitude", "Communication", "Backlogs", "Stream"]
        fi_df = pd.DataFrame({"Feature": feat_labels, "Importance": importances})
        fi_df = fi_df.sort_values("Importance")

        fig_fi = px.bar(fi_df, y="Feature", x="Importance", orientation="h",
                        title="🌲 Feature Importances",
                        color="Importance", color_continuous_scale="Purples",
                        text=fi_df["Importance"].apply(lambda x: f"{x:.3f}"))
        fig_fi.update_layout(paper_bgcolor="#1a1f30", plot_bgcolor="#1a1f30",
                             font_color="#e8ecf4", title_font_size=14,
                             coloraxis_showscale=False, height=380,
                             margin=dict(t=50, b=20, r=60))
        fig_fi.update_xaxes(gridcolor="#2a3050")
        fig_fi.update_yaxes(gridcolor="#2a3050")
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        st.markdown("""
        <div class="hero-card">
            <h3 style="color:#c8d0e8;margin:0 0 16px 0;">Model Configuration</h3>
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
                <tr><td style="color:#8892aa;padding:8px 0;">Algorithm</td><td style="color:#e8ecf4;text-align:right;">Random Forest Classifier</td></tr>
                <tr style="border-top:1px solid #2a3050;"><td style="color:#8892aa;padding:8px 0;">Trees</td><td style="color:#e8ecf4;text-align:right;">200 estimators</td></tr>
                <tr style="border-top:1px solid #2a3050;"><td style="color:#8892aa;padding:8px 0;">Max Depth</td><td style="color:#e8ecf4;text-align:right;">10</td></tr>
                <tr style="border-top:1px solid #2a3050;"><td style="color:#8892aa;padding:8px 0;">Train / Test Split</td><td style="color:#e8ecf4;text-align:right;">80% / 20%</td></tr>
                <tr style="border-top:1px solid #2a3050;"><td style="color:#8892aa;padding:8px 0;">Test Accuracy</td><td style="color:#7c6af7;text-align:right;font-weight:700;">{:.1f}%</td></tr>
                <tr style="border-top:1px solid #2a3050;"><td style="color:#8892aa;padding:8px 0;">Features</td><td style="color:#e8ecf4;text-align:right;">8</td></tr>
                <tr style="border-top:1px solid #2a3050;"><td style="color:#8892aa;padding:8px 0;">Training Samples</td><td style="color:#e8ecf4;text-align:right;">2,000</td></tr>
            </table>
        </div>
        """.format(accuracy * 100), unsafe_allow_html=True)

        st.markdown("""
        <div class="hero-card" style="margin-top:16px;">
            <h3 style="color:#c8d0e8;margin:0 0 12px 0;">📌 Key Insights</h3>
            <p style="color:#8892aa;font-size:0.88rem;line-height:1.7;margin:0;">
                • <strong style="color:#9d8ff0;">CGPA</strong> is the single strongest predictor of placement<br>
                • <strong style="color:#9d8ff0;">Internships</strong> have a strong multiplicative effect<br>
                • <strong style="color:#9d8ff0;">Backlogs</strong> negatively impact chances even with high CGPA<br>
                • <strong style="color:#9d8ff0;">CS/IT</strong> students have highest baseline placement rates<br>
                • <strong style="color:#9d8ff0;">Communication</strong> matters more than workshops<br>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # CGPA + Internship heatmap
    st.markdown('<p class="section-header">🔥 CGPA × Internship Placement Heatmap</p>', unsafe_allow_html=True)

    cgpa_bins2 = pd.cut(df["CGPA"], bins=[5, 6, 7, 7.5, 8, 8.5, 9, 10],
                        labels=["5–6", "6–7", "7–7.5", "7.5–8", "8–8.5", "8.5–9", "9–10"])
    pivot = df.assign(CGPA_Bin=cgpa_bins2).groupby(
        ["CGPA_Bin", "Internships"], observed=True)["Placed"].mean().unstack()
    pivot = (pivot * 100).round(1)

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{i} Intern" for i in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="Viridis", text=pivot.values,
        texttemplate="%{text:.0f}%", textfont={"size": 12},
        hovertemplate="CGPA: %{y}<br>Internships: %{x}<br>Placement Rate: %{z:.1f}%"
    ))
    fig_hm.update_layout(paper_bgcolor="#1a1f30", plot_bgcolor="#1a1f30",
                         font_color="#e8ecf4", height=340,
                         margin=dict(t=20, b=20),
                         xaxis=dict(tickfont_color="#8892aa"),
                         yaxis=dict(tickfont_color="#8892aa"))
    st.plotly_chart(fig_hm, use_container_width=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:24px;color:#4a5570;font-size:0.8rem;border-top:1px solid #2a3050;margin-top:32px;">
    Built with Streamlit · Random Forest Classifier · Synthetic Dataset · 🎓 For Educational Use
</div>
""", unsafe_allow_html=True)
