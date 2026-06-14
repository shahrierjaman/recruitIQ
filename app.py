"""
Resume Screening & Candidate Ranking System
Streamlit App — Professional UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import time
import joblib
import PyPDF2
import docx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecruitIQ — Resume Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(99,102,241,0.18);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(236,72,153,0.12);
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    line-height: 1.15;
    position: relative; z-index: 1;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.65);
    margin: 0;
    position: relative; z-index: 1;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.3);
    border: 1px solid rgba(99,102,241,0.5);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.85rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    position: relative; z-index: 1;
}

/* ── Stat Cards ── */
.stats-row { display: flex; gap: 1rem; margin-bottom: 1.75rem; }
.stat-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stat-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    color: #111827;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.stat-delta { font-size: 0.8rem; color: #10b981; font-weight: 500; }

/* ── Section Headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
    margin: 1.75rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e5e7eb;
    margin-left: 0.75rem;
}

/* ── Upload Zone ── */
.upload-zone {
    border: 2px dashed #c7d2fe;
    border-radius: 16px;
    background: #f5f3ff;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #6366f1; }

/* ── Candidate Card ── */
.candidate-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.candidate-card:hover {
    box-shadow: 0 8px 24px rgba(99,102,241,0.12);
    transform: translateY(-2px);
}
.rank-badge-gold {
    position: absolute; top: 0; right: 0;
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    color: white; font-weight: 800; font-size: 0.85rem;
    padding: 0.3rem 1rem;
    border-bottom-left-radius: 12px;
}
.rank-badge-silver {
    position: absolute; top: 0; right: 0;
    background: linear-gradient(135deg, #6b7280, #9ca3af);
    color: white; font-weight: 800; font-size: 0.85rem;
    padding: 0.3rem 1rem;
    border-bottom-left-radius: 12px;
}
.rank-badge-bronze {
    position: absolute; top: 0; right: 0;
    background: linear-gradient(135deg, #b45309, #d97706);
    color: white; font-weight: 800; font-size: 0.85rem;
    padding: 0.3rem 1rem;
    border-bottom-left-radius: 12px;
}
.rank-badge-other {
    position: absolute; top: 0; right: 0;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white; font-weight: 800; font-size: 0.85rem;
    padding: 0.3rem 1rem;
    border-bottom-left-radius: 12px;
}
.candidate-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.25rem;
}
.score-pill {
    display: inline-block;
    background: #ede9fe;
    color: #6d28d9;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.2rem 0.7rem;
    border-radius: 50px;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.score-pill-green {
    background: #d1fae5; color: #065f46;
}
.score-pill-pink {
    background: #fce7f3; color: #9d174d;
}

/* ── Progress Bar ── */
.progress-container {
    background: #f3f4f6;
    border-radius: 50px;
    height: 10px;
    margin: 0.5rem 0;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    transition: width 0.6s ease;
}
.progress-fill-green {
    background: linear-gradient(90deg, #10b981, #34d399);
}
.progress-fill-pink {
    background: linear-gradient(90deg, #ec4899, #f472b6);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29 0%, #302b63 60%, #24243e 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.25);
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Sidebar — all base text */
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Sidebar headings (### markdown) */
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.4rem !important;
    margin-bottom: 0.5rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    padding-bottom: 0.4rem;
}

/* Sidebar — slider labels & captions */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSlider p,
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption {
    color: #cbd5e1 !important;
}

/* Sidebar — slider track */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="progressbar"] {
    background: #6366f1 !important;
}

/* Sidebar — radio button text */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] p {
    color: #e2e8f0 !important;
}

/* Sidebar — text input */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="number"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input[type="text"]::placeholder,
[data-testid="stSidebar"] input[type="number"]::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* Sidebar — number input label */
[data-testid="stSidebar"] .stNumberInput label {
    color: #e2e8f0 !important;
}

/* Sidebar — divider */
[data-testid="stSidebar"] .divider {
    background: rgba(255,255,255,0.12) !important;
}

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
}
div.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Alerts ── */
.alert-success {
    background: #ecfdf5; border: 1px solid #6ee7b7;
    border-radius: 10px; padding: 0.85rem 1.2rem;
    color: #065f46; font-size: 0.9rem; margin: 0.75rem 0;
}
.alert-warning {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 10px; padding: 0.85rem 1.2rem;
    color: #92400e; font-size: 0.9rem; margin: 0.75rem 0;
}
.alert-info {
    background: #eff6ff; border: 1px solid #93c5fd;
    border-radius: 10px; padding: 0.85rem 1.2rem;
    color: #1e40af; font-size: 0.9rem; margin: 0.75rem 0;
}

/* ── Tag ── */
.tag {
    display: inline-block;
    background: #f3f4f6;
    color: #374151;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin: 0.2rem;
}

/* ── Divider ── */
.divider { height: 1px; background: #e5e7eb; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(file) -> str:
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"[PDF read error: {e}]"


def extract_text_from_docx(file) -> str:
    try:
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[DOCX read error: {e}]"


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return ""


def get_rank_badge(rank: int) -> str:
    if rank == 1:
        return '<div class="rank-badge-gold">🥇 Rank #1</div>'
    elif rank == 2:
        return '<div class="rank-badge-silver">🥈 Rank #2</div>'
    elif rank == 3:
        return '<div class="rank-badge-bronze">🥉 Rank #3</div>'
    else:
        return f'<div class="rank-badge-other">#{rank}</div>'


def score_label(score: float) -> str:
    if score >= 75:
        return "🟢 Excellent Match"
    elif score >= 55:
        return "🟡 Good Match"
    elif score >= 35:
        return "🟠 Partial Match"
    else:
        return "🔴 Low Match"


# ─────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models(model_dir: str = "resume_screening_model"):
    """Load TF-IDF vectorizer and Sentence Transformer."""
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    sbert_path = os.path.join(model_dir, "sbert_model")

    tfidf = None
    sbert = None

    if os.path.exists(tfidf_path):
        tfidf = joblib.load(tfidf_path)

    if os.path.exists(sbert_path):
        sbert = SentenceTransformer(sbert_path)
    else:
        # Fallback: download the base model
        sbert = SentenceTransformer("all-MiniLM-L6-v2")

    return tfidf, sbert


@st.cache_resource(show_spinner=False)
def fit_tfidf_on_jd(job_description_text: str):
    """Fit a quick TF-IDF if no saved model exists."""
    vec = TfidfVectorizer(max_features=5000, stop_words="english",
                          ngram_range=(1, 2), sublinear_tf=True)
    vec.fit([job_description_text])
    return vec


# ─────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────

def score_resume(resume_text: str, jd_text: str,
                 tfidf, sbert,
                 w_tfidf: float = 0.4, w_sbert: float = 0.6) -> dict:
    r_clean = clean_text(resume_text)
    j_clean = clean_text(jd_text)

    # TF-IDF score
    try:
        mat = tfidf.transform([r_clean, j_clean])
        tfidf_sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception:
        # If vectorizer wasn't fit on this vocab, fit on-the-fly
        vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
        mat = vec.fit_transform([r_clean, j_clean])
        tfidf_sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])

    # Sentence-BERT score
    try:
        emb = sbert.encode([r_clean, j_clean], convert_to_numpy=True)
        sbert_sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    except Exception:
        sbert_sim = tfidf_sim  # fallback

    final = (w_tfidf * tfidf_sim) + (w_sbert * sbert_sim)
    return {
        "tfidf_score": round(tfidf_sim * 100, 1),
        "sbert_score": round(sbert_sim * 100, 1),
        "final_score": round(final * 100, 1),
    }


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 1.5rem 0;">
        <div style="font-size:2.2rem;">🎯</div>
        <div style="font-size:1.1rem; font-weight:800; color:#ffffff;">RecruitIQ</div>
        <div style="font-size:0.75rem; color:#a5b4fc; margin-top:0.2rem;">AI-Powered Screening</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Scoring Weights")
    w_sbert = st.slider("Sentence-BERT weight", 0.0, 1.0, 0.6, 0.05,
                        help="Semantic similarity via transformer embeddings")
    w_tfidf = round(1.0 - w_sbert, 2)
    st.caption(f"TF-IDF weight auto-set to **{w_tfidf}**")

    st.markdown("### 🔧 Model Source")
    model_source = st.radio("", ["Auto-download (all-MiniLM-L6-v2)",
                                 "Load from saved model folder"],
                            label_visibility="collapsed")

    model_dir = "resume_screening_model"
    if model_source == "Load from saved model folder":
        model_dir = st.text_input("Model folder path", value=model_dir)

    st.markdown("### 🎛️ Filter Options")
    min_score = st.slider("Minimum match score (%)", 0, 90, 0, 5)
    top_n = st.number_input("Show top N candidates", min_value=1, max_value=50, value=10)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#cbd5e1; line-height:1.8; background:rgba(255,255,255,0.06);
                border-radius:10px; padding:0.9rem 1rem; border:1px solid rgba(255,255,255,0.1);">
        <span style="color:#a5b4fc; font-weight:700;">How it works</span><br>
        1. Upload resumes (PDF / DOCX / TXT)<br>
        2. Enter the job description<br>
        3. Click <b style="color:#ffffff;">Rank Candidates</b><br>
        4. Review ranked results &amp; charts
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">AI-Powered HR Tool</div>
    <h1 class="hero-title">Resume Screening &<br>Candidate Ranking</h1>
    <p class="hero-subtitle">
        Upload resumes, paste a job description, and get instant AI-ranked results
        using TF-IDF keyword matching and Sentence Transformer semantic analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────
with st.spinner("Loading AI models…"):
    tfidf_model, sbert_model = load_models(model_dir)

if tfidf_model is None:
    st.markdown("""
    <div class="alert-warning">
    ⚠️ <b>No saved TF-IDF model found.</b>
    A fresh vectorizer will be fitted on-the-fly from the job description.
    For better accuracy, run the Colab notebook first and save your model artifacts.
    </div>
    """, unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">📄 Upload Resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDF, DOCX, or TXT files here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"""
        <div class="alert-success">
        ✅ <b>{len(uploaded_files)} resume(s)</b> ready to process.
        </div>
        """, unsafe_allow_html=True)
        for f in uploaded_files:
            ext = f.name.split(".")[-1].upper()
            st.markdown(f'<span class="tag">📎 {f.name}</span>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📂</div>
            <div style="font-weight:600; color:#6366f1;">Drag & drop resumes here</div>
            <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.3rem;">PDF · DOCX · TXT</div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">📝 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste the full job description",
        height=280,
        placeholder="e.g. We are looking for a Senior Data Scientist with 5+ years of experience in machine learning, NLP, Python, TensorFlow, and cloud platforms...",
        label_visibility="collapsed",
    )
    if job_description:
        word_count = len(job_description.split())
        st.caption(f"📊 {word_count} words")

# ── Rank Button ────────────────────────────────────────────
st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("🚀  Rank Candidates", use_container_width=True)

# ─────────────────────────────────────────────────────────
# RANKING LOGIC
# ─────────────────────────────────────────────────────────

if run_btn:
    if not uploaded_files:
        st.markdown('<div class="alert-warning">⚠️ Please upload at least one resume.</div>',
                    unsafe_allow_html=True)
    elif not job_description.strip():
        st.markdown('<div class="alert-warning">⚠️ Please enter a job description.</div>',
                    unsafe_allow_html=True)
    else:
        # If no TF-IDF model, fit on JD
        active_tfidf = tfidf_model
        if active_tfidf is None:
            active_tfidf = fit_tfidf_on_jd(clean_text(job_description))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">⏳ Processing Resumes</div>',
                    unsafe_allow_html=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        resume_texts = {}

        for i, file in enumerate(uploaded_files):
            status_text.markdown(f"*Analyzing* **{file.name}** …")
            text = extract_text(file)
            resume_texts[file.name] = text

            scores = score_resume(
                text, job_description,
                active_tfidf, sbert_model,
                w_tfidf=w_tfidf, w_sbert=w_sbert
            )
            scores["candidate"] = file.name.rsplit(".", 1)[0]
            scores["filename"] = file.name
            scores["word_count"] = len(text.split())
            results.append(scores)
            progress_bar.progress((i + 1) / len(uploaded_files))
            time.sleep(0.05)

        status_text.empty()
        progress_bar.empty()

        # Sort and filter
        df_results = pd.DataFrame(results)
        df_results = df_results[df_results["final_score"] >= min_score]
        df_results = df_results.sort_values("final_score", ascending=False).head(top_n)
        df_results.reset_index(drop=True, inplace=True)
        df_results.index += 1

        # ── Summary Stats ────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)

        avg_score = df_results["final_score"].mean()
        top_score = df_results["final_score"].iloc[0] if len(df_results) > 0 else 0
        top_name = df_results["candidate"].iloc[0] if len(df_results) > 0 else "—"
        qualified = (df_results["final_score"] >= 55).sum()

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Screened</div>
                <div class="stat-value">{len(uploaded_files)}</div>
                <div class="stat-delta">resumes processed</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Top Score</div>
                <div class="stat-value">{top_score:.0f}%</div>
                <div class="stat-delta">{top_name[:20]}</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Average Score</div>
                <div class="stat-value">{avg_score:.0f}%</div>
                <div class="stat-delta">across all candidates</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Good Matches</div>
                <div class="stat-value">{qualified}</div>
                <div class="stat-delta">score ≥ 55%</div>
            </div>""", unsafe_allow_html=True)

        # ── Charts ────────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Score Visualizations</div>',
                    unsafe_allow_html=True)

        chart1, chart2 = st.columns(2)

        with chart1:
            fig = go.Figure()
            colors = ["#4F46E5" if s >= 55 else "#E5E7EB" for s in df_results["final_score"]]
            fig.add_trace(go.Bar(
                x=df_results["candidate"],
                y=df_results["final_score"],
                marker_color=colors,
                marker_line_width=0,
                text=[f"{s:.1f}%" for s in df_results["final_score"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(text="Overall Match Score", font=dict(size=14, family="Inter")),
                yaxis=dict(range=[0, 105], title="Score (%)"),
                xaxis=dict(tickangle=-25),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                margin=dict(t=50, b=60, l=40, r=20),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_results["candidate"], y=df_results["tfidf_score"],
                mode="lines+markers", name="TF-IDF",
                line=dict(color="#4F46E5", width=2.5),
                marker=dict(size=8),
            ))
            fig2.add_trace(go.Scatter(
                x=df_results["candidate"], y=df_results["sbert_score"],
                mode="lines+markers", name="Sentence-BERT",
                line=dict(color="#EC4899", width=2.5, dash="dot"),
                marker=dict(size=8),
            ))
            fig2.update_layout(
                title=dict(text="TF-IDF vs Sentence-BERT", font=dict(size=14, family="Inter")),
                yaxis=dict(range=[0, 100], title="Score (%)"),
                xaxis=dict(tickangle=-25),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.25),
                font=dict(family="Inter"),
                margin=dict(t=50, b=80, l=40, r=20),
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Radar Chart (top 3) ───────────────────────────────────
        if len(df_results) >= 2:
            top3 = df_results.head(3)
            fig3 = go.Figure()
            categories = ["TF-IDF Score", "SBERT Score", "Final Score"]
            for _, row in top3.iterrows():
                fig3.add_trace(go.Scatterpolar(
                    r=[row["tfidf_score"], row["sbert_score"], row["final_score"]],
                    theta=categories,
                    fill="toself",
                    name=row["candidate"][:20],
                    opacity=0.7,
                ))
            fig3.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title=dict(text="Top 3 Candidates — Radar View",
                           font=dict(size=14, family="Inter")),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                height=360,
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── Ranked Candidate Cards ────────────────────────────────
        st.markdown('<div class="section-header">🏆 Ranked Candidates</div>',
                    unsafe_allow_html=True)

        for rank, row in df_results.iterrows():
            badge = get_rank_badge(rank)
            label = score_label(row["final_score"])

            tfidf_bar = int(row["tfidf_score"])
            sbert_bar = int(row["sbert_score"])
            final_bar = int(row["final_score"])

            st.markdown(f"""
            <div class="candidate-card">
                {badge}
                <div class="candidate-name">{row["candidate"]}</div>
                <div style="margin-bottom:0.75rem;">
                    <span class="score-pill">{label}</span>
                    <span class="score-pill score-pill-green">Words: {row["word_count"]}</span>
                    <span class="score-pill score-pill-pink">File: {row["filename"]}</span>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin-top:0.5rem;">
                    <div>
                        <div style="font-size:0.72rem; font-weight:600; color:#9ca3af; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem;">TF-IDF Score</div>
                        <div style="font-size:1.3rem; font-weight:800; color:#4f46e5;">{row["tfidf_score"]:.1f}%</div>
                        <div class="progress-container">
                            <div class="progress-fill" style="width:{tfidf_bar}%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem; font-weight:600; color:#9ca3af; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem;">Sentence-BERT</div>
                        <div style="font-size:1.3rem; font-weight:800; color:#ec4899;">{row["sbert_score"]:.1f}%</div>
                        <div class="progress-container">
                            <div class="progress-fill progress-fill-pink" style="width:{sbert_bar}%;"></div>
                        </div>
                    </div>
                    <div>
                        <div style="font-size:0.72rem; font-weight:600; color:#9ca3af; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem;">Final Score</div>
                        <div style="font-size:1.3rem; font-weight:800; color:#10b981;">{row["final_score"]:.1f}%</div>
                        <div class="progress-container">
                            <div class="progress-fill progress-fill-green" style="width:{final_bar}%;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Download Results ──────────────────────────────────────
        st.markdown('<div class="section-header">⬇️ Export Results</div>',
                    unsafe_allow_html=True)

        export_df = df_results[["candidate", "filename", "tfidf_score",
                                "sbert_score", "final_score", "word_count"]].copy()
        export_df.index.name = "Rank"
        export_df.columns = ["Candidate", "File", "TF-IDF Score (%)",
                              "SBERT Score (%)", "Final Score (%)", "Word Count"]

        csv = export_df.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download Rankings CSV",
            data=csv,
            file_name="candidate_rankings.csv",
            mime="text/csv",
        )

        # Full table
        with st.expander("📋 View Full Results Table"):
            st.dataframe(export_df, use_container_width=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1rem 0;
            font-size:0.78rem; color:#9ca3af; border-top:1px solid #e5e7eb; margin-top:3rem;">
    <b>RecruitIQ</b> · Built with Streamlit · TF-IDF + Sentence Transformers + Cosine Similarity
</div>
""", unsafe_allow_html=True)