<div align="center">

# 🎯 RecruitIQ — Resume Screening & Candidate Ranking System

**AI-powered resume screening using TF-IDF keyword matching and Sentence Transformer semantic analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?style=flat-square)](https://www.sbert.net)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

![RecruitIQ Banner](https://img.shields.io/badge/-Upload%20Resumes%20%E2%86%92%20Paste%20JD%20%E2%86%92%20Get%20AI%20Rankings-4F46E5?style=for-the-badge)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Scoring Formula](#-scoring-formula)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
  - [Step 1 — Train on Google Colab](#step-1--train-on-google-colab)
  - [Step 2 — Run the Streamlit App](#step-2--run-the-streamlit-app)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Model Artifacts](#-model-artifacts)
- [Configuration](#-configuration)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

**RecruitIQ** is an end-to-end AI-powered HR tool that automates resume screening and candidate ranking. It combines classical NLP (TF-IDF) with modern deep learning (Sentence Transformers) to produce a hybrid similarity score between a candidate's resume and a target job description.

The project ships as two components:

| Component | Description |
|---|---|
| `Resume_Screening_Colab.ipynb` | Google Colab notebook for training and saving model artifacts |
| `app.py` | Streamlit web app for uploading resumes and viewing ranked results |

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING  (Colab)                        │
│                                                                 │
│  job_descriptions.csv  ──►  Clean & Preprocess                 │
│                                  │                              │
│                    ┌─────────────┴──────────────┐              │
│                    ▼                            ▼               │
│             TF-IDF Vectorizer        Sentence Transformer       │
│          (all-MiniLM-L6-v2 fit)    (all-MiniLM-L6-v2 encode)  │
│                    │                            │               │
│                    └─────────────┬──────────────┘              │
│                                  ▼                              │
│                       Save Artifacts to Drive                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE  (Streamlit)                     │
│                                                                 │
│  Upload Resume (PDF/DOCX/TXT)  +  Paste Job Description        │
│                    │                            │               │
│                    ▼                            ▼               │
│             TF-IDF Similarity         SBERT Cosine Similarity   │
│                    │                            │               │
│                    └─────────────┬──────────────┘              │
│                                  ▼                              │
│         Final Score = 0.4 × TF-IDF  +  0.6 × SBERT            │
│                                  │                              │
│                    Ranked Candidate Cards + Charts              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Scoring Formula

Each resume is evaluated against the job description using two complementary methods:

$$\text{Final Score} = w_{\text{tfidf}} \times \text{CosineSim}(\text{TF-IDF}) + w_{\text{sbert}} \times \text{CosineSim}(\text{SBERT})$$

| Component | Default Weight | What It Captures |
|---|---|---|
| **TF-IDF Cosine Similarity** | 0.4 | Keyword and phrase overlap |
| **Sentence-BERT Cosine Similarity** | 0.6 | Semantic meaning & context |

> Weights are fully adjustable via the sidebar slider in the Streamlit app.

**Match tiers:**

| Score | Label |
|---|---|
| ≥ 75% | 🟢 Excellent Match |
| 55–74% | 🟡 Good Match |
| 35–54% | 🟠 Partial Match |
| < 35% | 🔴 Low Match |

---

## 📁 Project Structure

```
recruitiq/
│
├── Resume_Screening_Colab.ipynb   # Training pipeline (run on Google Colab)
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
└── resume_screening_model/        # Generated after running the notebook
    ├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
    ├── sbert_model/               # Saved Sentence Transformer weights
    ├── job_embeddings.npy         # Pre-computed job profile embeddings
    ├── job_profiles.csv           # Cleaned job profile dataset
    └── config.json                # Model configuration
```

---

## 🚀 Quickstart

### Prerequisites

- Python 3.8+
- Google account (for Colab + Drive)
- `job_descriptions.csv` dataset uploaded to Google Drive

---

### Step 1 — Train on Google Colab

Open `Resume_Screening_Colab.ipynb` in Google Colab and run all cells in order.

The notebook walks through 10 steps:

```
Step 1  →  Install dependencies
Step 2  →  Mount Google Drive & load dataset
Step 3  →  Exploratory Data Analysis (EDA)
Step 4  →  Data preprocessing & text cleaning
Step 5  →  TF-IDF vectorization (5,000 features, bigrams)
Step 6  →  Sentence Transformer embeddings (all-MiniLM-L6-v2)
Step 7  →  Build & test the ResumeRanker class
Step 8  →  Visualize ranking results
Step 9  →  Save all model artifacts to Google Drive
Step 10 →  Summary & next steps
```

After Step 9, download the `resume_screening_model/` folder from your Google Drive and place it in your project root.

---

### Step 2 — Run the Streamlit App

**1. Clone the repository**

```bash
git clone https://github.com/your-username/recruitiq.git
cd recruitiq
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. (Optional) Place model artifacts**

```bash
# Copy your trained model folder here
mv ~/Downloads/resume_screening_model ./resume_screening_model
```

> If no saved model is found, the app auto-downloads `all-MiniLM-L6-v2` and fits TF-IDF on-the-fly from the job description.

**4. Launch the app**

```bash
streamlit run app.py
```

```bash
Get the model 
https://drive.google.com/file/d/15xDfwtD-fQ1o005XvzoCSv8QeTlHnogX/view?usp=sharing
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ✨ Features

- **Multi-format resume support** — Upload PDF, DOCX, or TXT files
- **Batch processing** — Screen multiple candidates simultaneously with a progress bar
- **Hybrid scoring** — Combines keyword-level TF-IDF with deep semantic SBERT similarity
- **Adjustable weights** — Fine-tune TF-IDF vs. SBERT balance via sidebar slider
- **Rich visualizations** — Bar charts, line charts, and radar charts via Plotly
- **Ranked candidate cards** — Gold / Silver / Bronze badges, score breakdowns, progress bars
- **Score filtering** — Set minimum match threshold to surface only relevant candidates
- **Top-N control** — Display exactly as many results as needed
- **CSV export** — Download the full ranked results table
- **No model required** — Falls back gracefully if no pre-trained artifacts are found

---

## 🛠 Tech Stack

| Layer | Library |
|---|---|
| **Web Framework** | [Streamlit](https://streamlit.io) |
| **NLP — Classical** | [scikit-learn](https://scikit-learn.org) TfidfVectorizer |
| **NLP — Semantic** | [sentence-transformers](https://www.sbert.net) `all-MiniLM-L6-v2` |
| **Similarity** | Cosine Similarity (sklearn) |
| **PDF Parsing** | PyPDF2 |
| **DOCX Parsing** | python-docx |
| **Data** | pandas, numpy |
| **Visualizations** | Plotly |
| **Model Persistence** | joblib |
| **Training Environment** | Google Colab + Google Drive |

---

## 💾 Model Artifacts

Generated by the Colab notebook and loaded by the Streamlit app:

| File | Description |
|---|---|
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer fitted on job profile corpus |
| `sbert_model/` | Fine-tuned / cached `all-MiniLM-L6-v2` weights |
| `job_embeddings.npy` | Pre-computed numpy embeddings for job profiles |
| `job_profiles.csv` | Cleaned and preprocessed job description dataset |
| `config.json` | Stores weights, model name, feature count, and column list |

---

## ⚙️ Configuration

`config.json` (auto-generated by Colab):

```json
{
  "tfidf_weight": 0.4,
  "sbert_weight": 0.6,
  "sbert_model_name": "all-MiniLM-L6-v2",
  "tfidf_max_features": 5000,
  "sample_size": 500,
  "columns": ["Qualifications", "Experience", "Work Type", "location", "Country", "Company Size"]
}
```

All weights can be overridden live in the Streamlit sidebar without touching any files.

---

## 📊 Screenshots

| Section | Description |
|---|---|
| **Hero + Upload** | Drag-and-drop resume uploader with job description text area |
| **Summary Cards** | Total screened, top score, average score, and good matches at a glance |
| **Bar Chart** | Overall match scores per candidate (highlighted ≥ 55%) |
| **Line Chart** | TF-IDF vs. Sentence-BERT comparison across candidates |
| **Radar Chart** | Multi-dimensional top-3 candidate comparison |
| **Candidate Cards** | Ranked cards with Gold / Silver / Bronze badges and per-metric progress bars |

---

## 🤝 Contributing

Contributions are welcome! To get started:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please keep functional logic and UI changes in separate commits.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ using **Streamlit** · **Sentence Transformers** · **scikit-learn**

</div>
