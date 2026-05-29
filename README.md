# COMP3000HK FYP — Email Spam Classification

**Course:** COMP3000HK Computing Project (Final Year Project)  
**Author:** Wong Fong Man  
**Repository:** [carmenwonghkuspace/COMP3000HK-FYP-WongFongMan](https://github.com/carmenwonghkuspace/COMP3000HK-FYP-WongFongMan)

Machine-learning pipeline for **spam vs ham** email classification: TF-IDF features, hyperparameter tuning, model comparison with cross-validation and statistical tests, SHAP explanations, and a browser demo for local scoring.

---

## COMP3000HK report and portfolio

| Document | Path |
|----------|------|
| **Final report (~10,000 words)** | [`docs/COMP3000HK-report.md`](docs/COMP3000HK-report.md) — export to PDF for PebblePad *(kept locally, not pushed to GitHub)* |
| Project initiation (Phase 02) | [`docs/pebblepad/02-project-initiation.md`](docs/pebblepad/02-project-initiation.md) |
| Submission checklist | [`docs/pebblepad-submission-checklist.md`](docs/pebblepad-submission-checklist.md) *(local only)* |
| Poster (A1 HTML → JPG) | [`docs/poster/poster.html`](docs/poster/poster.html) — see [`docs/poster/README.md`](docs/poster/README.md) *(local only)* |
| Video script (5 min) | [`docs/video-script.md`](docs/video-script.md) *(local only)* |
| Viva slides (10 min) | [`docs/viva-slides.md`](docs/viva-slides.md) *(local only)* |
| CPD reflections ×3 | [`docs/cpd-reflections.md`](docs/cpd-reflections.md) *(local only)* |
| GenAI declaration guide | [`docs/appendix-genai-declaration.md`](docs/appendix-genai-declaration.md) *(local only)* |

**Poster export:** `chmod +x scripts/export_poster_jpg.sh && ./scripts/export_poster_jpg.sh YOUR_STUDENT_ID`

---

## Project Overview

This project compares Logistic Regression, Random Forest, and XGBoost by performing hyperparameter tuning on the training set, then evaluating the models using cross-validation and a 30% hold-out split. It also generates visual charts for the report and includes a browser-based web interface for interactive use. The original email CSV dataset is large and is therefore not included in Git; after cloning the repository, please download it into `datasets/`.

---

## Repository layout

```
.
├── main.py                 # End-to-end training pipeline (run from here)
├── project_paths.py        # Root / datasets / result / web path helpers
├── requirements.txt
├── README.md
├── .gitignore
│
├── datasets/               # data.csv (gitignored; see .gitkeep)
├── notebooks/
│   ├── data_pulling.ipynb      # Kaggle download → datasets/data.csv
│   ├── data_processing.ipynb   # EDA / preprocessing experiments
│   └── main.ipynb              # Notebook version of the pipeline
├── scripts/
│   └── export_browser_json_from_pkl.py
├── result/                 # Figures, tables, models (partially gitignored)
└── web/                    # Static spam-scoring UI + model.json
```

---

## Setup

### 1. Python environment

Python **3.10+** recommended. Use a normal venv or conda env **outside** the repo (do not commit `.conda/`):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Dataset

Place the training CSV at:

```text
datasets/data.csv
```

**Download (first time):** open `notebooks/data_pulling.ipynb` and run all cells. It fetches  
[The Biggest Spam/Ham/Phish Email Dataset (~300k)](https://www.kaggle.com/datasets/akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000)  
via `kagglehub` and writes `datasets/data.csv`.

Labels in `main.py`: class `2` → spam, `0` → ham; rows with label `1` are dropped.

---

## Quick start

All commands below assume the **repository root** as the current directory.

### Train full pipeline

```bash
python main.py
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `FAST_RUN` | `1` | Faster subsample & smaller search |
| `FAST_RUN=0` | — | Larger sample & more tuning iterations |
| `TUNE_MODELS` | `1` | Grid / random search on train split |
| `TUNE_MODELS=0` | — | Fixed default hyperparameters |
| `N_JOBS_CV` | auto | Parallel CV jobs (use `1` if GPU OOM) |

```powershell
# Heavier run (PowerShell)
$env:FAST_RUN="0"; python main.py
```

**Outputs:** `result/` (figures, CSV tables, pickles) and `web/model.json`.

### Web demo

```bash
cd web
python -m http.server 8765
```

Open http://127.0.0.1:8765 — paste text and click **Score & explain**.
Website:https://filedn.eu/lVpslS7sgTLhBMo6HtIW6PJ/web/index.html

### Export browser JSON only

```bash
python scripts/export_browser_json_from_pkl.py
```

---

## Pipeline stages (`main.py`)

| Stage | Description | Example outputs |
|-------|-------------|-----------------|
| 1 | Load & EDA | `fig1_class_distribution.png` |
| 2 | 70/30 stratified split (hold-out locked) | — |
| 3 | Text cleaning + TF-IDF | `tfidf_vectorizer.pkl`, sparse matrix |
| 4 | **Hyperparameter tuning** (train only) | `table0_tuning_results.csv` |
| 5 | Stratified CV + permutation tests | `table1_*.csv`, `fig2_*.png` |
| 6 | Refit best model on full train | `best_model.pkl` |
| 7 | Hold-out evaluation + bootstrap CI | `table6_*.csv`, `fig3`–`fig5` |
| 8 | Interpretability (OR, SHAP) | `table2`–`table5`, `fig6`–`fig8` |
| 9 | Browser pack | `web/model.json` |

Hold-out data is **not** used for tuning or model selection.

---

## What is tracked in Git

| Tracked | Ignored (see `.gitignore`) |
|---------|----------------------------|
| `main.py`, `project_paths.py`, notebooks, `web/` UI | `datasets/*.csv` |
| Summary CSVs & PNG figures in `result/` | `result/*.pkl`, `processed_tfidf.csv` |
| `web/model.json` (demo weights) | `.conda/`, `.vscode/`, `__pycache__/` |

After clone you can run the web demo immediately; retrain with `python main.py` once `datasets/data.csv` exists.

---

## Notebooks

Run Jupyter with the **project root** as the working directory (or use the first cell in each notebook, which `chdir`s to the root).

| Notebook | Role |
|----------|------|
| `notebooks/data_pulling.ipynb` | Download Kaggle data |
| `notebooks/data_processing.ipynb` | Exploration / preprocessing |
| `notebooks/main.ipynb` | Interactive pipeline |

---

## License & citation

Academic work for COMP3000HK (HKU SPACE). If you use the Kaggle dataset, follow its license and cite the dataset on Kaggle.
