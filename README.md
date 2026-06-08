# 🫀 Organ Match AI

> AI-powered organ transplant matching system combining optimization algorithms, graph theory, machine learning, and survival analysis for fair and accurate donor-recipient allocation.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow)](YOUR_COLAB_LINK_HERE)

---

## 🧠 What This Project Does

Organ transplant matching is a life-or-death decision. Every year thousands of patients die waiting for a compatible donor. Wrong matching causes organ rejection, wasting a scarce resource.

This system uses **AI + optimization algorithms** to:
- Find the best donor-recipient match from thousands of combinations
- Predict organ rejection risk with ML models
- Predict 1-year and 5-year graft survival probability
- Explain every decision to doctors using feature importance
- Handle incompatible pairs through Kidney Exchange (KEP)

---

## 🏗️ System Architecture

```
Donor Data + Recipient Data
          ↓
  Compatibility Engine
  (ABO + HLA + AHP + TOPSIS)
          ↓
  Matching Algorithms
  (Gale-Shapley / Hungarian / Bipartite / KEP)
          ↓
  ML Risk Prediction
  (XGBoost + Survival Analysis)
          ↓
  Ranked Recommendations
  + Rejection Risk Score
  + Survival Probability
  + Feature Importance
```

---

## ⚙️ Algorithms Used

### Matching Algorithms

| Algorithm | Purpose | Guarantee |
|---|---|---|
| Gale-Shapley | Stable donor-recipient matching | No blocking pairs — fairest outcome |
| Hungarian | Global cost optimization | Minimizes total rejection risk across all matches |
| Bipartite Max Matching | Graph-based maximum matching | Maximizes total compatibility |
| Kidney Exchange (KEP) | Cross-exchange for incompatible pairs | Saves patients who would otherwise have no match |

### Scoring & Decision Support

| Method | Purpose |
|---|---|
| AHP (Analytic Hierarchy Process) | Multi-criteria weighted scoring — blood type, HLA, urgency, age, distance |
| TOPSIS | Final ranking by similarity to ideal solution |

### Machine Learning Models

| Model | Purpose | Why |
|---|---|---|
| XGBoost (Baseline) | Rejection risk prediction | Industry standard for tabular healthcare data |
| XGBoost (Optuna Tuned) | Optimized rejection risk | Bayesian hyperparameter tuning — best performance |
| XGBoost (RF Mode) | Ensemble baseline comparison | Simulates Random Forest using XGBoost's parallel trees |
| Kaplan-Meier | Survival curve estimation | Standard in clinical research — 1yr/5yr graft survival |
| Cox Proportional Hazards | Survival regression | Identifies which features most affect survival time |

---

## 📊 Results

### Matching Performance (20 donors × 50 recipients)

| Algorithm | Matches | Utilization | Avg AHP Score |
|---|---|---|---|
| Gale-Shapley | 18 | 90.0% | 0.6158 |
| Hungarian | 19 | 95.0% | 0.6291 |
| Bipartite | 19 | 95.0% | 0.6291 |
| KEP | 1 exchange | 2 pairs helped | — |

### ML Model Performance

| Model | AUC-ROC | F1 Score | Brier Score |
|---|---|---|---|
| XGBoost Baseline | > 0.82 | > 0.78 | < 0.18 |
| XGBoost Tuned | > 0.84 | > 0.80 | < 0.16 |
| XGBoost RF Mode | > 0.80 | > 0.76 | < 0.20 |

### Survival Predictions
- **1-year graft survival probability**: ~78%
- **5-year graft survival probability**: ~52%
- **Concordance index**: > 0.65

> ✅ Models trained on synthetic data calibrated with real OPTN/UNOS distributions  
> ✅ Validated against NHS Organ Donation Kaggle dataset

---

## 📁 Project Structure

```
organ-match-ai/
│
├── data/
│   ├── donors.csv              # Synthetic donor dataset (500 records)
│   ├── recipients.csv          # Synthetic recipient dataset (1000 records)
│   ├── generate_data.py        # Dataset generator (OPTN distributions)
│   └── kaggle/
│       ├── NHS_Organ_Donation.csv
│       └── Organ_Transplant.csv
│
├── matching/
│   └── compatibility_scorer.py # ABO + HLA + AHP + TOPSIS scorer
│
├── algorithms/
│   └── matching_algorithms.py  # Gale-Shapley, Hungarian, Bipartite, KEP
│
├── ml_models/
│   └── rejection_risk_model.py # XGBoost + Survival analysis
│
├── evaluation/                 # Accuracy metrics, comparison plots
├── dashboard/                  # Streamlit UI
├── results/                    # Saved metrics, model outputs
├── tests/                      # Unit tests
├── Notes.md                    # Development notes + validation plan
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Local Setup
```bash
git clone https://github.com/kmammu-stack/organ-match_ai-and-ml.git
cd organ-match_ai-and-ml

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Generate Dataset
```bash
python data/generate_data.py
```

### Run Compatibility Scorer
```bash
python matching/compatibility_scorer.py
```

### Run Matching Algorithms
```bash
python algorithms/matching_algorithms.py
```

### Run ML Models
```bash
python ml_models/rejection_risk_model.py
# Or run on Google Colab:(https://colab.research.google.com/drive/1x9khMwmn-e0oLIBzI5-RQUv3CMcc3RLm?usp=sharing)
```

---

## 📦 Tech Stack

| Category | Libraries |
|---|---|
| ML & Optimization | XGBoost, Optuna, NumPy |
| Survival Analysis | Lifelines (Colab) |
| Graph Algorithms | NetworkX, SciPy |
| Data | Pandas, Faker |
| Explainability | XGBoost Feature Importance, SHAP (Colab) |
| Dashboard | Streamlit, Plotly |
| Version Control | Git, GitHub |

---

## 🧪 Dataset

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| Synthetic Donors | Generated (OPTN distributions) | 500 records | Training |
| Synthetic Recipients | Generated (OPTN distributions) | 1000 records | Training |
| NHS Organ Donation | Kaggle | ~5,000 records | Validation |
| Organ Transplant Dataset | Kaggle (fkshaikh) | ~3,000 records | Validation |

**Why synthetic + real?**  
Patient-level HLA and crossmatch data is not publicly available without a formal UNOS research agreement. We generated synthetic data using real OPTN blood type distributions, then validated model outputs against NHS Kaggle data to confirm generalizability.

---

## 🎯 Key Design Decisions

**Why Gale-Shapley over greedy matching?**  
Greedy matching picks the locally best pair but can leave better global matches unused. Gale-Shapley guarantees stability — no donor-recipient pair would both prefer each other over their current match. This is exactly how the NRMP (National Resident Matching Program) works in the US.

**Why survival model instead of plain classification?**  
Predicting "rejection: yes/no" loses time information. Cox Proportional Hazards predicts *when* failure is likely — giving doctors a 1-year and 5-year probability. This is what clinical transplant programs actually use.

**Why XGBoost over deep learning?**  
Tabular medical data has <10,000 samples. XGBoost consistently outperforms neural networks on structured tabular data at this scale. Deep learning would overfit here.

**Why TOPSIS on top of AHP?**  
AHP assigns weights but TOPSIS provides a normalized ranking score by measuring distance from the ideal solution — giving a cleaner 0-1 score that's intuitive for doctors.

---

## 📈 Validation Strategy

```
Train on synthetic data
        ↓
Save baseline metrics (AUC, F1, Brier)
        ↓
Load Kaggle datasets
        ↓
KS test: compare synthetic vs real distributions
        ↓
Re-run models on Kaggle-adapted data
        ↓
Compare metrics: AUC drop < 0.05 = generalizes well
        ↓
Monte Carlo: 100 subsamples → mean ± std
```

---

## 🔬 Interview Talking Points

1. **Stable matching guarantee** — Gale-Shapley produces zero blocking pairs. Every match is fair by mathematical proof.
2. **Survival > classification** — Cox PH model gives time-to-event probability, not just binary rejection label. Clinically more meaningful.
3. **SHAP explainability** — Doctors need to know *why* a match scored 73% risk. HLA mismatch contributed +23%, PRA contributed +18%. Explainability is required by healthcare AI regulations.
4. **Synthetic + Kaggle validation** — Training on synthetic data with real distributions, validating on Kaggle proves the model generalizes beyond training data.
5. **KEP cross-exchange** — Kidney Exchange is actually used by UNOS in the US. Finding cycle exchanges in a directed graph is the real algorithm behind real transplant programs.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👩‍💻 Author

**Padmavathi** — [@kmammu-stack](https://github.com/kmammu-stack)

> Built as part of an AI/ML portfolio project combining healthcare domain knowledge with production-grade ML engineering.
