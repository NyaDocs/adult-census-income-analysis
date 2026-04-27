# Adult Census Income Analysis

## 📌 Overview
An end-to-end machine learning project that uses 1994 US Census data to predict whether an individual earns above $50K/year. The analysis goes beyond model accuracy — it translates statistical findings into actionable recommendations for HR professionals, workforce planners, and policy makers.

## ❓ Business Question
> *What demographic and occupational factors best predict whether an individual earns above $50K/year — and what should HR teams, policy makers, or workforce planners act on?*

## 🗂️ Dataset
- **Source:** [UCI Machine Learning Repository — Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult)
- **Origin:** Extracted from the 1994 US Census database by Barry Becker
- **Size:** 48,842 rows × 15 columns
- **Key columns:**

| Column | Type | Description |
|--------|------|-------------|
| `age` | Integer | Individual's age |
| `education` | Categorical | Highest education level attained |
| `education-num` | Integer | Education level as a numeric score |
| `occupation` | Categorical | Type of work |
| `sex` | Binary | Male / Female |
| `hours-per-week` | Integer | Weekly working hours |
| `capital-gain/loss` | Integer | Investment income/loss |
| `income` | Target | `>50K` or `<=50K` |

## 🔧 Methodology

```
Data Loading (ucimlrepo)
  → Cleaning (whitespace, '?' → NaN, mode imputation, deduplication)
  → EDA (distributions, gender gap, education × income, occupation rates)
  → Feature Engineering (label encoding, scaling, train/test split)
  → Modeling (Logistic Regression · Random Forest)
  → Evaluation (AUC-ROC, classification report, confusion matrix, feature importance)
  → Business Insight (plain-language recommendations for stakeholders)
```

## 📊 Key Findings

- **Education is the strongest single predictor.** Individuals with Doctorate or Professional School degrees show the highest rates of earning >$50K. HS graduates — the largest group — are overwhelmingly below $50K.
- **A significant and persistent gender gap exists.** Male workers earn >$50K at approximately 3× the rate of female workers, even when controlling for education level — indicating occupational segregation or pay inequity.
- **Occupation drives income as strongly as education.** Exec-Managerial and Prof-Specialty roles dominate the high-income bracket; Other-service and Priv-house-serv are almost entirely below $50K.
- **High earners peak between ages 35–55.** Workers under 30 are nearly uniformly in the <=50K bracket, pointing to structured early-career progression as a key lever.
- **Random Forest outperforms Logistic Regression** (Test AUC ~0.92 vs ~0.90), confirming that income is driven by non-linear interactions between features. The model is viable for production use in HR or policy tools.

## 🛠️ Tools Used
Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn · Jupyter Notebook

## 📁 Repository Structure

```
adult-census-income-analysis/
│
├── data/
│   └── .gitkeep              # Dataset is fetched via ucimlrepo — no CSV needed
│
├── notebooks/
│   └── adult_income_analysis.ipynb   # Full storytelling notebook
│
├── scripts/
│   └── adult_income_analysis.py      # Clean, reusable Python script
│
├── visuals/                           # All exported charts (auto-generated)
│   ├── 01_income_distribution.png
│   ├── 02_age_distribution.png
│   ├── 03_education_vs_income.png
│   ├── 04_gender_income_gap.png
│   ├── 05_occupation_income_rate.png
│   ├── 06_correlation_heatmap.png
│   ├── 07_roc_curves.png
│   ├── 08_confusion_matrix_rf.png
│   └── 09_feature_importance.png
│
├── README.md
├── requirements.txt
├── setup_repo.py              # One-time scaffold script
└── .gitignore
```

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/NyaDocs/adult-census-income-analysis.git
cd adult-census-income-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3a. Run the Python script**
```bash
cd scripts
python adult_income_analysis.py
```
All charts will be saved to `visuals/` automatically.

**3b. Or open the Jupyter Notebook**
```bash
jupyter notebook notebooks/adult_income_analysis.ipynb
```
Run all cells top to bottom. Charts render inline and are also saved to `visuals/`.

> **Note:** The dataset is fetched automatically via `ucimlrepo` — no manual download required.

---

## 📜 Citation

```
Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20
```

---

*Project by [Rahmadhania](https://github.com/YOUR_USERNAME) · April 2026*
