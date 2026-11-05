# 🩺 Chronic Kidney Disease (CKD) Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Clinical Machine Learning diagnostic pipeline designed to detect the presence or absence of **Chronic Kidney Disease (CKD)** from multidimensional patient health profiles.

The system incorporates robust clinical preprocessing via `ColumnTransformer` (median numerical imputation and one-hot categorical encoding) paired with a balanced **Random Forest Ensemble Classifier** to handle feature heterogeneity and clinical class imbalances.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Architecture](#-project-architecture)
- [Clinical Feature Catalog](#-clinical-feature-catalog)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Evaluation & Clinical Metrics](#-model-evaluation--clinical-metrics)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Sample Execution & Output](#-sample-execution--output)
- [Medical Disclaimer & License](#-medical-disclaimer--license)

---

## 🌟 Overview

Chronic Kidney Disease is a major global health condition characterized by the gradual loss of kidney function over time. Because early-stage CKD frequently presents without noticeable symptoms, diagnostic machine learning systems can assist healthcare professionals in identifying high-risk individuals using routine clinical labs, vital signs, and comorbidity markers.

This project delivers:
- **Clinical Data Leakage Prevention:** Eliminates direct proxies such as `ckd_stage` and unique identifiers to guarantee real-world generalization.
- **Dual-Branch Automated Preprocessing:** Treats laboratory metrics and patient demographics using specialized transformation strategies.
- **Balanced Ensemble Learning:** Uses 200 decision trees with cost-sensitive class balancing to penalize misclassifications of CKD-positive patients.
- **Reproducible Pipeline:** Combines preprocessing and classification into a unified Scikit-Learn `Pipeline` preventing data leakage during training and validation.

---

## 🚀 Key Features

- **Automated Feature Type Detection:** Dynamically separates numerical laboratory tests from categorical demographic and comorbidity variables.
- **Robust Imputation Strategy:**
  - Median imputation for numerical features to resist clinical outlier skewness.
  - Most-frequent imputation for categorical variables with unknown-category tolerance.
- **One-Hot Encoding:** Encodes ethnicity, gender, and clinical flags into binary indicators.
- **Random Forest Ensemble (200 Estimators):**
  - Reduces variance through bagging and random feature subspace sampling.
  - Built-in `class_weight="balanced"` to prevent majority-class bias.
- **Stratified Evaluation:** Ensures that the 80/20 train/test split maintains identical proportions of positive and negative CKD cases.
- **Comprehensive Evaluation:** Generates Accuracy, Precision, Recall, F1-Score, and a complete Confusion Matrix.

---

## 🏗️ Project Architecture

```
 Patient Health Records Dataset (10,000 Rows, 29 Features)
                         │
                         ▼
 Target Validation & Data Leakage Prevention (Drop ID, Stage, Target)
                         │
                         ▼
 Stratified 80/20 Train-Test Split (Preserving CKD prevalence)
                         │
                         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │               Scikit-Learn ColumnTransformer                 │
 │                                                              │
 │  Numerical Pipeline:                                         │
 │    SimpleImputer (Strategy = "median")                       │
 │                                                              │
 │  Categorical Pipeline:                                       │
 │    SimpleImputer (Strategy = "most_frequent")                │
 │    OneHotEncoder (handle_unknown = "ignore")                 │
 └──────────────────────────────────────────────────────────────┘
                         │
                         ▼
 Random Forest Classifier (200 Trees, class_weight="balanced")
                         │
                         ▼
 Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

---

## 📋 Clinical Feature Catalog

The model is trained on `CKD_10000_sample_synthetic.csv`, containing 10,000 comprehensive patient records across 29 features:

### 1. Patient Demographics & Vitals
| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical | Patient age in years |
| `gender` | Categorical | Male or Female |
| `ethnicity` | Categorical | Non-Hispanic White, Black, Mexican American, Other Hispanic, etc. |
| `education_level` | Numerical | Highest education level completed |
| `poverty_income_ratio`| Numerical | Ratio of family income to poverty threshold |
| `bmi` | Numerical | Body Mass Index ($\text{kg/m}^2$) |
| `weight_kg` / `height_cm` | Numerical | Anthropometric measurements |
| `bp_systolic` / `bp_diastolic` | Numerical | Systolic and diastolic blood pressure (mmHg) |

### 2. Blood Biomarkers & Chemistry
| Feature | Type | Description | Clinical Significance |
| :--- | :--- | :--- | :--- |
| `serum_creatinine` | Numerical | Serum creatinine (mg/dL) | Primary indicator of renal clearance |
| `bun` | Numerical | Blood Urea Nitrogen (mg/dL) | Measures nitrogenous waste accumulation |
| `serum_albumin` | Numerical | Serum albumin (g/dL) | Assesses liver and kidney protein retention |
| `phosphorus` | Numerical | Blood phosphorus level | Regulated by renal excretion |
| `bicarbonate` | Numerical | Serum bicarbonate level | Monitors metabolic acidosis |
| `calcium` | Numerical | Blood calcium level | Mineral and bone disorder marker |
| `uric_acid` | Numerical | Serum uric acid | Elevated in impaired filtration |

### 3. Urinalysis & Renal Indices
| Feature | Type | Description | Clinical Significance |
| :--- | :--- | :--- | :--- |
| `urine_albumin` | Numerical | Urine albumin concentration | Detects glomerular membrane leakage |
| `urine_creatinine` | Numerical | Urine creatinine concentration | Normalizes urinary excretion |
| `uacr` | Numerical | Urine Albumin-to-Creatinine Ratio | Gold standard marker for microalbuminuria |
| `egfr` | Numerical | Estimated Glomerular Filtration Rate | Core measure of overall kidney filtration |

### 4. Lifestyle & Comorbidity Indicators
| Feature | Type | Description |
| :--- | :--- | :--- |
| `diabetes_diagnosed` | Categorical/Binary | Diagnosed diabetic condition (0 = No, 1 = Yes) |
| `insulin_use` | Categorical/Binary | Current insulin therapy |
| `diabetes_pills` | Categorical/Binary | Oral hypoglycemic medication |
| `ever_smoked` / `current_smoker` | Categorical/Binary | Tobacco smoking history |

### 5. Target Variable
| Variable | Values | Meaning |
| :--- | :--- | :--- |
| `ckd_present` | `0` or `1` | **0** = CKD Absent, **1** = CKD Present |

*(Note: `participant_id` and `ckd_stage` are automatically excluded during training to eliminate identifier bias and prevent direct target leakage).*

---

## 🧠 Machine Learning Pipeline

### 1. Missing Value Strategy
Clinical datasets frequently contain missing laboratory records due to ordered test variability:
- **Numerical Features:** Imputed using the **median**, which is resilient against extreme diagnostic outliers.
- **Categorical Features:** Imputed using the **most frequent** category, preserving dominant distribution patterns.

### 2. Random Forest Classification
The classifier aggregates decisions from $B = 200$ bootstrap trees:
$$\hat{y} = \text{mode}\left\{ h_b(\mathbf{x}) \right\}_{b=1}^{B}$$

- **Class Balancing:** Class weights are assigned inversely proportional to class frequencies:
  $$w_j = \frac{N}{K \cdot n_j}$$
  This prevents the model from ignoring the minority CKD-positive cohort.

---

## 📈 Model Evaluation & Clinical Metrics

In medical diagnostic screening, balancing Precision and Recall is paramount:
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Minimizes false negatives, ensuring individuals with kidney damage are not missed.
- **Precision (Positive Predictive Value):** $\frac{TP}{TP + FP}$ — Minimizes unnecessary secondary referrals.
- **F1-Score:** Harmonic balance between Sensitivity and Precision.
- **Confusion Matrix:** Evaluates:
  $$\begin{bmatrix} \text{True Negatives (TN)} & \text{False Positives (FP)} \\ \text{False Negatives (FN)} & \text{True Positives (TP)} \end{bmatrix}$$

---

## 📁 Repository Structure

```
Kidney-disease-predition/
│
├── Kidney disease.py                 # Main executable Python training script
├── CKD_10000_sample_synthetic.csv     # Clinical dataset (10,000 patient records)
├── requirements.txt                  # Python dependencies list
├── .gitignore                        # Git ignore rules for clean repository
└── README.md                         # Comprehensive clinical & technical documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MNithinreddy777/Kidney-disease-predition.git
cd Kidney-disease-predition
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

Execute the main training script:
```bash
python "Kidney disease.py"
```

### Execution Steps:
1. Loads `CKD_10000_sample_synthetic.csv` and prints the initial dataset preview and dimensionality.
2. Removes missing target instances and excludes identifier/leakage columns (`participant_id`, `ckd_stage`).
3. Automatically categorizes 20+ numerical lab markers and categorical demographic features.
4. Builds the `ColumnTransformer` preprocessing pipeline and initializes the 200-tree Random Forest.
5. Performs an 80/20 stratified split and trains the ensemble model.
6. Outputs training/testing sample counts, Accuracy, detailed Classification Report, and Confusion Matrix.

---

## 🖥️ Sample Execution & Output

```text
CHRONIC KIDNEY DISEASE DATASET
   participant_id  age  gender          ethnicity  education_level  ...  ckd_present  ckd_stage
0          100001   68    Male     Other Hispanic              3.0  ...            0          0
1          100002   35  Female  Non-Hispanic White             3.0  ...            0          0
2          100003   43    Male  Non-Hispanic White             5.0  ...            0          0
3          100004   55    Male  Non-Hispanic White             4.0  ...            0          0
4          100005   60  Female     Other Hispanic              4.0  ...            0          0

[5 rows x 29 columns]

Dataset shape: (10000, 29)

CKD PREDICTION RESULTS
Training samples: 8000
Testing samples: 2000

Accuracy: 0.985

Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1802
           1       0.93      0.91      0.92       198

    accuracy                           0.98      2000
   macro avg       0.96      0.95      0.96      2000
weighted avg       0.98      0.98      0.98      2000

Confusion Matrix:
[[1788   14]
 [  17  181]]

Class labels: 0 = CKD absent, 1 = CKD present
```

---

## ⚕️ Medical Disclaimer & License

> [!CAUTION]
> **Clinical Notice:** This project is developed for educational, benchmarking, and machine learning research purposes. It is not intended to provide clinical medical diagnoses or replace licensed clinical advice, professional pathology assessments, or personalized medical evaluations.

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
