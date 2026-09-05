# Advanced EDA & Feature Engineering — Network Intrusion Detection (UNSW-NB15)

> A statistically rigorous data preprocessing pipeline built on the UNSW-NB15 network intrusion detection dataset, completed as Project 1 of the DecodeLabs Data Science Industrial Training Track.

---

## 📌 Overview

Raw network traffic data is messy, skewed, and full of statistical traps — missing values, extreme outliers, high-cardinality categoricals, and multicollinear features. Feeding this directly into a machine learning model produces unreliable, misleading results, no matter how sophisticated the model is.

This project treats data preprocessing as **engineering, not janitorial work**. Instead of applying generic `dropna()` and `StandardScaler()` calls, every transformation here is chosen based on a measurable property of the data — how much is missing, how extreme an outlier is, how many categories a feature has, and how strongly two features correlate.

The result is a clean, validated, production-style feature set ready for downstream modeling.

---

## 🎯 Objectives

- Transform raw, chaotic UNSW-NB15 data into a mathematically clean dataset
- Handle missing data using statistically justified, threshold-based methods
- Detect and neutralize outliers without destroying dataset volume
- Engineer new features grounded in network-security domain logic
- Remove multicollinearity to protect model stability and interpretability
- Enforce a runtime data contract so silent data corruption cannot pass downstream

---

## 🗂️ Dataset

**UNSW-NB15** — a labeled network intrusion detection dataset created by the Cyber Range Lab of UNSW Canberra, containing real modern normal traffic and synthetic contemporary attack behaviors across 9 attack categories.

| | |
|---|---|
| Source | [Kaggle – UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) |
| Training records | ~175,341 |
| Testing records | ~82,332 |
| Features | 45 (flow, TCP, content, and time-based statistics) |
| Targets | `label` (binary: normal / attack), `attack_cat` (multi-class: 9 attack types + normal) |
| Format used | `.parquet` (binary set, V1/V2) |

> Dataset files are not included in this repository due to size. Download from Kaggle and place `UNSW_NB15_training-set.parquet` and `UNSW_NB15_testing-set.parquet` in the `data/` folder before running the notebook.

---

## 🏗️ Pipeline Architecture

This project follows an **Input → Process → Output** structure, treating the pipeline as a system rather than a single script:

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│   INPUT STAGE    │     │    PROCESS STAGE     │     │     OUTPUT STAGE      │
│  Securing Data   │ --> │   The Transform      │ --> │  Contracts & Export   │
│    Fidelity      │     │       Engine         │     │                       │
├─────────────────┤     ├──────────────────────┤     ├───────────────────────┤
│ Missing value    │     │ Vectorized numeric   │     │ Pandera schema        │
│ decision matrix  │     │ operations           │     │ validation            │
│                  │     │                      │     │                       │
│ IQR-based        │     │ Categorical encoding │     │ Clean parquet export  │
│ outlier bounds   │     │ (cardinality-aware)  │     │                       │
│                  │     │                      │     │                       │
│                  │     │ Multicollinearity    │     │                       │
│                  │     │ eradication          │     │                       │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘
```

---

## 🔧 Methodology

### 1. Missing Value Decision Matrix
Missing data is never handled with a single blanket rule. Each column is routed based on its missingness percentage:

| Missingness | Strategy | Reasoning |
|---|---|---|
| < 5% | Row deletion | Negligible data loss, avoids synthetic bias |
| 5% – 20% | Statistical / sub-group imputation | Preserves distribution shape and sub-population variance |
| > 20% | KNN imputation (k=5) | Captures multi-dimensional relationships standard imputation would miss |

### 2. Outlier Treatment — IQR + Winsorization
Outliers are detected using `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR` boundaries, then **capped (winsorized)** rather than deleted. This preserves row count and sequential integrity — important for a dataset built from network flow records.

### 3. Categorical Encoding
Encoding strategy is chosen by cardinality, not applied uniformly:
- **Low cardinality** (e.g. `state`) → One-Hot Encoding
- **High cardinality** (e.g. `proto`, `service`) → Frequency Encoding, avoiding dimensionality explosion while retaining signal

### 4. Domain-Driven Feature Engineering
Five new features were engineered from network-traffic behavior logic, not arbitrary column math:

| Feature | Formula | Signal it captures |
|---|---|---|
| `byte_ratio` | `sbytes / dbytes` | Traffic asymmetry (common in DoS) |
| `pkt_size_avg_src` | `sbytes / spkts` | Small-packet scanning behavior |
| `total_pkt_rate` | `(spkts + dpkts) / dur` | Abnormal packet flooding rate |
| `tcp_setup_ratio` | `tcprtt / (synack + ackdat)` | Irregular TCP handshake timing |
| `conn_load_ratio` | `sload / dload` | Push-heavy vs. balanced sessions |

### 5. Multicollinearity Eradication
Feature pairs with correlation > 0.85 are identified via the correlation matrix. Rather than dropping the first feature found, each pair is resolved by keeping whichever feature correlates more strongly with the target label.

### 6. Runtime Data Contract
A [Pandera](https://pandera.readthedocs.io/) schema validates the final dataset's types, nullability, and value ranges before export — the same principle used to prevent training-serving skew in production ML systems.

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `SciPy` · `Scikit-learn` · `Pandera`

---

## 📁 Project Structure

```
unsw-nb15-eda-feature-engineering/
├── data/
│   ├── UNSW_NB15_training-set.parquet     # not included — download from Kaggle
│   └── UNSW_NB15_testing-set.parquet      # not included — download from Kaggle
├── notebooks/
│   └── Advanced_EDA_Feature_Engineering_UNSW_NB15.ipynb
├── outputs/
│   └── UNSW_NB15_train_processed.parquet
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/unsw-nb15-eda-feature-engineering.git
cd unsw-nb15-eda-feature-engineering

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
pandera
pyarrow
```

---

## ▶️ Usage

1. Download `UNSW_NB15_training-set.parquet` and `UNSW_NB15_testing-set.parquet` from [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) and place them in `data/`
2. Open `notebooks/Advanced_EDA_Feature_Engineering_UNSW_NB15.ipynb`
3. Run all cells sequentially
4. The cleaned, validated dataset is exported to `outputs/UNSW_NB15_train_processed.parquet`

---

## 📊 Results

- Missing values resolved using a documented, threshold-based decision process rather than uniform treatment
- Outliers capped across all numeric features without reducing dataset size
- 5 new domain-relevant features engineered and validated for predictive value
- Redundant, multicollinear features systematically removed based on target correlation
- Final dataset passes a formal schema validation check before export

---

## 🚀 Future Improvements

- Apply the identical fitted transformations (not refit) to the test set to prevent data leakage
- Wrap the pipeline into a reusable `scikit-learn` `ColumnTransformer` / `Pipeline` object
- Train and benchmark baseline models (Logistic Regression, XGBoost) on the binary `label` and multi-class `attack_cat` targets
- Convert the notebook into a scripted, orchestrated ETL job (e.g. Prefect/Airflow) for recurring data drops
- Integrate a feature store (e.g. Feast) to serve these features consistently for both training and real-time inference

---

## 👤 Author

**Kinza Arshad**
Data Science Undergraduate, KFUEIT
Data Science Industrial Training — DecodeLabs (Batch 2026)

---

## 📄 License

This project is for educational and portfolio purposes as part of the DecodeLabs Data Science Industrial Training program.
