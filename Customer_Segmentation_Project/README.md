# 🎯 DecodeLabs – Data Science Project 3  
## Unsupervised Learning: Customer Segmentation

**Industrial Training Kit · Batch 2026 · Powered by DecodeLabs**

---

### Project Goal
Use distance-based algorithms to discover hidden mathematical groupings in unlabeled retail data, reduce high-dimensional feature space with PCA, prove the optimal number of clusters with the Elbow Method + Silhouette Score, and translate raw clusters into actionable business **Personas**.

---

### IPO Architecture (as defined in the kit)

```
1. SCALE        →  Standardization (StandardScaler)
2. COMPRESS     →  Principal Component Analysis (95 % variance rule)
3. CLUSTER      →  K-Means (optimal K via Elbow + Silhouette)
4. TRANSLATE    →  Inverse transforms → Business Personas
```

---

### Key Requirements Covered

| Requirement | Implementation |
|-------------|----------------|
| PCA on 20+ features → 2/3 (+ more) dimensions | ✅ Automatic component selection by cumulative explained variance ≥ 95 % |
| Elbow Method | ✅ WCSS + KneeLocator (maximum curvature) |
| Silhouette Score | ✅ Full evaluation + recommended K |
| Actionable Business Personas | ✅ Reverse-engineered centroids + strategic action matrix |
| Clean modular Python | ✅ Full package structure under `src/` |
| Professional file structure | ✅ See below |

---

### Project Structure

```
customer_segmentation_project/
├── app.py                      # Interactive Streamlit dashboard
├── main.py                     # CLI entry point
├── requirements.txt
├── README.md
├── configs/
│   └── config.yaml
├── data/
│   ├── marketing_campaign.csv  # Customer Personality Analysis dataset (~2240 rows, 29 cols)
│   └── processed_customers.csv # Generated after feature engineering
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Loading + cleaning + feature engineering (20+ features)
│   ├── preprocessing.py        # StandardScaler + PCA (with inverse)
│   ├── clustering.py           # OptimalKFinder + KMeans + algorithm comparison
│   ├── personas.py             # Centroid inverse mapping + Persona generation
│   ├── visualization.py        # Elbow, PCA variance, 2D/3D, radar, profiles
│   └── pipeline.py             # End-to-end orchestration
├── models/                     # Saved scaler, PCA, KMeans (joblib)
├── outputs/
│   ├── plots/                  # All static & interactive plots
│   ├── k_evaluation.csv
│   ├── algorithm_comparison.csv
│   ├── centroids_original_space.csv
│   ├── cluster_profiles.csv
│   └── customers_with_segments.csv
└── reports/
    └── persona_matrix_*.md     # Strategic Persona Matrix report
```

---

### Dataset
**Customer Personality Analysis** (public Kaggle dataset)  
- ~2 240 customers  
- Demographics, product spending (6 categories), purchase channels, campaign responses, etc.  
- After feature engineering → **25+ numeric features** (Age, Income, Total_Spent, Tenure, Campaign rates, family size, education level, …)

---

### Quick Start

```bash
# 1. Create environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py

# Optional flags
python main.py --k 4              # force K=4
python main.py --variance 0.90    # lower PCA threshold
python main.py --kmin 2 --kmax 8

# 4. Launch interactive dashboard
streamlit run app.py
```

---

### What the Pipeline Produces

1. **Diagnostics**
   - Elbow curve + Silhouette scores for K = 2…10
   - Automatic recommended K

2. **Dimensionality Reduction**
   - PCA components that retain ≥ 95 % variance
   - Variance explained plot

3. **Clustering**
   - Final K-Means model
   - Side-by-side comparison with Agglomerative & DBSCAN

4. **Business Personas**
   - Centroids mapped back to original feature space
   - Named personas with descriptions & recommended marketing actions
   - Markdown report ready for stakeholders

5. **Visualizations**
   - Static PNGs (Elbow, PCA, 2-D clusters, feature profiles)
   - Interactive 3-D Plotly HTML
   - Full Streamlit dashboard

---

### Enhancements Beyond the Base Requirements

- Modular, production-style package layout
- Automatic feature engineering that expands the original columns into 20+ predictive features
- Multiple clustering algorithms comparison
- Interactive Streamlit studio (change PCA threshold / K range on the fly)
- 3-D PCA visualization
- Full inverse-transform path for interpretability
- Joblib model persistence
- Detailed logging & reproducible random seeds
- Downloadable segmented customer file

---

### Example Personas (typical output)

| Cluster | Persona | Typical Traits | Suggested Actions |
|---------|---------|----------------|-------------------|
| 0 | 💎 High-Value Trendsetters | High income + high spend | Exclusive perks, early access, experiential marketing |
| 1 | 🏦 Affluent Conservatives | High income, low spend | High-touch support, warranties, loyalty programs |
| 2 | 🛍️ Budget-Conscious Explorers | Younger, deal-oriented | Influencer campaigns, flash sales, BNPL |
| 3 | 📦 Conservative Minimizers | Older, price-sensitive | Clear value messaging, basic utility focus |

*(Exact names and statistics are data-driven and may vary slightly with different random seeds or K.)*

---

### Skills Demonstrated

- Dimensionality reduction (PCA)
- Distance-based clustering (K-Means)
- Model selection diagnostics (Elbow, Silhouette, CH, DB)
- Inverse transformation for business interpretability
- End-to-end ML engineering practices
- Interactive data-product development (Streamlit)

---

**DecodeLabs · Greater Lucknow, India**  
`decodelabs.tech@gmail.com` · `+91 92360 11887` · [www.decodelabs.tech](https://www.decodelabs.tech)
