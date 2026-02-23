# 🌌 Exoplanet ML — A Comparative Machine Learning Pipeline for Exoplanet Classification

> **An independent research project** applying 10 machine learning implementations to NASA Kepler Space Telescope data for automated exoplanet candidate classification.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f89939?logo=scikitlearn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-brightgreen)](https://xgboost.ai)
[![NASA Data](https://img.shields.io/badge/Data-NASA%20Exoplanet%20Archive-0b3d91)](https://exoplanetarchive.ipac.caltech.edu)

---

## 📖 About

This project implements a full machine learning pipeline on real NASA data to answer the question:

> *"Can machine learning reliably distinguish genuine exoplanet transits from false positives in Kepler photometric data — and which algorithms work best?"*

**XGBoost wins with 92.09% accuracy and AUC = 0.9763** on 7,326 labeled Kepler signals.

The project covers all major ML paradigms — regression, classification, clustering, ensemble methods, and deep learning — applied to a single, scientifically meaningful astrophysics problem.

---

## 🗂️ Project Structure

```
exoplanet_ml/
├── data/
│   ├── koi_cumulative.csv        # NASA KOI table — 9,564 Kepler signals
│   └── ps_composite.csv          # NASA Planetary Systems — 6,107 confirmed planets
├── notebooks/
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_naive_bayes.ipynb
│   ├── 06_svm.ipynb
│   ├── 07_knn.ipynb
│   ├── 08_gradient_boosting.ipynb
│   ├── 09_cnn_lightcurves.ipynb
│   └── 10_synthetic_comparison.ipynb
├── outputs/
│   ├── plots/                    # All 12 generated figures
│   ├── models/                   # Saved CNN model (.keras)
│   └── results/                  # Metrics and result CSVs
├── paper/
│   └── research_paper.docx       # Full independent research paper
└── venv/                         # Python virtual environment
```

---

## 📊 Results Summary

| # | Algorithm | Accuracy | ROC-AUC |
|---|-----------|----------|---------|
| 1 | Linear Regression | R²=0.23 | — |
| 2 | Logistic Regression | 74.01% | 0.8202 |
| 3 | K-Means Clustering | 2 stellar clusters | — |
| 4 | Decision Tree | 87.59% | 0.8800 |
| 5 | Naive Bayes | 71.01% | 0.8802 |
| 6 | SVM (RBF Kernel) | 84.45% | 0.9303 |
| 7 | k-NN (k=13) | 85.20% | 0.9228 |
| 8 | **XGBoost** | **92.09%** | **0.9763** |
| 9 | CNN (1D, light curves) | 54.17% | 0.6597 |
| 10 | Synthetic Comparison | All 6 models benchmarked | — |

**Top discriminating features:** `koi_prad` (planet radius), `koi_model_snr` (signal-to-noise), `koi_period` (orbital period)

---

## 🛰️ Datasets

Both datasets sourced directly from the **NASA Exoplanet Archive** — [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)

| Dataset | Source | Used In |
|---------|--------|---------|
| KOI Cumulative Table | Kepler → KOI Table (Cumulative list) | Programs 2–10 |
| Planetary Systems Composite | PSCompPars table | Program 1 |
| Kepler Light Curves | Fetched via `lightkurve` from NASA MAST | Program 9 |

---

## ⚙️ Setup

### Prerequisites
- Ubuntu / Debian Linux (or WSL on Windows)
- Python 3.12+
- Internet connection (for light curve fetching in Program 9)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/exoplanet_ml.git
cd exoplanet_ml
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow keras lightkurve astropy jupyter notebook
```

### 4. Download datasets
Go to [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/index.html) → Data section and download:
- **KOI Cumulative Table** → save as `data/koi_cumulative.csv`
- **Planetary Systems Composite Parameters** → save as `data/ps_composite.csv`

### 5. Launch Jupyter
```bash
jupyter notebook
```
Open your browser at `http://localhost:8888` and navigate to `notebooks/`.

---

## Running the Notebooks

Run notebooks in order — each is self-contained but builds on the same dataset:

```
01 → Linear Regression        (ps_composite.csv)
02 → Logistic Regression       (koi_cumulative.csv)
03 → K-Means Clustering        (koi_cumulative.csv)
04 → Decision Tree             (koi_cumulative.csv)
05 → Naive Bayes               (koi_cumulative.csv)
06 → SVM                       (koi_cumulative.csv)
07 → k-NN                      (koi_cumulative.csv)
08 → Gradient Boosting         (koi_cumulative.csv)
09 → CNN on Light Curves       (NASA MAST via lightkurve — needs internet)
10 → Synthetic Comparison      (generated in code)
```

>  **Note for Notebook 9:** Fetching 120 light curves from NASA MAST takes 5–15 minutes depending on internet speed. The notebook prints progress every 20 fetches.

---

## 🔬 Key Scientific Findings

- **XGBoost (92.09%)** is the best model for automated KOI vetting
- **Planet radius** (`koi_prad`) is the single most important feature — giant radii indicate false positives (eclipsing binaries)
- **K-Means** without supervision recovered real stellar spectral classes (F-type subgiants vs G-type dwarfs)
- **CNN underperforms** on small, unphase-folded datasets — deep learning needs 5,000+ phase-folded samples to beat classical ML (cf. AstroNet, Shallue & Vanderburg 2018)
- **Naive Bayes (71%)** fails because exoplanet features violate the independence assumption — transit depth and planet radius are physically correlated

---

## 📦 Dependencies

```
numpy >= 2.4
pandas >= 3.0
matplotlib >= 3.10
seaborn >= 0.13
scikit-learn >= 1.8
xgboost >= 3.2
tensorflow >= 2.20
keras >= 3.13
lightkurve >= 2.5
astropy >= 7.2
scipy >= 1.17
jupyter
notebook
```

---

## 📄 Research Paper

A full independent research paper is included in `paper/research_paper.docx`:

> **"A Comparative Machine Learning Pipeline for Exoplanet Candidate Classification Using NASA Kepler Photometric and Stellar Data"**
> Sayed Umair Ali, Department of Computer Science and Engineering, Ajay Binay Institute of Technology, 2026

---

##  References

1. Shallue & Vanderburg (2018) — *Identifying Exoplanets with Deep Learning* — AJ 155(2):94
2. Thompson et al. (2018) — *Planetary Candidates Observed by Kepler VIII* — ApJS 235(2):38
3. Borucki et al. (2010) — *Kepler Planet-Detection Mission* — Science 327:977
4. Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System* — KDD 2016
5. NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu
6. Lightkurve Collaboration (2018) — ascl:1812.013
7. Pedregosa et al. (2011) — *Scikit-learn: Machine Learning in Python* — JMLR 12:2825

---

## 👤 Author

**Sayed Umair Ali**
Department of Computer Science and Engineering
Ajay Binay Institute of Technology

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
