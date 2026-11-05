# Fetal Heart Rate Variability Prediction Using Deep Learning
 
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?logo=scipy&logoColor=white)
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
 
A data analysis and preprocessing pipeline for predicting **Mean Short-Term Variability (MSTV)** of the fetal heart rate from **cardiotocography (CTG)** recordings. The project covers cleaning, exploratory analysis, statistical testing, feature engineering and readable feature naming, preparing the data for a deep learning regression model in TensorFlow/Keras.
 
---
 
## Table of Contents
 
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Author](#author)
---
 
## Problem Statement
 
**Cardiotocography (CTG)** continuously records the fetal heart rate (FHR) and uterine contractions during pregnancy and labour. **Short-term variability** (the beat-to-beat fluctuation of the FHR) is one of the most important markers of fetal wellbeing:
 
- **Normal variability** indicates a healthy, well-oxygenated fetal nervous system.
- **Reduced variability** can signal hypoxia, fetal distress or the effect of medication.
This project frames MSTV estimation as a **supervised regression** problem: given the other CTG measurements, predict the MSTV value (in milliseconds).
 
---
 
## Dataset
 
**Source:** [Cardiotocography — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography)
 
- **Records:** 2,126 CTG examinations, each classified by expert obstetricians
- **Original columns:** 42
- **Target:** `MSTV` (Mean Short-Term Variability), range 0.2 – 7.0 ms, right-skewed (skew = 1.66)
<details>
<summary><b>Key feature groups (click to expand)</b></summary>
| Group | Features | Description |
|---|---|---|
| Recording window | `b`, `e` | Start and end instants of the analysed segment |
| Heart-rate events (counts) | `AC`, `FM`, `UC`, `DL`, `DS`, `DP` | Accelerations, fetal movements, uterine contractions, light / severe / prolonged decelerations |
| Heart-rate events (per second) | `AC.1`, `FM.1`, `UC.1`, `DL.1`, `DS.1`, `DP.1` | Same events normalised by recording length |
| Baseline | `LB` | Baseline FHR (beats per minute) |
| Variability | `ASTV`, `ALTV`, `MLTV`, **`MSTV`** | Abnormal / mean short- and long-term variability |
| FHR histogram | `Width`, `Min`, `Max`, `Nmax`, `Nzeros`, `Mode`, `Mean`, `Median`, `Variance`, `Tendency` | Shape statistics of the FHR distribution |
| Expert labels | `A`–`E`, `AD`, `DE`, `LD`, `FS`, `SUSP`, `CLASS`, `NSP` | FHR pattern classes and fetal state (Normal / Suspect / Pathologic) |
 
</details>
---
 
## Project Workflow
 
### 1. Data Cleaning
- No missing values in any column.
- Removed **10 duplicate rows** (2,126 → 2,116).
- Dropped `DR`, which is constant (always 0).
### 2. Outlier Treatment
- IQR detection flagged **1,774 rows** as containing at least one outlier. Dropping them would have left only ~340 records.
- **IQR capping** was applied instead, so no rows were lost.
- Capping turned 15 more columns into constants (zero variance), and they were dropped. That left **25 features + target**.
### 3. Exploratory Data Analysis
- **Univariate:** histograms, density plots and box plots for all features
- **Bivariate:** scatter plots of key CTG signals against MSTV, a correlation heatmap and a feature–target correlation chart
- **Pairwise:** pair plot of accelerations, contractions, decelerations, ASTV and MSTV
- **Multivariate:** accelerations vs. contractions coloured by MSTV
- **Target analysis:** MSTV distribution and trend
### 4. Statistical Analysis
 
| Test | Purpose | Key result |
|---|---|---|
| **Pearson correlation** | Linear association with MSTV | Strongest: `Width` (+0.66), `Variance` (+0.64), `Min` (−0.63), `DL.1` (+0.56), `ALTV` (−0.52) |
| **Spearman correlation** | Monotonic association with MSTV | Strongest: `Variance` (+0.78), `Width` (+0.72), `Min` (−0.70), `ALTV` (−0.68), `DL.1` (+0.62) |
| **Shapiro–Wilk** | Normality check | No feature (including MSTV) is normally distributed |
 
Features with **no significant** relationship to MSTV:
- `MLTV` (Pearson and Spearman)
- `b` (Pearson)
- `Tendency` and `B` (Spearman)
A **Yeo-Johnson power transform** was applied to all input features to reduce skewness.
 
### 5. Feature Engineering
 
| Feature | Formula | Intuition |
|---|---|---|
| `Combined_Variability_Stress` | ASTV + ALTV | Total burden of abnormal variability |
| `Acceleration_Deceleration_Ratio` | (AC + AC.1 + 1) / (DL + DL.1 + 1) | Balance of reassuring accelerations vs. decelerations |
| `Heart_Rate_Fluctuation_Range` | Max − Min | Overall spread of the FHR |
| `Contraction_Variability_Interaction` | UC × Variance | How contractions interact with FHR variability |
 
### 6. Readable Feature Names
All cryptic CTG abbreviations were renamed to descriptive names (for example, `ASTV` became `Abnormal_Short_Term_Variability` and `Width` became `FHR_Histogram_Width`). The cleaned dataset is exported as `cleaned_mstv.csv` with **29 features + target**.
 
---
 
## Key Findings
 
- 📊 **FHR histogram shape is the strongest predictor of MSTV.** A wider, higher-variance heart-rate distribution and a lower minimum FHR go with higher short-term variability.
- 📉 **Decelerations** (`DL`, `DL.1`) show strong positive correlations with MSTV.
- ⚠️ **Abnormal variability indicators** (`ASTV`, `ALTV`) are strongly *negatively* correlated with MSTV. This matches clinical expectations: the more time spent with abnormal variability, the lower the mean variability.
- 🤰 **Accelerations and uterine contractions** have moderate positive relationships with MSTV.
---
 
## Tech Stack
 
| Category | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistics | SciPy (`pearsonr`, `spearmanr`, `shapiro`) |
| Preprocessing | scikit-learn (`PowerTransformer`, `StandardScaler`) |
| Deep learning | TensorFlow / Keras |
| Environment | Google Colab |
 
---
 
## How to Run
 
### Option 1: Google Colab (recommended)
1. Download the CTG dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/193/cardiotocography) and export the raw data sheet as `cardiotocography_clean_MSTV.csv`.
2. Open the notebook in [Google Colab](https://colab.research.google.com/).
3. Upload the CSV to `/content/`.
4. Run all cells.
### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/fetal-heart-rate-variability-prediction.git
cd fetal-heart-rate-variability-prediction
pip install -r requirements.txt
jupyter notebook
```
Then update the `path` variable in the first code cell to point to your local copy of the CSV.
 
---
 
## Project Structure
 
```
fetal-heart-rate-variability-prediction/
├── Fetal_Heart_Rate_Variability_Prediction.ipynb   # Cleaning, EDA, statistics, feature engineering
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation
```
 
---
 
## Roadmap
 
- [x] Data cleaning & deduplication
- [x] Outlier capping and constant-feature removal
- [x] Univariate, bivariate and multivariate EDA
- [x] Statistical testing (Pearson, Spearman, Shapiro–Wilk)
- [x] Power transformation & feature engineering
- [x] Descriptive feature renaming
- [ ] Train/test split and feature scaling
- [ ] Linear Regression and Random Forest baselines
- [ ] Deep neural network regressor (Keras) with early stopping
- [ ] Evaluation with MAE, RMSE and R²
**Technical notes for the modelling stage**
- **Remove non-physiological and label columns.** `b` and `e` are the start and end times of the recording window, not heart-rate measurements, so they carry no clinical meaning. (They were renamed `Baseline_Value_1/2`, which is misleading.) `CLASS` and `B` are pattern labels that experts assigned after reviewing the full trace, variability included, so using them as inputs risks target leakage.
- **Revisit outlier capping.** Severe (`DS`) and prolonged (`DP`) decelerations and the fetal-state label `NSP` were removed because capping made them constant. These rare events are clinically meaningful, so capping only the continuous features would keep them.
- **Guard ratio features.** `Acceleration_Deceleration_Ratio` divides by power-transformed values plus 1, and the denominator can approach zero. Computing ratios on raw counts would avoid unstable values.
- **Consider transforming the target.** MSTV is right-skewed (skew 1.66, kurtosis 4.7). Modelling `log1p(MSTV)` and inverting the predictions often improves regression stability.
- **Prevent leakage.** Fit the power transformer and scaler on the training split only, inside a scikit-learn `Pipeline`.
---
 
## Disclaimer
 
This project is for **educational and research purposes only**. It is not a clinical decision-support tool and must not be used for fetal monitoring or medical decisions.
 
---
 
## Author
 
**Patricia Fuglo**
 
⭐ If you found this project useful, consider giving it a star!
 
