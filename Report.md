# Life Expectancy Analysis & Prediction — Project Report

**Course:** ECOM7126 — Machine Learning for Business and E-Commerce (2025-26)
**Programme:** MSc in E-Commerce and Internet Computing, HKU
**Dataset:** WHO Global Health Observatory — *Life_Expectancy_Data.csv*
**Coverage:** 193 countries × 16 years (2000 – 2015), 2 938 records, 22 features
**Notebook:** [Project_part2_Life_Expectancy.ipynb](Project_part2_Life_Expectancy.ipynb)

---

## 1. Systematic Data Analysis (EDA)

### 1.1 Dataset Structure

| Property | Value |
|---|---|
| Rows / Columns | 2 938 / 22 |
| Time span | 2000 – 2015 |
| Countries | 193 |
| Target variable | `Life expectancy` (years) |
| Categorical features | `Country`, `Status` (Developed / Developing) |
| Numerical features | 19 (health, immunisation, economic, social) |

### 1.2 Target Distribution

| Statistic | Value |
|---|---:|
| Mean | 69.2 yrs |
| Median | 72.1 yrs |
| Min / Max | 36 / 89 yrs |
| Std. Dev. | 9.5 yrs |
| Skewness | **− 0.64** (left-skewed) |

The left skew is produced by a tail of sub-Saharan countries with LE < 50 years during
the HIV/AIDS crisis years, pulling the mean below the median.

### 1.3 Data Quality Findings

- **Missing values** concentrated in `Population` (22 %), `Hepatitis B` (18 %),
  `GDP` (15 %), `Total expenditure` (8 %), and 9 other columns.
- **Severe right-skew (skew > 5)** in `Population`, `Measles`, `infant deaths`,
  `under-five deaths`, and `GDP` → require **log transformation** before modelling.
- **Multicollinearity**: `infant deaths` vs `under-five deaths` ≈ **r = 0.99** →
  one must be dropped.
- IQR-based outliers exist in almost every feature but most are genuine (large
  populations, disease outbreaks), so they are log-compressed rather than deleted.

### 1.4 Developed vs Developing Gap

| Status | Mean LE | Median | Range |
|---|---:|---:|---|
| Developed  | ≈ 79 yrs | 79 | 69 – 89 |
| Developing | ≈ 67 yrs | 71 | 36 – 85 |

- Gap ≈ **12 years**, but narrowing: developing countries gained **+5 yrs (2000→2015)**
  vs **+3 yrs** for developed ones — the global distribution is compressing.

### 1.5 Correlation with Life Expectancy

| Direction | Feature | r |
|---|---|---:|
| **Positive** | Schooling | +0.75 |
|  | Income composition of resources | +0.72 |
|  | BMI | +0.57 |
|  | Diphtheria / Polio coverage | +0.47 |
|  | GDP | +0.46 |
| **Negative** | Adult Mortality | − 0.70 |
|  | HIV/AIDS (0–4 deaths) | − 0.56 |
|  | Thinness 1-19 / 5-9 | − 0.47 |

### 1.6 Key Insights

1. **Education ≻ Wealth.** `Schooling` (r = 0.75) is a stronger correlate of LE than
   `GDP` (r = 0.46). How wealth is spent matters more than the wealth itself.
2. **HIV/AIDS is catastrophic.** Countries with HIV/AIDS > 5 per 1 000 live births
   average LE ≈ **51 yrs**; those with ≤ 0.1 average ≈ **73 yrs** — a **22-year gap**.
3. **GDP shows diminishing returns.** Beyond ≈ $10 000 GDP/capita additional wealth
   adds little to LE (logarithmic relationship).
4. **Immunisation success.** Polio and Diphtheria coverage climbed from ≈ 80 % to
   ≈ 90 % globally during 2000-2015; Hepatitis B coverage remains a gap, especially
   in developing countries.
5. **Bottom countries are clustered geographically.** Almost all bottom-15 countries
   are in sub-Saharan Africa, driven largely by HIV/AIDS and adult-mortality factors.

---

## 2. Machine-Learning Models

### 2.1 Modelling Pipeline

| Step | Decision |
|---|---|
| Target | `Life expectancy` |
| Rows removed | Rows with missing target (10 rows) |
| Feature dropped | `under-five deaths` (multicollinear with `infant deaths`) |
| Log-transform (`log1p`) | `GDP`, `Population`, `Measles`, `infant deaths` |
| Categorical encoding | `Status` → 1 (Developed) / 0 (Developing) |
| Identifier dropped | `Country` (to prevent label leakage) |
| Missing value handling | Median imputation **inside the pipeline** (no leakage) |
| Ridge-only step | `StandardScaler` (trees are scale-invariant) |

### 2.2 Train / Test Split — **Grouped by Country**

- Using `GroupShuffleSplit(test_size = 0.2, groups = Country)` — all records of a
  country go to *either* train or test.
- Train: **2 336 rows, 146 countries.**
  Test: **592 rows, 37 countries** (e.g., Belgium, Canada, Japan, Indonesia,
  Ethiopia, DR Congo, …).
- This directly simulates the project requirement: **predicting LE for countries
  that may not be in the dataset**.
- Cross-validation: 5-fold **`GroupKFold`** on the training set.

### 2.3 Models Compared

| # | Model | Rationale |
|---|---|---|
| 1 | **Ridge Regression** (α = 1.0) | Interpretable linear baseline; L2 stabilises collinear coefficients. |
| 2 | **Random Forest** (400 trees, min_samples_leaf = 2) | Non-linear bagging; captures interactions without tuning. |
| 3 | **Gradient Boosting** (400 trees, lr = 0.05, depth = 3) | Sequential boosting; usually the top performer on tabular data. |

### 2.4 Results — Held-out 37 Unseen Countries

| Model | Test R² | Test MAE (yrs) | Test RMSE (yrs) | CV R² (mean ± std) |
|---|---:|---:|---:|---:|
| Ridge Regression     | 0.824 | 3.20 | 4.06 | 0.787 ± 0.034 |
| Random Forest        | 0.920 | 1.95 | 2.73 | 0.888 ± 0.036 |
| **Gradient Boosting** | **0.928** | 1.97 | **2.59** | **0.897 ± 0.034** |

**Observations**

- Both tree ensembles cut the RMSE roughly **in half** vs Ridge, confirming the
  relationship is strongly non-linear (threshold effects of HIV/AIDS, plateauing
  GDP, interactions with `Schooling`).
- **Gradient Boosting is the best generaliser** (R² = 0.928, MAE ≈ 2 yrs). Random
  Forest is a very close runner-up.
- CV std ≈ 0.03 → the ranking is **stable across country samples**, not an artefact
  of one lucky split.
- A ~2-year MAE on a 36–89-year range means the ensemble explains ~93 % of
  country-to-country variance — strong generalisation to unseen countries.

### 2.5 Prediction for a Country Outside the Dataset

The helper `predict_life_expectancy(country_features)` in the notebook re-fits all
three models on the full dataset, applies the same log-transform & median
imputation, and returns one estimate per model plus an **ensemble average**.

| Scenario | Ridge | RF | GBM | **Ensemble** |
|---|---:|---:|---:|---:|
| Hypothetical *Developing* country (GDP $3.5 k, HIV/AIDS 0.3, Schooling 11 yrs) | 67.8 | 69.9 | 70.5 | **≈ 69.4 yrs** |
| Hypothetical *Developed* country (GDP $45 k, HIV/AIDS 0.1, Schooling 16 yrs)   | 79.8 | 83.5 | 84.4 | **≈ 82.6 yrs** |

The two tree models agree within ~1 year; Ridge under-predicts the developed
example by 3–4 years — a visible symptom of its inability to capture the GDP
plateau. The **ensemble mean is the recommended single-value prediction**.

---

## 3. Top 3 Factors Affecting Life Expectancy

Importances from the three models were each normalised to sum to 1, then averaged
to produce a **model-agnostic consensus ranking**:

| Rank | Feature | Avg. importance | Why it matters |
|:-:|---|---:|---|
| 🥇 **1** | **HIV/AIDS** (deaths per 1 000 live births, age 0–4) | **0.390** | The single most destructive factor in the longitudinal data. High-burden countries (mostly sub-Saharan Africa) suffer LE losses of 15–25 years. Progress on anti-retroviral treatment between 2000-2015 is the main reason global LE improved. |
| 🥈 **2** | **Adult Mortality** (probability of dying 15–60 yrs per 1 000) | **0.172** | Directly measures survival in working-age adults. Captures chronic-disease burden, road safety, maternal health, and conflict — all proximate causes of premature death. |
| 🥉 **3** | **Income composition of resources** (HDI income index, 0–1) | **0.147** | Reflects how effectively a nation converts income into human development (schooling, health). Consistently beats raw GDP because it rewards *equitable* spending, not just wealth. |

> **Why not `Schooling` and `GDP`?** Both are strong univariate correlates, but in
> a multivariate model their predictive signal is partly absorbed by
> `Income composition of resources` (which encodes education) and by `Adult
> Mortality` (which is the downstream outcome of lifetime education & healthcare).
> This is a classic example of how **correlation ≠ feature importance**.

### Policy Implications

1. **Fight HIV/AIDS** — prevention, ART access, and awareness campaigns yield the
   largest LE gains, especially in sub-Saharan Africa.
2. **Reduce Adult Mortality** — invest in chronic-disease management (CVD, cancer,
   diabetes), road safety, and maternal care.
3. **Improve HDI income composition** — translate GDP into schooling, healthcare
   access, and sanitation. Raw GDP growth alone has **diminishing returns** on LE.

---

## 4. Conclusions

- The WHO 2000-2015 longitudinal dataset is **rich but imperfect**: it requires
  careful handling of missing values, severe skewness, and multicollinearity
  before modelling.
- A **Gradient Boosting ensemble** is the best of the three models tested,
  achieving **R² ≈ 0.93 and MAE ≈ 2 years on entirely unseen countries** — a
  strong demonstration that LE can be predicted from socio-economic and health
  indicators alone.
- The **top three drivers** of LE, based on consensus across all three models,
  are **HIV/AIDS burden**, **Adult Mortality**, and the **Income composition of
  resources (HDI)** — suggesting that disease control, adult-health systems, and
  equitable development spending are the highest-leverage policy levers.
- The provided `predict_life_expectancy()` utility turns the models into a
  **practical tool**: feed in whatever indicators are known for a country not
  represented in the dataset and receive a robust ensemble estimate of its LE.

---

## Appendix A — Reproduction

1. Open [Project_part2_Life_Expectancy.ipynb](Project_part2_Life_Expectancy.ipynb).
2. Ensure `Life_Expectancy_Data.csv` is in the same folder.
3. Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`,
   `scikit-learn`.
4. Run all cells top to bottom — EDA first, then the Part 2 modelling section.

## Appendix B — Features Used in Modelling

```
Year, Status, Adult Mortality, infant deaths, Alcohol, percentage expenditure,
Hepatitis B, Measles*, BMI, Polio, Total expenditure, Diphtheria, HIV/AIDS,
GDP*, Population*, thinness 1-19 years, thinness 5-9 years,
Income composition of resources, Schooling
```
`*` = log-transformed before modelling. `under-five deaths` dropped (r ≈ 0.99 with
`infant deaths`). `Country` dropped to avoid identity leakage.
