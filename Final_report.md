# Final Report — Life Expectancy Prediction (WHO 2000–2015)

**Course:** ECOM7126 — Machine Learning for Business and E-Commerce (2025-26)
**Programme:** MSc in E-Commerce and Internet Computing, HKU
**Dataset:** WHO Global Health Observatory — `Life_Expectancy_Data.csv`
**Coverage:** 193 countries × 16 years (2000–2015), 2 938 records, 22 features
**Notebook:** [Project_part2_Life_Expectancy.ipynb](Project_part2_Life_Expectancy.ipynb)

The mission of this report is to answer four questions derived from the WHO longitudinal
dataset:

1. What does the EDA reveal about the data and about life expectancy (LE)?
2. Which three ML models were used to predict LE, and why?
3. How well did those models perform?
4. What are the **top 3 factors** that most affect Life Expectancy?

---

## 1. EDA — Process and Insights

### 1.1 Process

The exploratory analysis followed a systematic, reproducible sequence:

| Stage | What was done | Purpose |
|---|---|---|
| **Structure audit** | `.info()`, `.describe()`, dtype check, shape check | Confirm 2 938 rows × 22 cols, 2 categorical + 19 numeric features |
| **Target analysis** | Histogram, boxplot, skewness/kurtosis of `Life expectancy` | Understand the distribution to be learnt |
| **Missingness** | Null counts per column, missing-value heatmap | Decide imputation strategy |
| **Distribution check** | Skewness scan across all numeric features | Flag features that need log transform |
| **Bivariate** | Pearson correlation with target + correlation heatmap | Identify strong predictors & multicollinearity |
| **Group comparison** | Developed vs Developing (mean / median / range) | Check the main categorical split |
| **Temporal view** | Mean LE per year (2000–2015), per status | See longitudinal trends |
| **Outlier scan** | IQR rule per numeric feature | Decide keep / trim / log-compress |

### 1.2 Key Findings

**Target `Life expectancy`**

| Statistic | Value |
|---|---:|
| Mean | 69.2 yrs |
| Median | 72.1 yrs |
| Min / Max | 36 / 89 yrs |
| Std. Dev. | 9.5 yrs |
| Skewness | **− 0.64** (left-skewed) |

The left tail is populated by sub-Saharan countries during the HIV/AIDS crisis years,
pulling the mean below the median.

**Data Quality**

- **Missing values** are concentrated in `Population` (22 %), `Hepatitis B` (18 %),
  `GDP` (15 %), and `Total expenditure` (8 %) — handled by **median imputation inside
  the modelling pipeline** (no leakage).
- **Severe right-skew (skew > 5)** in `Population`, `Measles`, `infant deaths`,
  `under-five deaths`, and `GDP` → **log1p-transformed** before modelling.
- **Multicollinearity**: `infant deaths` ↔ `under-five deaths` with r ≈ **0.99**
  → `under-five deaths` dropped.
- Most IQR "outliers" are genuine (India-sized populations, measles outbreaks),
  so they are log-compressed rather than removed.

**Developed vs Developing**

| Status | Mean LE | Median | Range |
|---|---:|---:|---|
| Developed  | ≈ 79 yrs | 79 | 69 – 89 |
| Developing | ≈ 67 yrs | 71 | 36 – 85 |

The **~12-year gap** is closing — developing countries gained **+5 yrs** between
2000 and 2015 vs **+3 yrs** for developed countries.

**Correlation with LE**

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

### 1.3 EDA-Level Insights

1. **Education beats wealth.** `Schooling` (r = 0.75) correlates with LE more strongly than
   `GDP` (r = 0.46). *How* wealth is spent matters more than how much there is.
2. **HIV/AIDS is catastrophic.** Countries with HIV/AIDS > 5 deaths / 1 000 live births
   average LE ≈ **51 yrs**; those ≤ 0.1 average ≈ **73 yrs** — a **22-year gap**.
3. **GDP has diminishing returns.** Beyond ≈ $10 000 per capita, extra GDP adds
   little LE — a logarithmic, not linear, relationship.
4. **Immunisation success story.** Polio and Diphtheria coverage rose from ≈ 80 % to
   ≈ 90 % globally during 2000-2015; Hepatitis B coverage still lags in developing countries.
5. **Bottom-15 countries cluster geographically**, overwhelmingly sub-Saharan Africa —
   driven by HIV/AIDS and adult mortality.

---

## 2. Three Machine-Learning Models

### 2.1 Why These Three

The three models were picked to span the **bias–variance spectrum** and to let us
compare a linear baseline against two non-linear ensembles:

| # | Model | Family | Why it was chosen |
|---|---|---|---|
| 1 | **Ridge Regression** (α = 1.0) | Linear | Interpretable baseline. L2 penalty stabilises coefficients in presence of correlated predictors. |
| 2 | **Random Forest** (400 trees, `min_samples_leaf = 2`) | Bagging ensemble | Captures non-linearities and feature interactions without tuning; robust to outliers and scale. |
| 3 | **Gradient Boosting** (400 trees, lr = 0.05, depth = 3) | Boosting ensemble | Sequentially corrects residuals; typically the strongest performer on tabular data. |

### 2.2 Shared Modelling Pipeline

| Step | Decision |
|---|---|
| Target | `Life expectancy` |
| Rows removed | 10 rows with missing target |
| Feature dropped | `under-five deaths` (collinear), `Country` (identity leakage) |
| Log-transform (`log1p`) | `GDP`, `Population`, `Measles`, `infant deaths` |
| Categorical encoding | `Status` → 1 (Developed) / 0 (Developing) |
| Missing-value handling | **Median imputation inside the `Pipeline`** (no leakage) |
| Ridge-only step | `StandardScaler` (trees are scale-invariant) |

### 2.3 Train/Test Split — Grouped by Country

A **naïve random split would leak information**: different years of the same country
are highly similar, so random splitting lets the model memorise countries.

Instead:

- `GroupShuffleSplit(test_size = 0.2, groups = Country)` → all rows of a country go
  to *either* train or test.
- **Train:** 2 336 rows, 146 countries. **Test:** 592 rows, 37 unseen countries
  (Belgium, Canada, Japan, Indonesia, Ethiopia, DR Congo, …).
- Cross-validation: 5-fold **`GroupKFold`** on the training set.

This directly simulates the project requirement: **predicting LE for countries not
in the dataset.**

---

## 3. Model Results

### 3.1 Held-out Performance on 37 Unseen Countries

| Model | Test R² | Test MAE (yrs) | Test RMSE (yrs) | CV R² (mean ± std) |
|---|---:|---:|---:|---:|
| Ridge Regression      | 0.824 | 3.20 | 4.06 | 0.787 ± 0.034 |
| Random Forest         | 0.920 | 1.95 | 2.73 | 0.888 ± 0.036 |
| **Gradient Boosting** | **0.928** | 1.97 | **2.59** | **0.897 ± 0.034** |

### 3.2 Interpretation

- **Tree ensembles roughly halve the RMSE** vs Ridge, confirming the underlying
  relationship is strongly **non-linear** — threshold effects of HIV/AIDS, the GDP
  plateau, and interactions with `Schooling`.
- **Gradient Boosting is the best generaliser** (R² = 0.928, MAE ≈ 2 yrs). Random
  Forest is a very close runner-up (R² = 0.920).
- The **CV std ≈ 0.03** indicates the ranking is stable across country samples —
  not an artefact of one lucky split.
- A ~2-year MAE on a 36–89-year range means the ensemble explains ~93 % of
  country-to-country variance — strong generalisation to unseen countries.

### 3.3 Prediction for a Country Outside the Dataset

`predict_life_expectancy(country_features)` re-fits all three models on the full
dataset, applies the same preprocessing, and returns per-model and ensemble estimates.

| Scenario | Ridge | RF | GBM | **Ensemble** |
|---|---:|---:|---:|---:|
| Hypothetical *Developing* country (GDP $3.5 k, HIV/AIDS 0.3, Schooling 11 yrs) | 67.8 | 69.9 | 70.5 | **≈ 69.4 yrs** |
| Hypothetical *Developed* country (GDP $45 k, HIV/AIDS 0.1, Schooling 16 yrs)   | 79.8 | 83.5 | 84.4 | **≈ 82.6 yrs** |

The two tree models agree within ~1 year. Ridge under-predicts the developed
scenario by 3–4 years — a visible symptom of its inability to capture the GDP
plateau. The **ensemble mean is the recommended single-value prediction.**

---

## 4. Top 3 Factors Affecting Life Expectancy

Each model's feature importances were **normalised to sum to 1**, then averaged
across the three models to yield a **model-agnostic consensus ranking**:

| Rank | Feature | Avg. importance | Why it matters |
|:-:|---|---:|---|
| **1** | **HIV/AIDS** (deaths per 1 000 live births, age 0–4) | **0.390** | The single most destructive factor across 2000–2015. High-burden countries (mostly sub-Saharan Africa) lose 15–25 LE years. Roll-out of anti-retroviral treatment is the main reason global LE improved over the study period. |
| **2** | **Adult Mortality** (prob. of dying 15–60 yrs, per 1 000) | **0.172** | Direct measure of working-age survival. Captures chronic-disease burden (CVD, cancer, diabetes), road safety, maternal health, and conflict. |
| **3** | **Income composition of resources** (HDI income index, 0–1) | **0.147** | Measures how *effectively* a country converts income into human development (schooling, health). Consistently beats raw GDP because it rewards equitable spending. |

### Why not `Schooling` or `GDP`?

Both are strong **univariate** correlates of LE (r = 0.75 and 0.46 respectively),
but in a **multivariate** tree model their signal is partly absorbed by
`Income composition of resources` (which already encodes education) and by
`Adult Mortality` (the downstream outcome of lifetime education + healthcare).
This is a textbook example of **correlation ≠ feature importance**.

### Policy Implications

1. **Fight HIV/AIDS** — prevention, ART access, and awareness yield the largest
   LE gains, especially in sub-Saharan Africa.
2. **Reduce Adult Mortality** — invest in chronic-disease management,
   maternal care, and road safety.
3. **Improve HDI income composition** — translate GDP into schooling, healthcare,
   and sanitation. Raw GDP growth has diminishing returns on LE.

---

## 5. Conclusions

- The WHO 2000–2015 longitudinal dataset is **rich but imperfect**: it requires
  deliberate handling of missing values, severe skewness, and multicollinearity.
- A **Gradient Boosting ensemble** is the best of the three models tested:
  **R² ≈ 0.93** and **MAE ≈ 2 years** on entirely unseen countries.
- The **top three drivers** of LE — **HIV/AIDS burden**, **Adult Mortality**, and the
  **Income composition of resources (HDI)** — indicate that disease control,
  adult-health systems, and equitable development spending are the
  highest-leverage policy levers.
- The `predict_life_expectancy()` utility turns the models into a **practical tool**:
  feed in any indicators available for a new country and receive a robust
  ensemble estimate of its LE.

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
