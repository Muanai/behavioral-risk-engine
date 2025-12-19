# Project Profiling: Credit Risk Feature Engineering

## 1. Environment

**Hardware:**
- Laptop: Acer Nitro AN515
- CPU: Intel Core i5 11th Gen
- GPU: NVIDIA GeForce GTX 1650
- RAM: 16GB

**Software:**
- OS: Windows 11
- Python: 3.11
- Key Libraries: `numpy`, `pandas`, `numba`, `scikit-learn`, `lightgbm`, `pytest`, `matplotlib`, `seaborn`

**Virtual Environment:**
- `venv` used to isolate dependencies

---

## 2. Benchmarking

Performance of feature computation methods on sample dataset (~30,000 rows):

| Method                 | Runtime (s) |
|------------------------|------------|
| Pure Python loops       | ~53–58     |
| Pandas aggregation     | ~0.57–0.66 |
| Numba (JIT, warmup)    | ~0.00087–0.0012 |

**Insight:**  
- Numba achieves **~50,000x speedup** over plain Python loops.  
- Confirms necessity of JIT engine for stateful/loop-heavy features.

---

## 3. Feature Engineering

### Flagship Features
1. **Consecutive Late Streaks**
   - Maximum consecutive months of late payment per user.
2. **Payment-to-Bill Velocity**
   - Trend of ratio of payment over bill across months (slope).
3. **Critical Balance Utilization**
   - Count of months where usage exceeds 90% of credit limit.

**Implementation:**  
- Features computed using **Numba JIT** for high-speed execution on large datasets.  
- Multi-window approach allows computation per user efficiently.

---

## 4. Model Evaluation

**Models Used:**
- Logistic Regression (LR)
- LightGBM (LGBM)

**Sample Performance:**

| Model                | AUC   |
|---------------------|-------|
| Logistic Regression  | 0.542 |
| LightGBM             | 0.621 |

**Feature Insights:**
- LR Coefficients highlight `feat_max_late_streak` and `feat_pay_velocity`.  
- LGBM Feature Importance indicates `feat_pay_velocity` as key driver.

---

## 5. Impact Study (Ablation Scenarios)

**Scenarios:**

| Scenario               | Features Included              | AUC    |
|------------------------|--------------------------------|--------|
| Baseline               | Magnitude only (mean aggregates)| 0.6355 |
| Numba Behavior         | Trend only                     | 0.6211 |
| Hybrid                 | Magnitude + Trend (flagship)  | 0.6619 |

**Final Uplift (Hybrid vs Baseline):** +0.0265 (4.2% improvement)  

**Insight:**  
- Multi-window trend features add **measurable predictive power**.  
- Confirms industrial relevance of Numba-engineered features.

---

## 6. Quality Assurance & Robustness

### Unit Testing (Pytest)
- Core **Numba feature engine** logic tested.  
- Edge cases validated:
  - All-zero / all-late / mixed payment patterns
  - NaN and infinity values
  - Zero division handling  
- Ensures **mathematical correctness** and maintainability.

### Model Monitoring (PSI)
- Population Stability Index (PSI) checks **train vs test drift**.
- Traffic-light PSI interpretation:
  - GREEN: PSI < 0.1 → Stable
  - YELLOW: 0.1 ≤ PSI < 0.25 → Minor Drift
  - RED: PSI ≥ 0.25 → Major Drift  
- **Current Status:** GREEN (Stable, PSI = 0.0022)

---

## 7. Notes / Recommendations

- Feature computation is **highly scalable**, thanks to Numba.  
- PSI monitoring ensures model **robustness over time**.  
- Project structure supports:
  - Easy extension of new features  
  - Benchmarking additional algorithms  
  - Integration into industrial credit risk pipelines