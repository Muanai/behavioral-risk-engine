# Credit Risk Feature Engine (Numba‑Powered)

> **High‑performance temporal feature engineering for credit risk modeling**
> Built to answer one question: *are expensive, stateful features actually worth it?*

---

## Project Overview

This project is **not** about training the fanciest model.

It is about **engineering complex, behavioral credit risk features** that:

* are impossible or inefficient in pure SQL / Pandas
* require stateful, windowed, per‑user logic
* justify low‑level optimization using **Numba**

The project evaluates whether these features **meaningfully improve model performance** when combined with standard magnitude‑based aggregates.

---

## Core Idea

Traditional credit models rely heavily on *magnitude features*:

* mean bill amount
* average payment
* total utilization

These features are strong — but blind to **behavior over time**.

This project introduces a **Numba‑accelerated feature engine** to capture:

* temporal consistency
* behavioral trends
* threshold‑based risk patterns

Then answers the hard question:

> *Do these complex features actually deliver measurable uplift?*

---

## Architecture

```
Raw Transactions (CSV)
        ↓
Data Pivoting (User × Time)
        ↓
Numba Feature Engine
  ├─ Consecutive Late Streaks
  ├─ Payment‑to‑Bill Velocity
  └─ Critical Utilization Counts
        ↓
Engineered Feature Table
        ↓
Model Evaluation Layer
  ├─ Logistic Regression (interpretability)
  └─ LightGBM (non‑linear capacity)
        ↓
Impact Comparison (Baseline vs Behavior vs Hybrid)
```

---

## Feature Engineering (Flagship Features)

All behavioral features are implemented using **Numba JIT‑compiled loops** to allow:

* stateful logic
* per‑user temporal windows
* efficient execution at scale

### 1️. Consecutive Late Payment Streak

> *How many months in a row does a user fail to pay on time?*

* Captures consistency of delinquency
* Implemented as a stateful loop per user
* Non‑expressible as a simple aggregation

---

### 2️. Payment‑to‑Bill Velocity

> *Is the payment ratio trending upward or downward over time?*

* Computes a mini linear trend per user
* Sensitive to gradual deterioration
* Weak linearly, strong in interaction

---

### 3️. Critical Balance Utilization Count

> *How often does the user exceed 90% of their credit limit?*

* Threshold‑based risk signal
* Highlights repeated near‑limit behavior

---

##  Performance Benchmark

Feature computation speed comparison:

| Method       | Runtime (seconds) |
| ------------ |-------------------|
| Python loops | ~53–58            |
| Pandas       | ~0.57–0.75        |
| **Numba**    | **~0.0011**       |

> **Result:** Numba delivers *orders of magnitude* speedup, enabling complex feature logic at scale.

---

##  Modeling Strategy

Two models are intentionally used:

* **Logistic Regression**
  → interpretability & linear signal check

* **LightGBM**
  → interaction discovery & non‑linear capacity

The goal is *not* leaderboard performance, but **feature impact validation**.

---
## Quality Assurance & Robustness

This project follows industrial ML engineering standards to ensure correctness and reliability of features and models.

### Unit Testing (Pytest)
- Core Numba feature engine logic is covered by **unit tests**.
- Edge cases validated include:
  - All-zero payments
  - Consecutive late payments
  - Flat or downward trends for payment velocity
  - Handling of NaN and infinity values
  - Zero division safeguards
- Guarantees correctness of loops and stateful computations for flagship features.

### Model Monitoring (PSI)
- Population Stability Index (PSI) implemented to detect data drift between training and testing sets.
- PSI traffic-light interpretation:
  - **GREEN**: Stable
  - **YELLOW**: Minor Drift
  - **RED**: Major Drift → Retraining Required
- Current Model Status: **GREEN (Stable, PSI = 0.0022)**

---

##  Impact Study: Do Behavioral Features Matter?

Three controlled scenarios are evaluated:

### Scenario A — Baseline (Magnitude Only)

* Mean / aggregate features only
* Strong but static

**AUC: 0.6365**

---

### Scenario B — Behavioral Only (Numba Engine)

* Temporal & trend features only
* Orthogonal signal

**AUC: 0.6141**

---

### Scenario C — Hybrid (Magnitude + Behavior)

* Combined feature space
* Enables interaction learning

**AUC: 0.6582**

---

### Final Results

| Scenario | AUC    |
| -------- |--------|
| Hybrid   | 0.6619 |
| Baseline | 0.6355 |
| Behavior | 0.6211 |

**Final Uplift (Hybrid vs Baseline):**
 **+0.0265 AUC (+4.2%)**

> Temporal behavioral features alone underperform static aggregates, but **deliver measurable uplift when fused**, justifying their computational cost.

---

##  Key Takeaways

* Complex features should not replace simple aggregates — **they complement them**
* Behavioral signals are often **non‑linear and interaction‑dependent**
* Numba enables feature designs that are otherwise impractical
* Feature engineering impact must be **measured, not assumed**

---

## Tech Stack

* Python 3
* NumPy / Pandas
* **Numba** (JIT compilation)
* Scikit‑learn
* LightGBM

---

##  Notes

* `user_id` is treated strictly as a primary key, never as a model feature
* All evaluations use identical splits for fair comparison
* Focus is on **engineering rigor**, not leaderboard chasing

---

*Built with a focus on practicality, skepticism, and measurable impact.*
