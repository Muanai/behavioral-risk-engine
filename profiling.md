# Project Profiling: Credit Risk Feature Engineering

## 1. Environment

**Hardware:**
- Laptop: Acer Nitro AN515
- CPU: Intel Core i5 11th Gen
- GPU: NVIDIA GeForce GTX 1650
- RAM: 16GB

**Software & Infrastructure:**
- OS: Windows 11 (Development) / Linux Alpine (Docker Production)
- Data Science: Python 3.11 (`numpy`, `pandas`, `numba`, `scikit-learn`, `lightgbm`)
- Backend API: Go 1.21+
- Databases: PostgreSQL (Offline Store), Redis (Online Store)
- Containerization: Docker & Docker Compose

---

## 2. Feature Computation Benchmarking

Performance of feature computation methods on sample dataset (~30,000 rows):

| Method                 | Runtime (s) |
|------------------------|------------|
| Pure Python loops       | ~53–58     |
| Pandas aggregation     | ~0.57–0.66 |
| Numba (JIT, warmup)    | ~0.00087–0.0012 |

**Insight:**  
- Numba achieves **~50,000x speedup** over plain Python loops.  
- Confirms necessity of JIT engine for stateful/loop-heavy behavioral features.

---

## 3. Serving Layer Load Testing (Go + Redis)

We tested the Go Feature Service reading directly from the Redis Online Store to simulate real-time production traffic.

| Metric | Result |
|---|---|
| Throughput | **617 RPS** (Requests Per Second) |
| Latency (p99) | **< 2ms** |
| Error Rate | 0.00% |

**Insight:**
- The dual-store architecture (syncing Numba-computed features to Redis) guarantees blazing-fast, single-digit millisecond retrieval times.
- Safe for synchronous real-time inference in the critical path of loan origination.

---

## 4. Model Evaluation (5-Fold CV LightGBM)

**Model:** LightGBM (LGBM) using 5-Fold Cross-Validation.

| Scenario               | Features Included              | AUC Score |
|------------------------|--------------------------------|--------|
| Baseline               | Magnitude only (mean aggregates)| 0.6355 |
| Numba Behavior         | Trend only                     | 0.6211 |
| Hybrid                 | Magnitude + Trend (flagship)  | **0.6823** |

**Final Uplift (Hybrid vs Baseline):** **+0.0468 AUC (+7.3% improvement)**  

**Insight:**  
- Multi-window trend features (like `feat_max_late_streak` and `feat_pay_velocity`) add **massive predictive power**.  
- Moving beyond simple aggregations to complex behavioral tracking is highly justified by the +7.3% AUC uplift.

---

## 5. Quality Assurance & Robustness

### Unit Testing (Pytest)
- Core **Numba feature engine** logic tested for mathematical correctness.  
- Edge cases validated: All-zero, mixed payment patterns, NaN/infinity, zero division handling.

### Docker & Environment Parity
- Strict `.gitattributes` enforcement for `LF` line endings to ensure shell scripts run flawlessly inside Alpine Linux containers.
- Guaranteed parity between local development and production execution.

### Model Monitoring (PSI)
- Population Stability Index (PSI) checks **train vs test drift**.
- Traffic-light PSI interpretation:
  - GREEN: PSI < 0.1 → Stable
  - YELLOW: 0.1 ≤ PSI < 0.25 → Minor Drift
  - RED: PSI ≥ 0.25 → Major Drift  
- **Current Status:** GREEN (Stable, PSI = 0.0022)

---

## 6. Recommendations

- Feature computation is **highly scalable** thanks to Numba, and serving is **production-ready** thanks to Go & Redis.
- PSI monitoring ensures model **robustness over time**.  
- The infrastructure is ready for real-world integration as a microservice in an industrial Fintech pipeline.