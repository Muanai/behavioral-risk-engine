# Credit Risk Feature Engine (End-to-End Fintech Infrastructure)

![Go](https://img.shields.io/badge/Go-1.26-00ADD8?style=for-the-badge&logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Store-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![AUC Uplift](https://img.shields.io/badge/AUC%20Uplift-%2B7.3%25-blueviolet?style=for-the-badge)
![Latency](https://img.shields.io/badge/Latency-%3C2ms-brightgreen?style=for-the-badge)

> **High‑performance temporal feature engineering & real-time serving for credit risk modeling**
> Built to answer one question: *are expensive, stateful features actually worth it, and can they be served at scale?*

---

## Project Overview

This project has evolved from a local Python experiment into a **Production-Grade Fintech Infrastructure**.

It is about **engineering complex, behavioral credit risk features** that:
* are impossible or inefficient in pure SQL / Pandas
* require stateful, windowed, per‑user logic
* justify low‑level optimization using **Numba**
* are served instantly via a highly optimized **Go API** backed by **Redis**

The project evaluates whether these features **meaningfully improve model performance** and proves that they can be served with **ultra-low latency** in a containerized microservices environment.

---

## Core Idea

Traditional credit models rely heavily on *magnitude features* (e.g., mean bill amount, average payment). These features are strong — but blind to **behavior over time**.

This project introduces a **Numba‑accelerated feature engine** to capture:
* temporal consistency
* behavioral trends
* threshold‑based risk patterns

Then answers two hard questions:
1. > *Do these complex features actually deliver measurable uplift?*
2. > *Can we serve them in real-time under massive load?*

---

## Architecture: Dual-Store Microservices

The system employs a **Dual-Store Architecture (Offline & Online)** with strict container isolation to handle both high-throughput training and ultra-low latency real-time serving.

```text
Raw Transactions (CSV)
        ↓
Data Pivoting (User × Time)
        ↓
Numba Feature Engine 
(Stateful loops & Temporal windows)
        ↓
   [ Forking Pipeline ]
   ↙                  ↘
OFFLINE STORE        ONLINE STORE
 PostgreSQL            Redis (Memory)
 (Batch Training)      (Real-Time Serving + bgsave() Persistence)
   ↓                        ↓
LightGBM Model       Feature Service Layer
(5-Fold Stratified)  Go API (Alpine Docker)
   ↓                        ↓
OOF ROC-AUC Eval     High-Throughput Inference
```

---

## Feature Engineering (Flagship Features)

All behavioral features are implemented using **Numba JIT‑compiled loops** to allow stateful logic, per‑user temporal windows, and efficient execution at scale.

### 1️. Consecutive Late Payment Streak
> *How many months in a row does a user fail to pay on time?*
* Captures consistency of delinquency
* Implemented as a stateful loop per user

### 2️. Payment‑to‑Bill Velocity
> *Is the payment ratio trending upward or downward over time?*
* Computes a mini linear trend per user
* Sensitive to gradual deterioration

### 3️. Critical Balance Utilization Count
> *How often does the user exceed 90% of their credit limit?*
* Threshold‑based risk signal highlighting repeated near‑limit behavior

---

## Production-Grade Latency & Load Testing

Complex features are useless if they choke production systems. The Go API (`feature_service`) was rigorously benchmarked using 10,000 concurrent requests against the Dockerized Redis online store.

**FEATURE SERVICE BENCHMARK REPORT**
* **Total Requests:** 10,000
* **Concurrency:** 100
* **Failed Requests:** 0 (0% failure rate)
* **Throughput:** **617.69 Requests per Second (RPS)**
* **Average Latency:** **1.61 miliseconds** per request

> **Result:** The Dual-Store architecture successfully decouples heavy analytical computation from real-time serving, delivering sub-2ms latency at scale.

---

## Engineering Maturity & Resilience

This is not an academic script. It is built with industrial ML engineering standards:

* **Dockerization & Multi-Stage Builds:** The serving layer is containerized using `golang:1.26.4-alpine`, producing an extremely lightweight and secure production image.
* **Data Persistence Strategy:** Redis RAM volatility is mitigated by enforcing an explicit `bgsave()` immediately after the Python pipeline finishes, guaranteeing data permanence into the Docker Volume (`/data`).
* **Data Integrity & Consistency:** Enforced LF line endings (`eol=lf`) via `.gitattributes` to prevent cross-OS (Windows/Linux) execution bugs, along with binary locking for `.pkl` models.
* **Quality Assurance:** Comprehensive Pytest coverage for Numba edge cases (all-zero payments, NaNs, zero-division).
* **Model Monitoring:** Implemented Population Stability Index (PSI) tracking to detect data drift (Current Status: **GREEN / Stable**).

---

## Impact Study: The Power of Hyperparameter Tuning

Three controlled scenarios are evaluated using a rigorous **5-Fold Stratified Cross-Validation** strategy with strict hyperparameter regularization.

| Scenario | Feature Set | OOF ROC-AUC |
| -------- | ----------- | ----------- |
| Baseline | Magnitude Only | 0.6355 |
| Behavior | Temporal & Trends Only | 0.6211 |
| **Hybrid** | **Magnitude + Behavior** | **0.6823** |

**Final Uplift (Hybrid vs Baseline):**
**+0.0468 AUC (+7.3%)**

---

## Key Takeaways

* Complex features should not replace simple aggregates — **they complement them**.
* Behavioral signals are often **non‑linear and interaction‑dependent**.
* Numba enables feature designs that are otherwise computationally prohibitive.
* **Microservices architecture works:** You can compute heavy features offline and still serve them under 2ms.

---

## Tech Stack

**Model & Data Layer:**
* Python 3
* NumPy / Pandas
* **Numba** (JIT compilation)
* LightGBM & Scikit‑learn

**Infrastructure & Serving:**
* **Go (Golang)** (Real-Time API)
* **Redis** (Online Feature Store)
* **PostgreSQL** (Offline Feature Store)
* **Docker & Docker Compose** (Container Orchestration)

---

*Built with a focus on practicality, skepticism, and measurable impact.*
