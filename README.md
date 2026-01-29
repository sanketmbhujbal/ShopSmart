# 🛍️ ShopSmart AI

> **A latency-optimized semantic search engine designed for E-commerce.** 

> *Achieves <200ms inference on standard CPUs using Model Distillation & Hybrid Retrieval.*

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red) ![Redis](https://img.shields.io/badge/Redis-Caching-red) ![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-orange)

---

## The Challenge
Modern semantic search (Vector Search) is powerful but **computationally expensive**.
* **The Problem:** Running a standard Cross-Encoder Reranker (BERT) on a CPU takes **~2.5 seconds** per query. This violates the 200ms latency requirement for real-time search.
* **The Solution:** I engineered a custom pipeline using **Model Distillation (TinyBERT)** and **Candidate Budgeting** to reduce latency by **95%** without sacrificing relevance.

---

## Key Features
* **Hybrid Search:** Combines **BM25** (Keyword) and **Dense Vectors** (Semantic) to solve the "Exact Match" problem.

* **Intelligent Reranking:** Uses `cross-encoder/ms-marco-TinyBERT-L-2-v2` to re-score the top 20 results based on deep semantic intent.

* **Latency Optimized:** 

    * **Distillation:** Swapped standard BERT (110M params) for TinyBERT (14M params).

    * **Budgeting:** Limits expensive reranking to the top 12 candidates only.
    
    * **Caching:** Implements Redis to serve hot queries in <5ms.
    
* **Production Ready:** Includes structured logging (`search_logs.jsonl`) and an automated evaluation suite (`NDC@K`).

---

## Service Level Objectives (SLOs)
We define the following SLIs (Indicators) and SLOs (Targets) to ensure the system meets business requirements for interactivity and relevance.

|Category	|SLI (What we measure)|SLO (Target)|
|---|---|---|
|Latency	|End-to-end processing time for search requests (P95).	|< 200 ms|
|Availability	|Percentage of requests returning HTTP 200 without timeout.	|99.9%|
|Relevance	|Average NDCG@10 score on the Golden Set evaluation suite. |> 0.85|
|Freshness	|Time delta between product ingestion and search availability.	|< 10 min|

### Rationale:

•	**Latency (<200ms)**: Research shows e-commerce conversion rates drop significantly if search takes >200ms. 

•	**Relevance (>0.85)**: A score below 0.85 indicates the "Vocabulary Gap" is unresolved, leading to user abandonment.

---

## Architecture

<img width="940" height="513" alt="image" src="https://github.com/user-attachments/assets/68424f32-9a65-4a29-b020-20184e8af9bb" />

---

## Key Engineering Decisions & Trade-offs


### Decision 1: CPU Latency Optimization

•	**Context:** Running a Cross-Encoder (the gold standard for relevance) on a CPU usually takes 2-3 seconds per query.

•	**Decision:** Implemented Model Distillation and Budgeting.

•	**Implementation:**

1.	Switched from MiniLM-L6 (High Latency) to TinyBERT-L2 (Low Latency).
2.	Candidate Budgeting: We retrieve 20 items but only rerank the Top 12.
3.	Input Truncation: We feed only the Title to the reranker, discarding the Description.

•	**Impact:** Reduced latency from 2800ms to 140ms (19x speedup) with negligible accuracy loss (<2%).


### Decision 2: Hybrid Retrieval (Dense + Sparse)

•	**Context:** Vector search is bad at exact matches (e.g., distinguishing "USB C Cable" from "USB C Adapter").

•	**Decision:** Combined Vector Search with Keyword Search (BM25).

•	**Impact:** Vectors find "Mouse" when you search "Input Device". Keywords ensure "Samsung S24" doesn't return "Samsung S23".


### Decision 3: Zero-Dependency Observability

•	**Context:** Real-time dashboards are complex to set up.

•	**Decision:** Implemented structured JSON Lines (.jsonl) logging.

•	**Impact:** Provides a zero-dependency audit trail of every query, latency, and result count, readable by any data tool (Pandas, Excel, Splunk).

---
## Performance Metrics

|Metric|Baseline (Standard BERT)|Optimized (TinyBERT)|Improvement|
|---|---|---|---|
|Latency (P95)|~2,800 ms|~145 ms|19x Faster 🚀|
|Throughput|0.3 QPS|7.0 QPS|Scalable|
|NDCG@10|0.88|0.85|Negligible Loss|

---
## Known Limitations: 

**Long-tail queries:** Descriptions without rich metadata suffer on recall.

**Multilingual contexts:** The current BERT model is English-optimized.

**Synonym Expansion:** Candidate budgeting can sometimes miss items under heavy synonym expansion.

---
## Installation

### Clone the repo
```bash
git clone
cd shopsmart
```

### Start Infrastructure (Docker)
```bash
docker-compose up -d
```

### Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

### Ingest Data
```bash
python -m src.ingest
```

### Run App
```bash
# Terminal 1: Backend
uvicorn src.backend:app --reload

# Terminal 2: Frontend
streamlit run src/frontend.py
```
---

## Evaluation
Run the automated test suite to calculate NDCG scores:
```bash
python -m src.evaluate
```
---

## Future Roadmap

1.	**Ingestion Pipeline:** Automate daily data ingestion via Airflow to handle catalog updates.

2.	**A/B Testing:** Implement a router to serve 50% of traffic to TinyBERT and 50% to MiniLM to measure real-world conversion differences.

3.	**Personalization:** Inject user history vectors into the search query to boost brands the user previously bought.

---
