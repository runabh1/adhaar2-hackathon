

# Aadhaar Service Stress Dashboard

A full-stack **data-driven governance dashboard** to analyze, visualize, and explain **Aadhaar service stress risks** across Indian districts using **machine learning, analytics, and AI-generated policy insights**.

---

## 📌 Project Overview

The **Aadhaar Service Stress Dashboard** helps administrators and policymakers:

* Monitor **service stress levels** at district level
* Identify **high-risk regions**
* Understand **why** a district is risky (explainability)
* Receive **AI-generated policy recommendations**
* Export ranked risk data for reporting and planning

The system combines:

* Statistical risk modeling
* Time-series analysis
* Interactive visualization
* AI-assisted decision support

---

## 🏗️ Architecture

```
┌────────────┐        HTTP/JSON        ┌──────────────┐
│  Frontend  │  ───────────────────▶  │   FastAPI    │
│ (HTML/JS)  │                        │   Backend    │
│            │  ◀───────────────────  │              │
└────────────┘        CSV / JSON       └──────┬───────┘
                                              │
                                   ┌──────────┴──────────┐
                                   │  ML Model + Dataset  │
                                   │  (Pandas + sklearn) │
                                   └─────────────────────┘
```

---

## 🧠 Core Components

### 1️⃣ Frontend (Dashboard UI)

* **Technology:** HTML, TailwindCSS, Vanilla JavaScript
* **Features:**

  * State / District / Date filters
  * KPI cards (Risk score, biometric ratio, pressures)
  * Trend charts (Chart.js)
  * Top-risk and hotspot analysis
  * Markdown-rendered AI explanations
  * CSV export

📄 File: `index.html`

---

### 2️⃣ Backend API

* **Technology:** FastAPI
* **Responsibilities:**

  * Serve filtered Aadhaar stress data
  * Compute rankings & percentiles
  * Provide risk verdicts (LOW / MEDIUM / HIGH)
  * Generate AI-assisted explanations & policy recommendations
  * Stream ranked CSV downloads

📄 File: `main.py`

---

### 3️⃣ Machine Learning Model

* **Model Type:** Regression-based service stress estimator
* **Input:** Operational Aadhaar indicators
* **Output:** Continuous `service_stress_risk` score
* **Evaluation:** MAE, RMSE, Spearman rank correlation

📦 File: `aadhaar_service_stress_model.pkl`

---

### 4️⃣ Dataset

* **Source:** Aggregated Aadhaar enrollment & update metrics
* **Granularity:** District × Date
* **Key Columns:**

  * `service_stress_risk`
  * `biometric_to_enrolment_ratio`
  * `child_update_pressure`
  * `elderly_update_pressure`

📊 File: `aadhaar_merged_dataset.csv`

---

## ✨ Key Features

* 📊 **District-level stress scoring**
* 📈 **Risk trend over time**
* 🏆 **Top-risk district ranking**
* 🔍 **Explainable risk analysis**
* 🤖 **AI-generated policy recommendations**
* 📥 **CSV export (Streamlit-equivalent logic)**
* 🧼 **State-safe UI (clears old AI outputs on reload)**

---

## 🤖 AI Capabilities

The system generates:

* **Risk explanations** (why a district is risky)
* **Actionable policy recommendations**, including:

  * Infrastructure expansion
  * Staffing optimization
  * Child-friendly and elderly-focused services
  * Emergency service load balancing

AI outputs are rendered using **Markdown → HTML** for clarity and professionalism.

---

## 📦 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd aadhaar-service-dashboard
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_google_generative_ai_key
```

---

## ▶️ Running the Project

### Start Backend

```bash
uvicorn main:app --reload
```

Backend runs at:

```
http://localhost:8000
```

### Open Frontend

Open `index.html` directly in your browser
(or serve it using a local server).

---

## 📡 API Endpoints (Core)

| Endpoint                                           | Description            |
| -------------------------------------------------- | ---------------------- |
| `/states`                                          | List all states        |
| `/districts/{state}`                               | Districts for a state  |
| `/dates/{state}/{district}`                        | Available dates        |
| `/risk`                                            | Risk metrics           |
| `/risk-verdict/{score}`                            | LOW / MEDIUM / HIGH    |
| `/risk-percentile/{state}/{district}/{date}`       | Comparative percentile |
| `/risk-trend/{state}/{district}`                   | Time-series trend      |
| `/top-districts`                                   | Top-risk districts     |
| `/district-hotspots/{state}`                       | State hotspots         |
| `/risk-explanation/{state}/{district}/{date}`      | AI explanation         |
| `/policy-recommendation/{state}/{district}/{date}` | AI policy              |
| `/download-ranked-data`                            | Ranked CSV export      |

---

## 📥 CSV Export Logic

The CSV export **matches Streamlit logic exactly**:

* Grouped by **district**
* Mean aggregation of risk metrics
* Sorted by **highest service stress risk**
* Streamed as `text/csv`

---

## 🎯 Use Cases

* UIDAI operational planning
* Resource allocation decisions
* District-level monitoring
* Policy simulations
* Academic / SIH / hackathon submissions

---

## 🔒 Disclaimer

This project is for **educational, analytical, and demonstration purposes**.
Final administrative decisions must always involve **human oversight**.

---
