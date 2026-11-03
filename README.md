# AgriCast-AI: Agricultural Commodity Price Intelligence & Forecasting Platform

[![Next.js](https://img.shields.io/badge/Next.js-16.4-black?logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Django](https://img.shields.io/badge/Django-5.1+-092e20?logo=django)](https://www.djangoproject.com/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-SARIMA-blue)](https://www.statsmodels.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end commodity price forecasting and decision intelligence platform. Built with **SARIMA**, **Holt-Winters**, and **XGBoost**, backed by **FastAPI** & **Django REST Framework**, and visualized using an interactive **Next.js** dashboard with Recharts.

---

## 🌟 Key Features

- **Multi-Model Forecasting Suite:**
  - **SARIMA $(0,1,1)(0,1,1)_{12}$ (Champion):** Captures seasonal fluctuations and stochastic trends with a **12.58% MAPE** and **₹1,034.80 MAE**.
  - **Holt-Winters Exponential Smoothing (Challenger):** Additive seasonal modeling with rapid inference.
  - **XGBoost Multi-Step Regressor:** Non-linear lag feature regression for non-standard market shocks.
- **Model Persistence & Sub-Millisecond Latency:**
  - Pre-fitted models and metadata packaged into `models/sarima_model.pkl` to eliminate repetitive real-time training overhead.
- **Dynamic 95% Confidence Intervals:**
  - Quantifies forecast uncertainty and potential market upside/downside risk across 6M, 12M, and 24M horizons.
- **Seasonality & Harvest Cycle Decomposition:**
  - Aggregates multi-decade historical trends into monthly seasonality indices to pinpoint post-harvest low-price buying windows versus off-season peaks.
- **Model Drift & Anomaly Governance:**
  - Automated 36-month rolling backtest monitoring with an active 15% MAPE threshold alerting system.
- **Custom Commodity Upload:**
  - Drag-and-drop CSV upload for real-time forecasting of custom crops, grains, and spices (Wheat, Rice, Onion, Cotton, etc.).
- **Procurement & Budget Scenario Simulator:**
  - Estimate procurement budgets, risk buffers, and best/worst-case cost exposures based on user-defined purchase volumes.

---

## 📊 Model Evaluation Benchmark

Across chronological rolling-origin validation windows (36 out-of-sample observations):

| Model | MAE (₹/quintal) | RMSE (₹/quintal) | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **SARIMA $(0,1,1)(0,1,1)_{12}$** | **1,034.80** | **1,604.65** | **12.58%** | **Champion** |
| **Holt-Winters (Additive)** | 1,289.40 | 1,892.10 | 15.20% | Challenger |
| **XGBoost (Recursive Lag-12)** | 1,420.50 | 2,045.30 | 17.80% | Benchmark |

---

## 🏗️ Architecture

```text
┌────────────────────────────────────────────────────────┐
│                   Next.js 16 (App Router)              │
│       Recharts Charts • Drift Monitor • CSV Upload     │
└───────────────────────────▲────────────────────────────┘
                            │ HTTP (Port 3000)
┌───────────────────────────▼────────────────────────────┐
│                    FastAPI Gateway                     │
│         API Routing • CORS • Validation (Port 8001)    │
└───────────────────────────▲────────────────────────────┘
                            │ HTTP (Port 8000)
┌───────────────────────────▼────────────────────────────┐
│                Django REST Core Service                │
│    Model Serving Engine • In-Memory Caching • Bundles   │
└───────────────────────────▲────────────────────────────┘
                            │ Loads
┌───────────────────────────▼────────────────────────────┐
│         models/sarima_model.pkl (Master Bundle)        │
│   SARIMA • Holt-Winters • XGBoost • Seasonality Stats  │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- Git LFS

### 1. Clone the Repository
```bash
git clone https://github.com/SivaPanyam/AgriCast-AI.git
cd AgriCast-AI
```

### 2. Backend Setup
Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install django djangorestframework statsmodels xgboost pandas numpy httpx uvicorn fastapi
```

Train and export the `.pkl` model bundle:
```bash
python train_and_save_model.py
```

Start the Django and FastAPI services:
```bash
# Terminal 1: Django core service
python manage.py runserver 8000

# Terminal 2: FastAPI Gateway
uvicorn fastapi_service.main:app --port 8001 --host 127.0.0.1
```

### 3. Frontend Setup
```bash
# Terminal 3: Next.js Frontend
cd frontend
npm install
npm run dev
```

Visit **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 📁 Repository Structure

```text
├── config/                  # Django project settings & URL configuration
├── data/                    # Historical agricultural commodity price datasets
├── fastapi_service/         # High-performance asynchronous FastAPI gateway
├── forecasting/             # Django app for statistical modeling & API endpoints
├── frontend/                # Next.js frontend with Tailwind CSS & Recharts
├── models/                  # Serialized .pkl artifacts (tracked via Git LFS)
├── notebooks/               # Research & model development Jupyter notebooks
├── outputs/                 # Exported forecasts and evaluation tables
├── manage.py                # Django CLI management script
├── train_and_save_model.py  # Model training & artifact serialization pipeline
└── README.md
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
