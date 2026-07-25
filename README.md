# 🛰️ AI-Based Space Object Detection & Collision Prediction Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/dashboard-streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Space Situational Awareness (SSA)** framework that ingests satellite orbital tracking data (TLEs), propagates trajectories using SGP4 orbital mechanics, detects close-approach conjunction events, scores relative risk using a machine learning model (trained on real ESA Kelvins Collision Avoidance Challenge data), computes analytical collision probability (Chan's 2D Pc), and generates plain-language AI advisories — all presented through a **NASA Mission Control-style dashboard**.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#-system-architecture)
- [Pipeline Stages](#-pipeline-stages)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Dataset & Model Training](#-dataset--model-training)
- [Model Performance](#-model-performance)
- [Dashboard Walkthrough](#-dashboard-walkthrough)
- [Project Structure](#-project-structure)
- [Known Limitations](#-known-limitations)
- [Future Scope](#-future-scope)
- [License](#-license)

---

## 🌟 Project Overview

This project was developed as part of the **BSERC Internship Program** to demonstrate a working, end-to-end AI-based space collision prediction system. It addresses the growing challenge of **space traffic management** as mega-constellations (Starlink, OneWeb) and orbital debris continue to increase the risk of collisions in Earth's orbit.

### Key Features

| Feature | Description |
|---|---|
| **7-Stage SSA Pipeline** | Complete from data acquisition to AI-generated advisories |
| **Real ML Training** | Gradient Boosting model trained on 162,634 real ESA Kelvins CDM events |
| **Analytical Physics** | Chan's 2D Probability of Collision (Pc) alongside ML scores |
| **3D Orbital Visualization** | Interactive Plotly 3D globe with orbit trails and threat vectors |
| **AI Mission Advisor** | Chat assistant grounded in session data + Claude API integration |
| **Offline Ready** | Works fully offline with sample data and template AI reports |
| **Exportable Briefings** | Downloadable PDF and HTML mission reports |
| **NASA Mission Control UI** | Professional dark-themed dashboard with 5 interactive tabs |

---

## 🎯 Problem Statement

Satellite operators receive hundreds of automated close-approach (conjunction) warnings per week. Each warning is a dense numerical message (miss distance, relative velocity, covariance data) that requires specialist training to interpret. With decision windows as short as 24–48 hours, and mega-constellations pushing conjunction volume higher every year, manual triage does not scale.

This system automates the triage process by:
1. **Detecting** close approaches using real orbital mechanics (SGP4)
2. **Scoring** each event's risk using a trained ML model
3. **Explaining** the risk with analytical physics and feature breakdowns
4. **Advising** operators with plain-language AI-generated reports

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NASA MISSION CONTROL DASHBOARD                    │
│                         (Streamlit app.py)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │
│  │ MISSION  │  │   3D     │  │  THREAT  │  │TELEMETRY │  │  AI  │  │
│  │ CONTROL  │  │  ORBITAL │  │ASSESSMENT│  │   FEED   │  │ADVISOR│  │
│  │   Tab    │  │ THEATER  │  │   Tab    │  │   Tab    │  │ Tab  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────┘  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                       7-STAGE SSA PIPELINE                          │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  STAGE 1 │  │  STAGE 2 │  │  STAGE 3 │  │  STAGE 4 │            │
│  │  Data    │→│  Orbit   │→│Conjunction│→│   Risk   │            │
│  │Acquisition│  │Propagation│  │ Detection │  │  Scoring  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│       │              │              │              │               │
│       ▼              ▼              ▼              ▼               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│  │  STAGE 5 │  │  STAGE 6 │  │  STAGE 7 │                         │
│  │Analytical│→│  AI      │→│  Visual- │                         │
│  │  Pc & Δv  │  │  Reports │  │  ization  │                         │
│  └──────────┘  └──────────┘  └──────────┘                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Pipeline Stages

### Stage 1: Data Acquisition
- **Live mode**: Fetches real TLE data from CelesTrak's public API (supports `stations`, `starlink`, `iridium-33-debris`, `cosmos-2251-debris` groups)
- **Offline mode**: Uses pre-packaged sample data (~40 objects spanning Starlink, ISS, OneWeb, GPS, GEO, debris fields)
- **Historical replay**: Pre-collision TLEs from the 2009 Iridium 33 / Cosmos 2251 collision
- **Custom upload**: Supports user-uploaded CSV (CDM format) and TLE files

**Module**: `space_project/fetch_data.py`, `space_project/sample_data.py`, `space_project/upload_handler.py`

### Stage 2: SGP4 Orbit Propagation
- Uses **Skyfield** (SGP4 propagator) — the same model used by NORAD/Space-Track
- Configurable prediction window (1–7 days) and time step (30–300 seconds)
- Vectorized propagation for performance
- Orbital regime classification: **LEO** (160–2000 km), **MEO** (2000–35786 km), **GEO** (~35786 km), **HEO** (>35786 km)

**Module**: `space_project/propagate.py`

### Stage 3: Conjunction Detection
- Pairwise O(N²) minimum distance search across all tracked objects
- Flags events below a configurable distance threshold (5–150 km)
- Computes: miss distance, relative velocity, time-to-closest-approach (TCA)

**Module**: `space_project/conjunction.py`

### Stage 4: ML Risk Scoring
- **Algorithm**: Gradient Boosting Regressor (200 estimators, max depth 3, learning rate 0.05)
- **Features**: `miss_distance_km`, `relative_velocity_kms`, `tca_hours_from_now`, `size_class`
- **Training data**: Real ESA Kelvins Collision Avoidance Challenge dataset (162,634 train / 24,484 test)
- **Output**: Risk score (0–100) and category (Low / Medium / High / Critical)

**Module**: `space_project/risk_model.py`

### Stage 5: Analytical Physics (Pc & Delta-V)
- **Chan's 2D Probability of Collision**: Analytical Pc using position uncertainty covariance
- **Delta-V estimation**: Simplified Hohmann-like avoidance burn calculation
- **Pc vs ML comparison**: Disagreement analysis between analytical and ML methods

**Module**: `space_project/pc_analytical.py`

### Stage 6: AI Risk Advisory Generation
- **LLM mode**: Anthropic Claude API generates structured risk advisories
- **Template mode**: Deterministic fallback with identical output format
- **Session-grounded chat**: AI Mission Assistant answers questions about current session data

**Module**: `space_project/ai_report.py`, `space_project/mission_assistant.py`

### Stage 7: Visualization & Dashboard
- **5-tab NASA Mission Control UI** built with Streamlit
- 3D orbital theater with interactive Plotly globe
- Ranked conjunction table with feature explainability
- Kessler Syndrome cascade simulator
- PDF/HTML briefing export

**Module**: `app.py`, `space_project/export_report.py`

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.9+ | Core development |
| **Dashboard** | Streamlit | Interactive web UI |
| **Orbital Mechanics** | Skyfield (SGP4) | NORAD-standard orbit propagation |
| **Machine Learning** | scikit-learn (GBR) | Gradient Boosting risk model |
| **Analytical Pc** | NumPy-based Chan formula | Collision probability computation |
| **3D Visualization** | Plotly | Interactive 3D orbital theater |
| **Generative AI** | Anthropic Claude API | Natural-language risk advisories |
| **Data Handling** | pandas, NumPy | Tabular data processing |
| **Export** | fpdf2 | PDF briefing generation |
| **Data Sources** | CelesTrak, ESA Kelvins CDM | Real orbital data |

---

## 📥 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd space_collision_prediction_project

# 2. (Recommended) Create a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set Anthropic API key for AI reports
set ANTHROPIC_API_KEY=your_key_here
# On macOS/Linux:
# export ANTHROPIC_API_KEY=your_key_here
```

### Launch the Dashboard

```bash
streamlit run app.py
```

The application runs **fully offline** out of the box using embedded sample data and template AI fallback reports.

---

## 🎮 Usage Guide

### Quick Start

1. Run `streamlit run app.py`
2. Select a data source in the sidebar (Offline Sample, Live, Historical Replay, or Upload)
3. Configure the prediction window, time step, and conjunction threshold
4. Click **🚀 RUN FULL PIPELINE**
5. Explore the results across 5 tabs

### Data Sources

| Source | Description | Internet Required |
|---|---|---|
| **Offline Sample (~40 objects)** | Pre-packaged TLEs for Starlink, ISS, debris, etc. | ❌ No |
| **Live (CelesTrak)** | Fetches current TLEs from CelesTrak API | ✅ Yes |
| **Historical Replay (2009)** | TLEs from before Iridium/Cosmos collision | ❌ No |
| **Upload Custom Dataset** | Upload your own CSV or TLE files | ❌ No |

### Dashboard Tabs

| Tab | What You'll See |
|---|---|
| 🌐 **Mission Control** | Overview metrics, constellation risk exposure, orbital regime breakdown, anomaly flags |
| 🌍 **3D Orbital Theater** | Interactive 3D globe with satellite orbit trails and red threat vectors |
| ⚠️ **Threat Assessment** | Ranked conjunction table, feature explainability, Pc vs ML comparison, model validation |
| 📡 **Telemetry Feed** | Live state vector stream, Kessler Syndrome cascade simulator |
| 🤖 **AI Mission Advisor** | Chat assistant, automated risk advisories, PDF/HTML briefing export |

---

## 📊 Dataset & Model Training

### ESA Kelvins Collision Avoidance Challenge Dataset

The model is trained on the **official ESA Kelvins Collision Avoidance Challenge dataset**, a real-world dataset of conjunction data messages (CDMs).

| Statistic | Value |
|---|---|
| **Training samples** | 162,634 |
| **Test samples** | 24,484 |
| **Features** | 4 (miss distance, relative velocity, TCA, size class) |
| **Target** | Risk score (0–100) derived from log10(Pc) |

### Feature Mapping from ESA CDM

| ESA CDM Column | Mapped Feature | Conversion |
|---|---|---|
| `miss_distance` (m) | `miss_distance_km` | ÷ 1000 |
| `relative_speed` (m/s) | `relative_velocity_kms` | ÷ 1000 |
| `time_to_tca` (days) | `tca_hours_from_now` | × 24, clip ≥ 0 |
| `c_object_type` | `size_class` | ROCKET BODY→3, PAYLOAD→2, else→1 |
| `risk` (log10 Pc) | `risk_score` | ((risk + 30) / 30) × 100, clip [0,100] |

### Model Configuration

```python
GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
```

---

## 📈 Model Performance

### Regression Metrics (on 24,484 held-out test events)

| Metric | Value | Interpretation |
|---|---|---|
| **Mean Absolute Error (MAE)** | **23.86** points | Average prediction error on 0–100 scale |
| **Root Mean Squared Error (RMSE)** | **27.74** points | Penalizes large errors more heavily |
| **R² (Coefficient of Determination)** | **0.2855** | Model explains ~28.6% of risk variance |

### Classification Accuracy

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Low | 79% | 16% | 0.26 | 7,878 |
| Medium | 12% | 54% | 0.20 | 2,729 |
| **High** | **56%** | **56%** | **0.56** | 10,615 |
| Critical | 82% | 1% | 0.02 | 3,262 |
| **Overall Accuracy** | | | **36%** | 24,484 |

### Feature Importances

| Feature | Importance |
|---|---|
| `tca_hours_from_now` | **74.4%** |
| `miss_distance_km` | 13.2% |
| `size_class` | 6.9% |
| `relative_velocity_kms` | 5.5% |

### Calibration Table

| Risk Bucket | Count | Mean Predicted | Mean Actual | Error |
|---|---|---|---|---|
| 0–10 | 22 | 9.1 | 14.4 | -5.3 |
| 10–20 | 1,539 | 17.1 | 12.1 | +5.0 |
| 20–30 | 4,032 | 25.0 | 23.0 | +2.1 |
| 30–40 | 3,822 | 35.0 | 33.3 | +1.7 |
| 40–50 | 4,405 | 45.0 | 45.4 | **-0.3** ✅ |
| 50–60 | 5,499 | 55.2 | 57.4 | -2.2 |
| 60–70 | 4,724 | 64.1 | 65.8 | -1.7 |
| 70–80 | 408 | 72.4 | 73.4 | -1.0 |
| 80–90 | 33 | 82.8 | 82.0 | **+0.8** ✅ |

> **Note**: An MAE of ~24 points is an **honest, expected baseline** for a 4-feature model trained on noisy Pc data. This is not a limitation of the implementation but a reflection of the inherent uncertainty in orbital collision prediction. The project reports these metrics transparently rather than overfitting or hiding results.

---

## 🖥️ Dashboard Walkthrough

### 🌐 Mission Control Tab
- **Metric HUD cards**: Tracked Objects, Conjunction Events, Critical Threats, High Threats, Anomalies Flagged
- **Constellation Risk Exposure**: Grouped risk analysis by constellation (Starlink, debris fields, ISS, etc.)
- **Orbital Regime Breakdown**: Pie chart showing LEO/MEO/GEO/HEO distribution
- **Statistical Anomaly Detection**: Z-score based orbital anomaly flags

### 🌍 3D Orbital Theater
- Textured 3D Earth rendered with Plotly surfaces
- Orbit trails for all tracked objects
- Red conjunction threat lines between close-approach pairs
- Configurable controls (show/hide orbits, focus on specific events)

### ⚠️ Threat Assessment
- Ranked conjunction table with NASA Green/Yellow/Orange/Red threat scale
- Feature explainability breakdown per event
- Analytical Pc vs ML Score disagreement analysis
- Delta-V avoidance burn estimates
- Model validation panel with feature importance bar chart

### 📡 Telemetry Feed
- Monospaced live state vector stream
- Kessler Syndrome cascade simulator (toggleable "what if" panel)

### 🤖 AI Mission Advisor
- Session-grounded chat assistant (answers questions like "which pair is highest risk?")
- Automated risk advisories per event
- Exportable PDF and HTML mission briefings

---

## 📁 Project Structure

```
space_collision_prediction_project/
├── app.py                          # Main Streamlit dashboard
├── test_dataset.py                 # Model evaluation test suite
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
│
├── space_project/                  # Core pipeline modules
│   ├── __init__.py                 # Package init (version 2.0.0)
│   ├── fetch_data.py               # TLE data acquisition from CelesTrak
│   ├── sample_data.py              # Offline sample TLE data (~40 objects)
│   ├── propagate.py                # SGP4 orbit propagation
│   ├── conjunction.py              # Pairwise conjunction detection
│   ├── risk_model.py               # ML risk scoring (GBR model)
│   ├── pc_analytical.py            # Chan's 2D analytical Pc
│   ├── ai_report.py                # AI risk advisory generation
│   ├── mission_assistant.py        # Session-grounded chat assistant
│   ├── explainability.py           # Feature importance & explainability
│   ├── anomaly_detection.py        # Statistical orbital anomaly detection
│   ├── kessler_sim.py              # Kessler Syndrome cascade simulator
│   ├── export_report.py            # PDF/HTML briefing export
│   ├── upload_handler.py           # Custom dataset upload handler
│   │
│   └── models/                     # Trained model artifacts
│       ├── risk_model.joblib       # Trained Gradient Boosting model
│       └── training_report.json    # Training metrics (MAE, importances, etc.)
│
├── dataSet/                        # Dataset files (not tracked in git)
│   ├── train_data.csv              # 162,634 training events
│   ├── test_data.csv               # 24,484 test events
│   └── Project_Proposal_BSERC_Final.pdf
│
└── Implentation plan/              # Planning documents (not tracked in git)
    ├── 01_PRD.md                   # Product Requirements Document
    ├── 02_TRD.md                   # Technical Requirements Document
    ├── 03_APP_FLOW.md              # Application Flow
    ├── 04_UI_UX_DESIGN_BRIEF.md    # UI/UX Design Brief
    ├── 05_BACKEND_SCHEMA.md        # Backend Schema
    ├── 06_IMPLEMENTATION_PLAN.md   # Implementation Plan
    └── implementation_plan.md      # NASA Mission Control Redesign Plan
```

---

## ⚠️ Known Limitations

In compliance with project specifications, this system explicitly acknowledges the following operational boundaries:

1. **No Real-Time Streaming Data**: Operating on snapshot runs (batch propagation over a user-configured time window).
2. **No Autonomous Maneuver Execution**: Avoidance Δv figures are advisory approximations (Pc threshold guidance), not automated thrust commands.
3. **Not a Certified Operational Tool**: Developed as a demonstrable SSA research & decision-support framework; not certified for live operational satellite collision avoidance without human analyst oversight.
4. **Statistical Risk Bounds**: Collision probability (Pc) and ML risk scores are statistical estimates bounded by tracking data uncertainty — never presented as absolute guarantees of safety or collision.
5. **Model Accuracy**: The MAE of ~24 points is an honest, expected baseline for a 4-feature model on noisy Pc data. The model is intended for triage and prioritization, not as a substitute for detailed CDM analysis.

---

## 🔭 Future Scope

- **Real-time streaming data ingestion** (continuous monitoring, not batch snapshots)
- **Multi-sensor fusion** (radar, optical, laser ranging data integration)
- **Automated maneuver planning** with optimized burn trajectories
- **Federated learning** across multiple operator nodes
- **Space-Track.org API integration** for expanded catalog access
- **Mobile alerting** via SMS/email/push notifications
- **Multi-user authentication** and role-based access control

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ for the BSERC Internship Program</sub><br>
  <sub>© 2024 — AI-Based Space Object Detection & Collision Prediction Framework</sub>
</p>
