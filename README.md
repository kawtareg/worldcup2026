# worldcup2026: FIFA World Cup 2026 Prediction

A machine learning system predicting match outcomes and simulating the 2026 World Cup. Features a live web dashboard updated as the tournament unfolds.

> Built as part of a machine learning & data science learning journey - 4th year Computer Science Engineering, INSA Rouen.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green?logo=fastapi)
![Deployed](https://img.shields.io/badge/Deployed-Render-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Live Demo

🌐 [kawtareg.github.io/worldcup2026](https://kawtareg.github.io/worldcup2026)

---

## Features

- Match outcome prediction (W/D/L) with calibrated probabilities
- Monte Carlo tournament simulation (1000 simulations)
- Live track record — predictions vs real results updated daily
- Simulate the tournament from the current bracket stage
- Automated nightly data refresh via GitHub Actions
- REST API with FastAPI

---

## Stack

| Tool | Role |
|---|---|
| Python 3.12 | Language |
| XGBoost | Match outcome prediction |
| FastAPI | REST API |
| pandas / numpy | Data processing |
| GitHub Actions | CI/CD & nightly data refresh |
| Render | API hosting |
| GitHub Pages | Frontend hosting |

---

## Architecture

**Data pipeline:**

martj42/international_results → precompute.py → features.csv

eloratings.net (Kaggle)       → df_teams.csv

**Prediction pipeline:**

features.csv + xgboost.pkl → FastAPI → GitHub Pages frontend

**Simulation:**

Monte Carlo (1000 tournaments) → win probabilities per team

tournament_state.json → simulate from current bracket stage

---

## Installation

### Prerequisites

- Python 3.12+
- uv (`pip install uv`)
- ELO ratings CSV from [Kaggle](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings) → place in `data/raw/elo_ratings_wc2026.csv`

### 1. Clone the project

```bash
git clone https://github.com/kawtareg/worldcup2026
cd worldcup2026
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Precompute features

```bash
cd src && uv run precompute.py
```

### 4. Train the model

```bash
uv run main.py
```

### 5. Run the API

```bash
uv run uvicorn api:app --reload
```

---

## Project Structure

```
worldcup2026/
├── data/
│   ├── raw/              # results.csv, elo_ratings_wc2026.csv
│   ├── processed/        # features.csv, df_teams.csv
│   └── tournament_state.json
├── models/
│   └── xgboost.pkl
├── src/
│   ├── data.py           # data loading & processing
│   ├── features.py       # feature engineering
│   ├── train.py          # model training
│   ├── simulate.py       # Monte Carlo simulation
│   ├── precompute.py     # offline feature computation
│   ├── api.py            # FastAPI endpoints
│   └── main.py           # training pipeline
├── index.html            # frontend dashboard
└── .github/workflows/    # nightly data refresh
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /monte-carlo` | Win probabilities per team |
| `GET /predictions` | All WC 2026 match predictions |
| `GET /upcoming` | Next 10 matches with probabilities |
| `GET /history` | Past predictions vs real results |
| `GET /simulate-from-now` | Simulate tournament from current stage |
| `GET /match/{home}/{away}` | Predict a single match |

---

## Model

- **Algorithm**: XGBoost classifier (3 classes: W/D/L)
- **Features**: ELO difference, weighted form (opponent-adjusted), average goals scored/conceded
- **Log-loss**: 1.013 (vs 1.099 random baseline)
- **Evaluation**: Temporal train/test split (pre/post 2020)
- **Optimization**: Grid search over depth, learning rate, subsample

---

## What I Learned

- End-to-end ML pipeline from raw data to production API
- Feature engineering with temporal weighting and opponent strength
- Probabilistic evaluation (log-loss, why accuracy is misleading)
- Monte Carlo simulation for uncertainty quantification
- FastAPI with async background tasks and lifespan management
- CI/CD with GitHub Actions for automated data refresh
- Deployment on Render + GitHub Pages

---

## Roadmap

- [ ] Agent that auto-updates tournament bracket from live sources
- [ ] Poisson bivariate model comparison
- [ ] Player-level features (club performance, injuries)

---

## Author

**Kawtar El Gueddari** — [GitHub](https://github.com/kawtareg) · INSA Rouen Normandie