from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import joblib

from data import load_results, process_results, load_elo, process_elo
from features import matches_per_team, add_form_features, add_elo_features, add_results
from train import prepare_data, train_model
from simulate import simulate_group_stage, build_predictions_df, simulate_group_stage, simulate_knockout_match
from simulate import resolve_bracket, ROUND_OF_32, simulate_tournament, monte_carlo, predict_match, evaluate_predictions

from contextlib import asynccontextmanager

state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ELO...", flush=True)
    state['df_elo'] = process_elo(load_elo())
    print("Loading results...", flush=True)
    state['df_raw'] = load_results()
    print("Processing results...", flush=True)
    state['df'] = process_results(state['df_raw'], state['df_elo'])
    print("Building team view...", flush=True)
    state['df_teams'] = matches_per_team(state['df'])
    print("Adding form features...", flush=True)
    state['df'] = add_form_features(state['df'], state['df_teams'], n=5)
    print("Adding elo features...", flush=True)
    state['df'] = add_elo_features(state['df'], state['df_elo'])
    print("Adding results...", flush=True)
    state['df'] = add_results(state['df'])
    print("Loading/training model...", flush=True)
    model_path = Path('..') / 'models' / 'xgboost.pkl'
    if model_path.exists():
        state['model'] = joblib.load(model_path)
    else:
        X_train, X_test, y_train, y_test = prepare_data(state['df'])
        state['model'] = train_model(X_train, X_test, y_train, y_test)
    print("Loading monte carlo...", flush=True)
    state['monte_carlo'] = monte_carlo(state['model'], state['df_teams'],
        state['df_elo'], n=100)
    yield

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "World Cup 2026 Prediction API"}

import json


@app.get("/predictions")
def get_predictions():
    pred_df = build_predictions_df(state['model'], state['df_raw'],
        state['df_teams'], state['df_elo'])
    pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
    return json.loads(pred_df.to_json(orient='records'))

@app.get("/monte-carlo")
def get_winners():
    return state['monte_carlo']

@app.get("/match/{home}/{away}")
def predict_single_match(home: str, away: str, knockout: bool = False):
    date = pd.Timestamp('today')
    if knockout:
        return {'winner': simulate_knockout_match(state['model'], state['df_teams'],
            state['df_elo'], home, away, date)}
    probas = predict_match(state['model'], state['df_teams'], state['df_elo'], home, away, date)
    return {'P(W)': float(probas[0][0]), 'P(D)': float(probas[0][1]), 'P(L)': float(probas[0][2])}

@app.get("/simulate-bracket")
def simulate_bracket():
    results, best_thirds = simulate_group_stage(state['model'], state['df_teams'], state['df_elo'])
    matches = resolve_bracket(ROUND_OF_32, results, best_thirds)
    date = pd.Timestamp('2026-06-29')
    history, winner = simulate_tournament(state['model'], state['df_teams'], state['df_elo'], matches, date)
    return {'history': history, 'winner': winner}

@app.get("/upcoming")
def get_upcoming():
    pred_df = build_predictions_df(state['model'], state['df_raw'],
        state['df_teams'], state['df_elo'])
    pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
    upcoming = pred_df[pred_df['actual'].isna()].sort_values('date').head(10)
    return json.loads(upcoming.to_json(orient='records'))

@app.get("/history")
def get_history():
    pred_df = build_predictions_df(state['model'], state['df_raw'],
        state['df_teams'], state['df_elo'])
    pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
    played = pred_df[pred_df['actual'].notna()].sort_values('date', ascending=False)
    accuracy = evaluate_predictions(pred_df)
    return {
        'accuracy': float(accuracy),
        'matches': json.loads(played.to_json(orient='records'))
    }