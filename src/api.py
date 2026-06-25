from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

from data import load_results, load_elo, process_elo
from train import prepare_data, train_model
from simulate import simulate_group_stage, build_predictions_df, simulate_group_stage, simulate_knockout_match
from simulate import resolve_bracket, ROUND_OF_32, simulate_tournament, monte_carlo, predict_match, evaluate_predictions

state = {}

async def run_monte_carlo_background():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: monte_carlo(state['model'], state['df_teams'], state['df_elo'], n=1000))
    state['monte_carlo'] = result

@asynccontextmanager
async def lifespan(app: FastAPI):
    state['monte_carlo'] = []
    print("Loading df elo...", flush=True)
    state['df_elo'] = process_elo(load_elo())
    print("Loading dfraw...", flush=True)
    state['df_raw'] = load_results()
    print("Loading df...", flush=True)
    state['df'] = pd.read_csv(Path(__file__).parent.parent / 'data' / 'processed' / 'features.csv')
    state['df']['date'] = pd.to_datetime(state['df']['date'])
    print("Loading df teams...", flush=True)
    state['df_teams'] = pd.read_csv(Path(__file__).parent.parent / 'data' / 'processed' / 'df_teams.csv')
    state['df_teams']['date'] = pd.to_datetime(state['df_teams']['date'])
    state['df_teams']['goals_scored'] = pd.to_numeric(state['df_teams']['goals_scored'], errors='coerce').fillna(0)
    state['df_teams']['goals_conceded'] = pd.to_numeric(state['df_teams']['goals_conceded'], errors='coerce').fillna(0)
    state['df_teams']['tournament_weighting'] = pd.to_numeric(
    state['df_teams']['tournament_weighting'], errors='coerce').fillna(0.1)
    print("Loading model...", flush=True)
    model_path = Path('..') / 'models' / 'xgboost.pkl'
    if model_path.exists():
        state['model'] = joblib.load(model_path)
    else:
        X_train, X_test, y_train, y_test = prepare_data(state['df'])
        state['model'] = train_model(X_train, X_test, y_train, y_test)
    print("Loading mc...", flush=True)
    asyncio.create_task(run_monte_carlo_background())
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "World Cup 2026 Prediction API"}


@app.get("/predictions")
def get_predictions():
    pred_df = build_predictions_df(state['model'], state['df_raw'],
        state['df_teams'], state['df_elo'])
    pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
    return json.loads(pred_df.to_json(orient='records'))

@app.get("/monte-carlo")
def get_winners():
    if not state['monte_carlo']:
        return {"status": "calculating", "message": "Monte Carlo simulation in progress, try again in a minute"}
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