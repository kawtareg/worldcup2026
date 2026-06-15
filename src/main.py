from pathlib import Path
import pandas as pd
import joblib

from data import load_results, process_results, load_elo, process_elo
from features import matches_per_team, add_form_features, add_elo_features, add_results
from train import prepare_data, train_model
from simulate import simulate_group_stage, build_predictions_df, simulate_group_stage, resolve_bracket, ROUND_OF_32, simulate_tournament

def main():
    df_elo = load_elo()
    df_elo = process_elo(df_elo)
    df_raw = load_results()
    df = process_results(df_raw, df_elo)
    df_teams = matches_per_team(df)
    df = add_form_features(df, df_teams, n=5)
    df = add_elo_features(df, df_elo)
    df = add_results(df)
    model_path = Path('..') / 'models' / 'xgboost.pkl'
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        X_train, X_test, y_train, y_test = prepare_data(df)
        model = train_model(X_train, X_test, y_train, y_test)

    predictions = build_predictions_df(model, df_raw, df_teams, df_elo)
    print(predictions.head(10))
    results, best_thirds = simulate_group_stage(model, df_teams, df_elo)
    matches = resolve_bracket(ROUND_OF_32, results, best_thirds)
    date = pd.Timestamp('2026-06-29')
    history, winner = simulate_tournament(model, df_teams, df_elo, matches, date)
    print(f"Vainqueur de la CDM 2026 : {winner}")

if __name__ == "__main__":
    main()
