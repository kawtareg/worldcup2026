from pathlib import Path
import pandas as pd
import joblib

from data import load_results, process_results, load_elo, process_elo
from features import matches_per_team, add_form_features, add_elo_features, add_results
from train import prepare_data, train_model
from simulate import predict_match, simulate_knockout_match, simulate_group

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
    qualifies = simulate_group(
        model, df_teams, df_elo,
        teams=['Brasil', 'Morocco', 'Scotland', 'Haiti'],
        dates=[pd.Timestamp('2026-06-15')] * 6
    )
    print(f"Qualified : {qualifies}")
if __name__ == "__main__":
    main()
