from pathlib import Path
import pandas as pd
import joblib

from data import load_results, process_results, load_elo, process_elo
from features import matches_per_team, add_form_features, add_elo_features, add_results
from train import prepare_data, train_model
from simulate import simulate_group_stage, build_predictions_df, simulate_group_stage
from simulate import resolve_bracket, ROUND_OF_32, simulate_tournament, monte_carlo

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

    n=1000
    results_mc = monte_carlo(model, df_teams, df_elo, n=n)
    for team, wins in results_mc[:10]:
        print(f"{team}: {wins/n:.1%}")

if __name__ == "__main__":
    main()
