from data import load_results, process_results, load_elo, process_elo
from features import matches_per_team, add_form_features, add_elo_features, add_results
from train import prepare_data, train_model

def main():
    df_elo = load_elo()
    df_elo = process_elo(df_elo)
    df_raw = load_results()
    df = process_results(df_raw, df_elo)
    df_teams = matches_per_team(df)
    df = add_form_features(df, df_teams, n=5)
    df = add_elo_features(df, df_elo)
    df = add_results(df)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
