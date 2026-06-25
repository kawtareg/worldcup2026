from pathlib import Path

from data import load_results, process_results, load_elo, process_elo
from features import matches_per_team, add_form_features, add_elo_features, add_results

df_elo = load_elo()
df_elo = process_elo(df_elo)
df_raw = load_results(force_download=True)
df = process_results(df_raw, df_elo)
df_teams = matches_per_team(df)
teams_path = Path(__file__).parent.parent / "data" / "processed" / "df_teams.csv"
df_teams.to_csv(teams_path, index=False)
df = add_form_features(df, df_teams, n=5, df_elo=df_elo)
df = add_elo_features(df, df_elo)
df = add_results(df)
path = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
df.to_csv(path, index=False)
