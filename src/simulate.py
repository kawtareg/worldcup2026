"""Functions to simulate FIFA World Cup 2026 matches and tournament outcomes."""

from features import get_elo, get_team_form
import pandas as pd

def predict_match(model, df_teams, df_elo, home_team, away_team, date):
    """Predict match outcome probabilities using the trained model.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe from matches_per_team.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
        date (pd.Timestamp): Date of the match.

    Returns:
        np.ndarray: Array of probabilities [P(W), P(D), P(L)].
    """
    home_form, home_wins = get_team_form(df_teams, home_team, date)
    away_form, away_wins = get_team_form(df_teams, away_team, date)
    elo_diff = get_elo(df_elo, home_team, date) - get_elo(df_elo, away_team, date)
    X = pd.DataFrame({
        'home_form': [home_form],
        'away_form': [away_form],
        'elo_diff': [elo_diff],
        'home_wins': [home_wins],
        'away_wins': [away_wins]
    })
    probas = model.predict_proba(X)
    return probas