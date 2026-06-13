"""Functions to simulate FIFA World Cup 2026 matches and tournament outcomes."""

import pandas as pd
import numpy as np
from features import get_elo, get_team_form
from itertools import combinations

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

def simulate_knockout_match(model, df_teams, df_elo, home_team, away_team, date):
    """Simulate a knockout match and return the winner.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
        date (pd.Timestamp): Date of the match.

    Returns:
        str: Name of the winning team.
    """
    probas = predict_match(model, df_teams, df_elo, home_team, away_team, date)
    p_home = probas[0][0] / (probas[0][0] + probas[0][2])
    p_away = probas[0][2] / (probas[0][0] + probas[0][2])
    winner = np.random.choice([home_team, away_team], p=[p_home, p_away])
    return winner

def simulate_group(model, df_teams, df_elo, teams, dates):
    """Simulate a group stage and return the two qualified teams.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.
        teams (list): List of 4 team names in the group.
        dates (list): List of 6 match dates.

    Returns:
        list: The two qualified teams sorted by points descending.
    """
    matchs = list(combinations(teams, 2))
    points = {team: 0 for team in teams}
    for i, match in enumerate(matchs):
        probas = predict_match(model, df_teams, df_elo, match[0], match[1], dates[i])
        result = np.random.choice(['W', 'D', 'L'], p=probas[0])
        if result == 'W':
            points[match[0]] += 3
        elif result == 'L':
            points[match[1]] += 3
        else:
            points[match[0]] += 1
            points[match[1]] += 1
    return sorted(points, key=lambda x: points[x], reverse=True)[:2]