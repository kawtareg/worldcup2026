"""Functions to simulate FIFA World Cup 2026 matches and tournament outcomes."""

import pandas as pd
import numpy as np
from itertools import combinations
from features import get_elo, get_team_form

GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
    'B': ['Qatar', 'Switzerland', 'Canada', 'Bosnia and Herzegovina'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

ROUND_OF_32 = [
    ('A1', 'B2'), ('C1', 'F2'), ('E1', 'I1'), ('G1', 'D1'),
    ('H1', 'J2'), ('K1', 'L2'), ('B1', 'A2'), ('D2', 'G2'),
    ('F1', 'C2'), ('I2', 'E2'), ('J1', 'H2'), ('L1', 'K2'),
    ('T1', 'T2'), ('T3', 'T4'), ('T5', 'T6'), ('T7', 'T8'),
]

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
    p_home = float(probas[0][0])
    p_away = float(probas[0][2])
    total = p_home + p_away
    p_home = p_home / total
    p_away = p_away / total
    p_home = round(p_home, 10)
    p_away = round(p_away, 10)
    p_away = 1 - p_home
    winner = np.random.choice([home_team, away_team], p=[p_home, p_away])
    return winner

def simulate_group(model, df_teams, df_elo, teams, date):
    """Simulate a group stage and return the two qualified teams.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.
        teams (list): List of 4 team names in the group.
        date (list): List of 6 match dates.

    Returns:
        list: The two qualified teams sorted by points descending.
    """
    matches = list(combinations(teams, 2))
    points = {team: 0 for team in teams}
    for match in matches:
        probas = predict_match(model, df_teams, df_elo, match[0], match[1], date)
        result = np.random.choice(['W', 'D', 'L'], p=probas[0])
        if result == 'W':
            points[match[0]] += 3
        elif result == 'L':
            points[match[1]] += 3
        else:
            points[match[0]] += 1
            points[match[1]] += 1
    qualifies = sorted(points, key=lambda x: points[x], reverse=True)
    return qualifies[0], qualifies[1], qualifies[2], points[qualifies[2]]

def simulate_group_stage(model, df_teams, df_elo):
    """Simulate all 12 group stages and return qualified teams and best third-place finishers.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.

    Returns:
        tuple: 
            - results (dict): Maps group letter to (first, second) qualified teams.
            - best_thirds (list): 8 best third-place finishers as (group, (team, points))
            sorted by points.
    """
    results = {}
    thirds = {}

    for group, teams in GROUPS.items():
        first, second, third, points_third = simulate_group(
            model, df_teams, df_elo, teams, pd.Timestamp('2026-06-11'))
        results[group] = (first, second)
        thirds[group] = (third, points_third)

    best_thirds = sorted(thirds.items(), key=lambda x: x[1][1], reverse=True)[:8]
    return results, best_thirds

def get_predicted_winner(probas, home_team, away_team):
    """Determine the predicted winner from model output probabilities.

    Args:
        probas (np.ndarray): Output of predict_match — shape (1, 3) with [P(W), P(D), P(L)].
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.

    Returns:
        str: Predicted winner — home_team, away_team, or 'Draw'.
    """
    idx = np.argmax(probas[0])
    if idx == 0:
        return home_team
    if idx == 1:
        return 'Draw'
    return away_team

def get_actual_result(row):
    """Extract the actual match result from a DataFrame row.

    Args:
        row (pd.Series): A row from the matches DataFrame with home_score and away_score.

    Returns:
        str or None: Actual result — home_team, away_team, 'Draw', or None if not played yet.
    """
    if pd.isna(row['home_score']):
        return None
    if row['home_score'] > row['away_score']:
        return row['home_team']
    if row['home_score'] < row['away_score']:
        return row['away_team']
    return 'Draw'

def compare_results(predicted, actual):
    """Compare predicted and actual results.

    Args:
        predicted (str): Predicted result.
        actual (str or None): Actual result, None if match not played yet.

    Returns:
        bool or None: True if correct, False if wrong, None if match not played.
    """
    if actual is None:
        return None
    if actual == predicted: 
        return True
    return False

def build_predictions_df(model, df_raw, df_teams, df_elo):
    """Build a DataFrame with predictions and actual results for all 2026 World Cup matches.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_raw (pd.DataFrame): Raw match results including future matches.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.

    Returns:
        pd.DataFrame: Predictions with columns home_team, away_team, date,
            P(W), P(D), P(L), predicted, actual, correct.
    """
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    wc_matches = df_raw[(df_raw['tournament'] == 'FIFA World Cup') &
        (df_raw['date'].dt.year == 2026)].copy()
    rows = []
    for _, row in wc_matches.iterrows():
        probas = predict_match(model, df_teams, df_elo, row['home_team'], row['away_team'],
            row['date'])
        predicted = get_predicted_winner(probas, row['home_team'], row['away_team'])
        actual = get_actual_result(row)
        correct = compare_results(predicted, actual)
        rows.append({
        'home_team': row['home_team'],
        'away_team': row['away_team'],
        'date': row['date'],
        'P(W)': probas[0][0],
        'P(D)': probas[0][1],
        'P(L)': probas[0][2],
        'predicted': predicted,
        'actual': actual,
        'correct': correct
    })
    return pd.DataFrame(rows)

def evaluate_predictions(predictions_df):
    """Compute accuracy of predictions on played matches.

    Args:
        predictions_df (pd.DataFrame): Output of build_predictions_df.

    Returns:
        float: Proportion of correct predictions on played matches.
    """
    played = predictions_df[predictions_df['actual'].notna()]
    accuracy = played['correct'].mean()
    return accuracy

def resolve_team(code, results, best_thirds):
    """Resolve a bracket code to a team name.

    Args:
        code (str): Bracket code like 'A1', 'B2', or 'T1'.
        results (dict): Maps group letter to (first, second) qualified teams.
        best_thirds (list): Sorted list of best third-place finishers as (group, (team, points)).

    Returns:
        str: Team name corresponding to the bracket code.
    """
    if code[1] == '3' or code[0] == 'T':
        rank = int(code[1]) - 1
        return best_thirds[rank][1][0]
    else:
        group = code[0]
        rank = int(code[1]) - 1
        return results[group][rank]

def resolve_bracket(bracket, results, best_thirds):
    """Resolve all bracket codes to real team names.

    Args:
        bracket (list): List of (home_code, away_code) tuples.
        results (dict): Maps group letter to (first, second) qualified teams.
        best_thirds (list): Sorted list of best third-place finishers.

    Returns:
        list: List of (home_team, away_team) tuples with real team names.
    """
    matches = []
    for home_code, away_code in bracket:
        home = resolve_team(home_code, results, best_thirds)
        away = resolve_team(away_code, results, best_thirds)
        matches.append((home, away))
    return matches

def simulate_knockout_stage(model, df_teams, df_elo, matches, date):
    """Simulate one knockout round and return the winners.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.
        matches (list): List of (home_team, away_team) tuples.
        date (pd.Timestamp): Date used for feature calculation.

    Returns:
        list: List of winning team names.
    """
    winners = []
    for match in matches:
        winner = simulate_knockout_match(model, df_teams, df_elo, match[0], match[1], date)
        winners.append(winner)
    return winners

def simulate_tournament(model, df_teams, df_elo, matches, date):
    """Simulate the full knockout tournament from round of 32 to the final.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.
        matches (list): List of (home_team, away_team) tuples for the round of 32.
        date (pd.Timestamp): Date used for feature calculation.

    Returns:
        tuple:
            - history (dict): Maps round name to list of winners.
            - winner (str): Name of the tournament winner.
    """
    rounds = ['R32', 'R16', 'QF', 'SF', 'F']
    history = {}
    for round_name in rounds:
        winners = simulate_knockout_stage(model, df_teams, df_elo, matches, date)
        history[round_name] = winners
        if round_name == 'F':
            break
        matches = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
    return history, winners[0]

def monte_carlo(model, df_teams, df_elo, n=10000):
    """Run Monte Carlo simulations to estimate each team's probability of winning the World Cup.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        df_teams (pd.DataFrame): Team-level match dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.
        n (int): Number of simulations to run. Defaults to 10000.

    Returns:
        list: List of (team, wins) tuples sorted by wins descending.
    """
    wins = {team: 0 for team in [t for teams in GROUPS.values() for t in teams]}
    for i in range(n):
        results, best_thirds = simulate_group_stage(model, df_teams, df_elo)
        matches = resolve_bracket(ROUND_OF_32, results, best_thirds)
        date = pd.Timestamp('2026-06-29')
        _, winner = simulate_tournament(model, df_teams, df_elo, matches, date)
        wins[winner]+=1
    return sorted(wins.items(), key=lambda x: x[1], reverse=True)
