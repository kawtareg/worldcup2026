"""Functions to engineer features for football match outcome prediction."""

import pandas as pd
import numpy as np

def matches_per_team(df):
    """Reshape match dataframe to have one row per team participation.

    Args:
        df (pd.DataFrame): Match results with home/away structure.

    Returns:
        pd.DataFrame: One row per team per match, with goals_scored, goals_conceded and opponent.
    """
    df1 = df[['date', 'home_team', 'home_score', 'away_score', 
            'tournament', 'neutral', 'away_team']].rename(
        columns={'home_team': 'team', 'home_score': 'goals_scored',
            'away_score': 'goals_conceded', 'away_team': 'opponent'})
    df2 = df[['date', 'away_team', 'away_score', 'home_score',
            'tournament', 'neutral', 'home_team']].rename(
        columns={'away_team': 'team', 'away_score': 'goals_scored',
            'home_score': 'goals_conceded', 'home_team': 'opponent'})
    return pd.concat([df1, df2]).sort_values('date')

def tournament_weighting(df):
    """Map tournament names to importance weights.

    Args:
        df (pd.DataFrame): Match dataframe with tournament column.

    Returns:
        pd.Series: Weight for each match based on tournament importance.
    """
    mappings = {
        'FIFA World Cup': 1.0,
        'UEFA Euro, Copa America': 0.85,
        'UEFA Nations League, AFCON': 0.7,
        'FIFA World Cup qualification': 0.6,
        'Friendly': 0.2
    }
    df['tournament_weighting'] = df['tournament'].map(mappings).fillna(0.1)
    return df['tournament_weighting']

def get_team_form(df, team, date, n=5, df_elo=None):
    """Compute weighted form metrics for a team over its last n matches.

    Args:
        df (pd.DataFrame): Team-level match dataframe from matches_per_team.
        team (str): Team name.
        date (pd.Timestamp): Reference date — only matches before this date are used.
        n (int): Number of recent matches to consider. Defaults to 5.
        df_elo (pd.DataFrame, optional): ELO ratings dataframe for opponent weighting.

    Returns:
        tuple: (goals_diff, wins, avg_scored, avg_conceded, clean_sheets)
    """
    filtered_df = df[(df['date'] < date) & (df['team'] == team)].tail(n).copy()
    weights = tournament_weighting(filtered_df)

    if df_elo is not None and 'opponent' in filtered_df.columns:
        mean_elo = df_elo['rating'].mean()
        opponent_weights = filtered_df['opponent'].apply(
            lambda opp: get_elo(df_elo, opp, date) / mean_elo
        )
        weights = weights * opponent_weights

    diff_per_match = filtered_df['goals_scored'] - filtered_df['goals_conceded']
    goals_diff = (diff_per_match * weights).sum()
    wins = (filtered_df['goals_scored'] > filtered_df['goals_conceded']).sum()
    avg_scored = filtered_df['goals_scored'].mean() if len(filtered_df) > 0 else 0
    avg_conceded = filtered_df['goals_conceded'].mean() if len(filtered_df) > 0 else 0
    clean_sheets = (filtered_df['goals_conceded'] == 0).sum()
    return goals_diff, wins, avg_scored, avg_conceded, clean_sheets

def get_elo(df_elo, team, date):
    """Get the most recent ELO rating for a team before a given date.

    Args:
        df_elo (pd.DataFrame): ELO ratings dataframe.
        team (str): Team name.
        date (pd.Timestamp): Reference date.

    Returns:
        int: Most recent ELO rating before the given date, or 1000 if not found.
    """
    filt_df_elo = df_elo[(df_elo['country'] == team) & (df_elo['snapshot_date']< date)]
    if filt_df_elo.empty:
        return 1000
    most_recent_elo = filt_df_elo.tail(1)
    return most_recent_elo['rating'].values[0]

def add_form_features(df, df_teams, n=5, df_elo=None):
    """Add home and away form features to match dataframe.

    Args:
        df (pd.DataFrame): Match results dataframe.
        df_teams (pd.DataFrame): Team-level dataframe from matches_per_team.
        n (int): Number of recent matches for form calculation. Defaults to 5.
        df_elo (pd.DataFrame, optional): ELO ratings for opponent weighting.

    Returns:
        pd.DataFrame: Match dataframe with form columns added.
    """
    df[['home_form', 'home_wins', 'home_avg_scored', 
        'home_avg_conceded', 'home_clean_sheets']] = df.apply(
        lambda row: get_team_form(df_teams, row['home_team'], row['date'], n, df_elo),
        axis=1, result_type='expand')
    df[['away_form', 'away_wins', 'away_avg_scored',
        'away_avg_conceded', 'away_clean_sheets']] = df.apply(
        lambda row: get_team_form(df_teams, row['away_team'], row['date'], n, df_elo),
        axis=1, result_type='expand')
    return df

def add_elo_features(df, df_elo):
    """Add ELO rating features to match dataframe.

    Args:
        df (pd.DataFrame): Match results dataframe.
        df_elo (pd.DataFrame): Processed ELO ratings dataframe.

    Returns:
        pd.DataFrame: Match dataframe with home_elo, away_elo and elo_diff columns added.
    """
    df['home_elo'] = df.apply(lambda row: get_elo(df_elo, row['home_team'], row['date']), axis=1)
    df['away_elo'] = df.apply(lambda row: get_elo(df_elo, row['away_team'], row['date']), axis=1)
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    return df

def add_results(df):
    """Add match result column from the home team perspective.

    Args:
        df (pd.DataFrame): Match results dataframe with home_score and away_score.

    Returns:
        pd.DataFrame: Match dataframe with result column — W, D or L.
    """
    conditions = [df['home_score'] > df['away_score'], df['home_score'] < df['away_score']]
    choices = ['W', 'L']
    df['result'] = np.select(conditions, choices, default='D')
    return df
