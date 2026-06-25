"""Functions to load and preprocess football match results and ELO ratings."""

from pathlib import Path
import pandas as pd

def load_results(force_download=False):
    """Load international football match results from local storage or download from GitHub.

    Returns:
        pd.DataFrame: Raw match results with columns date, home_team, away_team,
            home_score, away_score, tournament, neutral.
    """
    path = Path(__file__).parent.parent / "data" / "raw" / "results.csv"
    if not path.exists() or force_download:
        df = pd.read_csv(
            'https://github.com/martj42/international_results/raw/refs/heads/master/results.csv')
        df = df[['date', 'home_team', 'away_team', 'home_score',
            'away_score', 'tournament', 'neutral']]
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    return df

def load_elo():
    """Load ELO ratings from local CSV file.

    Returns:
        pd.DataFrame: Raw ELO ratings.
    """
    path_to_elo = Path(__file__).parent.parent / "data" / "raw" / "elo_ratings_wc2026.csv"
    df_elo = pd.read_csv(path_to_elo)
    return df_elo

def process_elo(df_elo):
    """Clean ELO ratings dataframe.

    Args:
        df_elo (pd.DataFrame): Raw ELO ratings.

    Returns:
        pd.DataFrame: ELO ratings with parsed dates and relevant columns only.
    """
    df_elo = df_elo[['snapshot_date','country','rating']]
    df_elo['snapshot_date'] = pd.to_datetime(df_elo['snapshot_date'])
    return df_elo

def process_results(df, df_elo):
    """Clean and filter match results.

    Args:
        df (pd.DataFrame): Raw match results.
        df_elo (pd.DataFrame): ELO ratings used to filter teams.

    Returns:
        pd.DataFrame: Filtered matches with parsed dates, only completed matches
            between teams present in the ELO dataset, from 1930 onwards.
    """
    df = df.dropna(subset=['home_score', 'away_score'])
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] > '1930-01-01' ]
    countries = df_elo['country'].unique()
    df = df[df['home_team'].isin(countries) & df['away_team'].isin(countries)]
    return df
